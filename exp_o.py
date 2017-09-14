#========================================================
# Origin: exp_n.py
# Change:
# 1. change optimizer: adagram -> nadam
# 2. yield for loop: batch_size -> samples_per_epoch
# 3. add 'topic_size = 100' to config_o.ini
# 4. change "# padding topic" section
# 5. change dense layer in "build_model": Dense(1,sigmoid)
# 5.1. remove: to_categorical(labels,2) in "batch_generator_train"
# 6. add function: f1, import keras backend as K
# 7. add function: check_path
#========================================================
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import numpy as np
import time
import tensorflow as tf
import pickle
import math
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support

from data_manager import DataManager
from build_model import BuildModel

def debug(data):
    rusers_path = data.dicFile['rusers_path']+'_test.pkl'
    uaids_path = data.dicFile['uaids_path']+'_test.pkl'
    labels, r_aid, c_aid, c_user = data.load_h5py(data.dicFile['test_data_path'])
    r_users = data.load_pickle(rusers_path)
    u_aids = data.load_pickle(uaids_path)
    print(type(labels[-1])) #int
    print(type(r_aid[-1])) # int
    print(type(r_users[-1][0])) # list
    print(type(c_aid[-1])) # int
    print(type(c_user[-1])) # int
    print(type(u_aids[-1][0])) # list: [(time, aid), (time, aid)]
    

if __name__ == '__main__':
    # tenserflow setting
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    gpu_options = tf.GPUOptions(allow_growth=True)
    sess = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options))

    data = DataManager(sys.argv)
    
    t = time.time()
    model = BuildModel(data.dicPar, data.vSize, data.word_emb, data.uSize)
    model.print_params()
    my_model = model.build()
    print('Build model Elapse: ', time.time()-t)

    train_labels, train_r_aid, train_c_aid, train_c_user = data.load_h5py(data.train_path)
    dev_labels, dev_r_aid, dev_c_aid, dev_c_user = data.load_h5py(data.dev_path)
    test_labels, test_r_aid, test_c_aid, test_c_user = data.load_h5py(data.test_path)
    
    train_r_users, train_u_aids, dev_r_users, dev_u_aids, test_r_users, test_u_aids = data.load_all_pickle()


    nb_train = len(train_labels)
    batch_size_train = data.dicPar['batch_size']
    nb_batch_train = math.ceil(nb_train/batch_size_train)
    
    nb_dev = len(dev_labels)
    nb_test = len(test_labels)
    batch_size_dev = data.dicPar['batch_size_dev']
    batch_size_test = data.dicPar['batch_size_test']
    nb_batch_dev = math.ceil(nb_dev/batch_size_dev)
    nb_batch_test = math.ceil(nb_test/batch_size_test)
    
    print('nb_train, nb_dev, nb_test')
    print(nb_train, nb_dev, nb_test)

    # Fit model
    t = time.time()
    from keras.callbacks import EarlyStopping, ModelCheckpoint
#    save_epoch = ModelCheckpoint(data.save_epoch_path)
    my_model.fit_generator(
        data.batch_generator(
            nb_batch_train,
            batch_size_train,
            train_r_aid,
            train_r_users,
            train_c_aid,
            train_c_user,
            u_aids=train_u_aids,
            labels=train_labels
        ),
        steps_per_epoch = nb_batch_train,
        epochs=data.dicPar['max_epoch'],
#        callbacks=[stop,save],
#        callbacks=[save_epoch],
        validation_data = data.batch_generator(
            nb_batch_dev,
            batch_size_dev, 
            dev_r_aid, 
            dev_r_users, 
            dev_c_aid, 
            dev_c_user,
            u_aids=dev_u_aids,
            labels=dev_labels, 
        ),
        validation_steps = nb_batch_dev
#        validation_steps = 2
    )
    print('Fitting Elapse: ', time.time()-t)
#    my_model.save_weights(data.save_final_path)

    print('Predict test data')
    classes = my_model.predict_generator(
       data.batch_generator(
            nb_batch_test,
            batch_size_test,
            test_r_aid, 
            test_r_users, 
            test_c_aid, 
            test_c_user,
            u_aids=test_u_aids
        ),
        nb_batch_test,
        verbose=1
    )
    
#    # Load weight
#    weight_path = data.path+'h5/itr499_weight.h5'
#    from keras.models import load_model
#    my_model.load_weights(weight_path) 

#    print(max(label for label in classes))
#    print(min(label for label in classes))
#    print(sum(1 for label in classes if label > 0.5))
#    print(sum(1 for label in classes if label > 0.6))
#    print(sum(1 for label in classes if label > 0.7))
#    print(sum(label for label in classes)/len(classes))
#    with open(check_path(data.save_predict_path),'wb') as outfile:
#        pickle.dump(classes, outfile)

    classes = classes.reshape(-1)
    prediction = classes.round().astype(int)
    print(confusion_matrix(train_labels, prediction))
    print(precision_recall_fscore_support(train_labels, prediction))
