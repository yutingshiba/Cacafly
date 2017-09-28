import sys
import os
import time
import json
import h5py
import pickle
import configparser
import numpy as np

class DataManager:
    def __init__(self, argv):
        self.dicFile = {}
        self.dicPar = {}

        self.readConfigs(argv)
        self.path = self.dicFile['path']
        self.train_path = self.path+self.dicFile['train_file']
        self.dev_path = self.path+self.dicFile['dev_file']
        self.test_path = self.path+self.dicFile['test_file']
#        self.save_epoch_path = self.check_path(self.path+self.dicFile['save_epoch_weight']+'_{}e.h5'.format(self.dicPar['max_epoch']))
#        self.save_final_path = self.check_path(self.path+self.dicFile['save_final_weight']+'_{}e.h5'.format(self.dicPar['max_epoch']))
#        self.save_predict_path = self.check_path(self.path+self.dicFile['save_prediction']+'_{}e.pkl'.format(self.dicPar['max_epoch']))
        self.save_epoch_path = self.check_path(self.path+self.dicFile['save_epoch_weight'])
        self.save_final_path = self.check_path(self.path+self.dicFile['save_final_weight'])
        self.save_predict_path = self.check_path(self.path+self.dicFile['save_prediction'])
        
        self.uDic = self.load_json(self.dicFile['path']+self.dicFile['user_dic'])
        self.uSize = len(self.uDic)
        
        t = time.time()
        self.vDic, self.word_emb = self.load_embedding(self.dicFile['embedding_file'], self.dicPar['v_dim'])
        self.vSize = len(self.vDic)
        print('Load embedding elapse:', time.time()-t)
## ------------------------------------------------------------
#        t = time.time()
#        titles = self.load_file(self.dicFile['title'])
#        self.title_padded = self.pad_data(titles, self.dicPar['max_title'], self.vDic)
#        print(self.title_padded[-1])
#        print('Load title elapse:', time.time()-t)

#        t = time.time()
#        topics = self.load_file(self.dicFile['topic'])
#        self.topic_padded = self.pad_data(topics, self.dicPar['max_topic'])
#        print(self.topic_padded[-1])
#        print('Load topic elapse:', time.time()-t)

        t = time.time()
        contents = self.load_file(self.dicFile['content'])
        self.content_padded = self.pad_data(contents, self.dicPar['max_content'], self.vDic)
#        print(self.content_padded[-1])
#        sys.exit()
        print('Load content elapse:', time.time()-t)
## -----------------------------------------------------------------------------------
    def check_path(self, file_path):
        if os.path.exists(file_path):
            print('Warning!!!', file_path, 'already exist!')
            sys.exit()
        return file_path

    def load_file(self, file_path):
        lines = []
        with open(file_path) as fin:
            for line in fin:
                lines.append(line)
        return lines
    
    def pad_data(self, data, max_length, dic=None):
        data_padded = np.zeros((len(data)+1, max_length))
        for idx, line in enumerate(data, 1):
            line = line.split(' ')
            if len(line) > max_length:
                # select the first max_length words
                line = line[:max_length]
            if dic:
                for wid, word in enumerate(line):
                    if word in dic:
                        data_padded[idx][wid] = dic[word]
                    else:
                        data_padded[idx][wid] = dic['<unk>']
            else: # case of topics
                for wid, word in enumerate(line):
                    data_padded[idx][wid] = int(word)+1 # +1 because topic id range is 0-99
        return data_padded

    def pad_users(self, user_list, max_length, user_dic):
        user_padded = np.zeros((max_length))
        if len(user_list) > max_length:
            user_list = user_list[-max_length:]
        for idx, uid in enumerate(user_list):
            if str(uid) in user_dic:
                user_padded[idx] = user_dic[str(uid)]
            else:
                user_padded[idx] = user_dic['<unk>']
        return user_padded

    def readConfigs(self, sys_argv):
        argLen = len(sys_argv)
        if argLen!=2:
            print('Error!')
            print('Input example: python UTCNN_release.py config.ini')
            sys.exit()
        config = configparser.ConfigParser()
        config.read(str(sys_argv[1]))
        sections = config.sections()

        def num(s):
            try:
                return int(s)
            except ValueError:
                return float(s)

        for section in sections:
            options = config.options(section)
            for option in options:
                try:
                    if section == 'Files':
                        self.dicFile[option] = config.get(section, option)
                    else:
                        self.dicPar[option] = num(config.get(section, option))
                except Exception as e:
                    print('Config syntax error:',section, option)
                    print(e)
                    sys.exit()

    def load_h5py(self, file_path):
        t_load = time.time()
        with h5py.File(file_path, 'r') as infile:
            labels = infile['labels'][:]
            r_aid = infile['r_aid'][:]
            c_aid = infile['c_aid'][:]
            c_user = infile['c_user'][:]
#            print(labels[-3])
#            print(r_aid[-3])
#            print(c_aid[-3])
#            print(c_user[-3])
            
            r_users = []
            r_users_raw = infile['r_users'][:]
            for ru_string in r_users_raw:
                r_user = ru_string.decode('utf-8').strip('%').split('%')
                r_users.append(r_user)
#            print(r_users[-3])
            
            u_aids = []
            u_aids_raw = infile['u_aids'][:]
            for ua_string in u_aids_raw:
                u_aid = ua_string.decode('utf-8').strip('%').split('%')
                u_aids.append(u_aid)
#            print(u_aids[-3])
 #           sys.exit()

        print('load', file_path,'elapse: ', time.time()-t_load)
        return labels, r_aid, c_aid, c_user, r_users, u_aids

    def load_pickle(self, file_path):
        t_load = time.time()
        with open(file_path, 'rb') as infile:
            data = pickle.load(infile)
        print('load', file_path, 'elapse: ', time.time()-t_load)
        return data

    def load_json(self, file_path):
        t_load = time.time()
        with open(file_path) as infile:
            data = json.load(infile)
        print('load', file_path, 'elapse: ', time.time()-t_load)
        return data

    def load_embedding(self, Embfile, dim):
        print('Load embedding file:', Embfile)
        try:
            with open(Embfile) as f:
                lines = [line.rstrip('\n') for line in f]
        except Exception as e:
            print('Input error',Embfile)
            print(e)
            sys.exit()

        voc_dic = {}
        emb = []
        emb.append(np.zeros(dim))
        voc_dic['<zero>'] = 0
        for idx, line in enumerate(lines,1):
            tokens = line.split(' ')
            embedding = np.array(tokens[1:])
            emb.append(embedding.astype(np.float))
            voc_dic[tokens[0]] = idx
        if len(voc_dic) != len(emb):
            print('voc_dic size not matching embedding size!')
            sys.exit()
        return voc_dic, np.array(emb)

    def batch_generator(self, nb_batch, batch_size, r_aid, r_users, c_aid, c_user, u_aids=None, labels=None):
        maxUArticles = self.dicPar['max_uarticles']
        C_user = []
        C_content = []
        R_users = []
        R_content = []
#        R_topics = []
        U_titles = []

        for aid in r_aid:
            R_content.append(self.content_padded[aid])
#            R_topics.append(self.topic_padded[aid])
        for users in r_users:
            R_users.append(self.pad_users(users, self.dicPar['max_rusers'], self.uDic))
        for aid in c_aid:
            C_content.append(self.content_padded[aid])
        for user in c_user:
            if str(user) in self.uDic:
                C_user.append(self.uDic[str(user)])
            else:
                C_user.append(self.uDic['<unk>'])
        if maxUArticles > 0:
            for aids in u_aids:
                article_titles = np.zeros((maxUArticles, self.dicPar['max_title']))
                if len(aids) > maxUArticles:
                    aids = aids[:maxUArticles]
                for idx, aid in enumerate(aids):
                    article_titles[idx] = self.title_padded[aid[1]]
                article_titles = article_titles.reshape((-1,))
                U_titles.append(article_titles)

        if labels is not None:
            # shuffle data
            from sklearn.utils import shuffle
            if maxUArticles > 0:
#                labels, C_user, C_content, R_users, R_title, R_topics, U_titles \
#                    = shuffle(labels, C_user, C_content, R_users, R_title, R_topics, U_titles)
                labels, C_user, C_content, R_content, U_titles \
                    = shuffle(labels, C_user, C_content, R_content, U_titles)
            else:
#                labels, C_user, C_content, R_users, R_title, R_topics \
#                    = shuffle(labels, C_user, C_content, R_users, R_title, R_topics)
                labels, C_user, C_content, R_content \
                    = shuffle(labels, C_user, C_content, R_content)

        while 1:
            for i in range(nb_batch):
                if maxUArticles > 0:
                    inputs=[
                        np.array(C_user[i*batch_size:(i+1)*batch_size]),
                        np.array(C_content[i*batch_size:(i+1)*batch_size]), 
                        np.array(R_users[i*batch_size:(i+1)*batch_size]), 
                        np.array(R_content[i*batch_size:(i+1)*batch_size]), 
#                        np.array(R_topics[i*batch_size:(i+1)*batch_size]),
#                        np.array(U_titles[i*batch_size:(i+1)*batch_size])
                    ]
                else:
                    inputs=[
                        np.array(C_user[i*batch_size:(i+1)*batch_size]),
                        np.array(C_content[i*batch_size:(i+1)*batch_size]), 
                        np.array(R_users[i*batch_size:(i+1)*batch_size]), 
                        np.array(R_content[i*batch_size:(i+1)*batch_size]), 
#                        np.array(R_topics[i*batch_size:(i+1)*batch_size]),
                    ]
                if labels is not None:
                    outputs=labels[i*batch_size:(i+1)*batch_size]
                    yield (inputs, outputs)
                else:
                    yield (inputs)

    def find_trues(self, y_true, y_pred):
#        result = np.zeros(len(y_true))
#        for idx, pred in y_pred:
#            if pred == y_true[idx]:
#                result[idx]=1
        result = abs(-1 + (y_true^y_pred))
        return result

#if __name__ == '__main__':
#    data_manager = DataManager(sys.argv)
#    print(data_manager.dicPar['topic_size'])

