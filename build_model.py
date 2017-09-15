import sys
import numpy as np
from keras.models import Model
#from keras.models import load_model
from keras.layers import Embedding, Input
from keras.layers.merge import Dot, Concatenate, Average
from keras.layers.core import Lambda, Reshape, Dense
from keras.layers.pooling import MaxPooling1D, MaxPooling2D
from keras.layers.convolutional import Conv1D, Conv2D
from keras.utils.np_utils import to_categorical
from keras.optimizers import Adagrad, Nadam
from keras import metrics
from keras import initializers
from keras import backend as K

class BuildModel():
    def __init__(self, dicPar, voc_size, word_embeddings, user_size):
        self.load_dicPar(dicPar)
        self.vSize = voc_size
        self.word_emb = word_embeddings
        self.uSize = user_size

    def load_dicPar(self, dicPar):
        # Load parameters
        self.vDim = dicPar['v_dim']
        self.uDim = dicPar['u_dim']
        self.tDim = dicPar['t_dim']
        self.mini_uDim = dicPar['mini_u_dim']
        self.mini_tDim = dicPar['mini_t_dim']
        self.rndBase = dicPar['rnd_base']
        self.flength1 = dicPar['flength1']
        self.flength2 = dicPar['flength2']
        self.flength3 = dicPar['flength3']
        self.conSize = dicPar['con_size']
#        self.lSize = dicPar['l_size']
        self.lr = dicPar['lr']
        self.tSize = dicPar['topic_size']
        self.maxContentLen = dicPar['max_content']
        self.maxTitleLen = dicPar['max_title']
        self.maxTopics = dicPar['max_topic']
        self.maxCUser = dicPar['max_cuser']
        self.maxRUsers = dicPar['max_rusers']
        self.maxUArticles = dicPar['max_uarticles']

    def build_embeddings(self):
        min_bound = -self.rndBase
        max_bound = self.rndBase
        self.user_vector_emb = np.random.uniform(min_bound, max_bound, size=(self.uSize, self.uDim))
        self.user_matrix_emb = np.random.uniform(min_bound, max_bound, size=(self.uSize, self.vDim*self.mini_uDim))
#        self.word_emb = word_embeddings
        self.topic_matrix_emb = np.random.uniform(min_bound, max_bound, size=(self.tSize, self.vDim*self.mini_tDim))

#    def convolution(self, input_length, input_width, input_matrix):
    def convolution(self, input_matrix):
        min_bound = -self.rndBase
        max_bound = self.rndBase
        kernel_size1 = self.flength1
        kernel_size2 = self.flength2
        kernel_size3 = self.flength3
        con_size = self.conSize
        input_length = int(input_matrix.shape[1])    #input_matrix.shape=(?,len,wid)
        input_width = int(input_matrix.shape[2])

        r1 = np.random.uniform(min_bound, max_bound, size=(kernel_size1,input_width,con_size)) 
        r2 = np.random.uniform(min_bound, max_bound, size=(kernel_size2,input_width,con_size)) 
        r3 = np.random.uniform(min_bound, max_bound, size=(kernel_size3,input_width,con_size)) 
        rb = np.random.uniform(min_bound, max_bound, size=(con_size,))
        con1 = Conv1D(con_size, kernel_size1, weights=[r1,rb], activation='tanh')(input_matrix)
        con2 = Conv1D(con_size, kernel_size2, weights=[r2,rb], activation='tanh')(input_matrix)
        con3 = Conv1D(con_size, kernel_size3, weights=[r3,rb], activation='tanh')(input_matrix)
        c1Avg = MaxPooling1D(pool_size=(input_length-kernel_size1+1))(con1)
        c2Avg = MaxPooling1D(pool_size=(input_length-kernel_size2+1))(con2)
        c3Avg = MaxPooling1D(pool_size=(input_length-kernel_size3+1))(con3)
        c1_reshape = Reshape((con_size,))(c1Avg)
        c2_reshape = Reshape((con_size,))(c2Avg)
        c3_reshape = Reshape((con_size,))(c3Avg)
        avg_convs = Average()([c1_reshape,c2_reshape,c3_reshape])
        return avg_convs

    def build(self):
        # Load embeddings
        self.build_embeddings()
        
        # Current user representation (user vector)
        input_CUser = Input(shape=(self.maxCUser,), dtype='int32')
        CUV = Embedding(
            input_dim = self.uSize, 
            output_dim = self.uDim, 
            input_length = self.maxCUser, # only one current user
            weights = [self.user_vector_emb]
        )(input_CUser)
        cuser_representation = Reshape((self.uDim,))(CUV)
        print('shape of user_representation', cuser_representation.shape)
        
        # Current article representation (content vector)
        input_CContent = Input(shape=(self.maxContentLen,),dtype='int32')
        CCV = Embedding(
            input_dim = self.vSize,
            output_dim = self.vDim,
            input_length = self.maxContentLen,
            weights = [self.word_emb],
            trainable = False
        )(input_CContent)
#        ccontent_pooling = MaxPooling1D(pool_size=(self.maxContentLen))(CCV) # shape=(-1,1,50)
#        ccontent_representation = Reshape((self.vDim,))(ccontent_pooling)
        print(CCV.shape)
        ccontent_representation = self.convolution(CCV)
        print('shape of current title representation', ccontent_representation.shape)

        # Recommended articles representation
        
            # RUsers matrix
        input_RUsers = Input(shape=(self.maxRUsers,), dtype='int32')
        RUM = Embedding(
            input_dim = self.uSize, 
            output_dim = self.vDim*self.mini_uDim, 
            input_length = self.maxRUsers, 
            weights = [self.user_matrix_emb]
        )(input_RUsers)
        rusers_matrix_pooling = MaxPooling1D(pool_size=(self.maxRUsers))(RUM)
        rusers_representation = Reshape((self.vDim, self.mini_uDim))(rusers_matrix_pooling)
        #print('shape of recommend users representation', rusers_representation.shape)
        
            # RTopics matrix
        input_RTopics = Input(shape=(self.maxTopics,),dtype='int32')
        RTopicsM = Embedding(
            input_dim = self.tSize,
            output_dim = self.vDim*self.mini_tDim,
            input_length = self.maxTopics,
            weights = [self.topic_matrix_emb]
        )(input_RTopics)
        rtopics_pooling = MaxPooling1D(pool_size=(self.maxTopics))(RTopicsM)
        rtopics_representation = Reshape((self.vDim, self.mini_tDim))(rtopics_pooling)
        #print('shape of recommend topics representation', rtopics_representation.shape)

            # content word embedding
        input_RTitle = Input(shape=(self.maxTitleLen,), dtype='int32')
        RTitle = Embedding(
            input_dim = self.vSize,
            output_dim = self.vDim,
            input_length = self.maxTitleLen,
            weights=[self.word_emb],
            trainable = False
        )(input_RTitle)
        
            # user_matrix . content
        #print('RTitle shape',RTitle.shape)  #(?,40,50)
        dot_rtitle_rusers = Dot(axes=(2,1))([RTitle, rusers_representation])
        
            # content . topic_matrix
        dot_rtitle_rtopics = Dot(axes=(2,1))([RTitle, rtopics_representation])
    
            # (user_matrix . content) + (content . topic_matrix)
        r_title_user_topics = Concatenate()([dot_rtitle_rusers, dot_rtitle_rtopics])
        RTUT = Reshape((self.maxTitleLen,self.mini_uDim+self.mini_tDim))(r_title_user_topics)
        
        RT = Reshape((self.maxTitleLen, self.mini_tDim))(dot_rtitle_rtopics)
        RArticle_representation = self.convolution(RT)
        print('shape of RArticle_representation', RArticle_representation.shape)
         
        '''
        # User history representation
            # current user matrix
        CUM = Embedding(
            input_dim = self.uSize, 
            output_dim = self.vDim*self.mini_uDim, 
            input_length = self.maxCUser, 
            weights = [self.user_matrix_emb]
        )(input_CUser)
        cuser_matrix = Reshape((self.vDim, self.mini_uDim))(CUM)

            # history articles representation
        input_UArticles = Input((self.maxUArticles*self.maxTitleLen,))
        title_emb_layer = Embedding(
            input_dim = self.vSize,
            output_dim = self.vDim,
            input_length = self.maxUArticles*self.maxTitleLen,
            weights=[self.word_emb], 
            trainable = False
        )(input_UArticles)
        title_reshape = Reshape((self.maxUArticles, self.maxTitleLen, self.vDim))(title_emb_layer)
        title_pooling = MaxPooling2D(pool_size=(1, self.maxTitleLen))(title_reshape)
        title_repre = Reshape((self.maxUArticles, self.vDim))(title_pooling)  # shape:(maxUArticles, vDim)

            # histories . CU + histories . RT
        HU = Dot(axes=(2,1))([title_repre,cuser_matrix]) #shape:(maxUArticles, mini_uDim)
        HT = Dot(axes=(2,1))([title_repre,rtopics_representation]) #shape:(maxUArticles, mini_tDim)
        HUT = Concatenate()([HU,HT])
        HUT_reshape = Reshape((self.maxUArticles, self.mini_uDim+self.mini_tDim))(HUT)
        
            # maxpooling
#        HUT_pooling = MaxPooling1D(pool_size=(maxUArticles))(HUT_reshape)
#        user_history_representation = Reshape((mini_uDim+mini_tDim,))(HUT_pooling)
#        print('shape of user_history_representation', user_history_representation.shape)
        
            # convolution
        user_history_representation = self.convolution(HUT_reshape)
        print('shape of user_history_representation', user_history_representation.shape)
        '''
        # Concatenate all representations
        final = Concatenate()([
            cuser_representation,
            ccontent_representation,
            RArticle_representation,
#            user_history_representation
        ])
        print('shape of final_layer', final.shape)
    #    rD  = np.random.uniform(low=-rndBase,high=rndBase,size=(uDim+vDim+conSize+mini_uDim+mini_tDim,lSize))
    #    rD  = np.random.uniform(low=-rndBase,high=rndBase,size=(uDim+vDim+conSize,lSize))
    #    rDb = np.random.uniform(low=-rndBase,high=rndBase,size=(lSize,))
    #    predict = Dense(lSize,activation='softmax',weights=[rD,rDb])(final)
#        predict = Dense(1,activation='sigmoid',bias_initializer=initializers.RandomUniform())(final)
        predict = Dense(1, activation='sigmoid')(final)
        print('shape of predict', predict.shape)

        # Compile model
        model = Model(
#            inputs=[input_CUser,input_CContent,input_RUsers,input_RTitle,input_RTopics,input_UArticles], 
            inputs=[input_CUser,input_CContent,input_RUsers,input_RTitle,input_RTopics], 
#            inputs=[input_CContent,input_RTitle,input_RTopics], 
            outputs=predict
        )
    #    ag = Adagrad(lr)
    #    model.compile(ag, 'categorical_crossentropy',['accuracy'])
    #    model.compile(ag, 'binary_crossentropy',['accuracy'])
        model.compile(optimizer=Nadam(lr=self.lr), loss='binary_crossentropy', metrics=[self.f1, 'acc'])
        return model

    def f1(self, y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predict_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        if true_positives == 0:
            return 0
        if predict_positives == 0:
            return 0
        if positives == 0:
            return 0
        precision = true_positives/predict_positives
        recall = true_positives/positives
        f1_score = 2 * (precision * recall) / (precision + recall)
        return f1_score

    def print_params(self): 
        print('uSize, vSize, tSize')
        print(self.uSize, self.vSize, self.tSize)
        print('maxContentLen, maxTitleLen, maxTopics')
        print(self.maxContentLen, self.maxTitleLen, self.maxTopics)
        print('maxCUser, maxRUsers, maxUArticles')
        print(self.maxCUser, self.maxRUsers, self.maxUArticles)
