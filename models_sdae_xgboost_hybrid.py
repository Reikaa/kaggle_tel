from models_deeplearn import SDAE
from theano import function, config, shared
import theano.tensor as T
import numpy as np
from math import ceil

def get_shared_data(data_ixy):
    if len(data_ixy)==3:
        data_ids,data_x,data_y = data_ixy
    elif len(data_ixy)==2:
        data_ids,data_x = data_ixy
        data_y = None

    id_tensor = shared(value=np.asarray(data_ids,dtype=config.floatX),borrow=True)
    id_int_tensor =  T.cast(id_tensor,'int32')

    shared_x = shared(value=np.asarray(data_x,dtype=config.floatX),borrow=True)

    if data_y is not None:
        shared_y = shared(value=np.asarray(data_y,dtype=config.floatX),borrow=True)
        return id_int_tensor,shared_x,T.cast(shared_y,'int32')
    else:
        return id_int_tensor,shared_x

def tensor_divide_test_valid(train_data):

    '''
            0th class 6.58, 1st class 2.58, 2nd class 1 (ratios)
    '''
    import csv

    my_train_ids = []
    my_valid_ids = []
    my_train_ids_v2 = [[],[],[]]

    data_x = []
    data_y = []
    valid_x,valid_y = [],[]
    data_x_v2 = [[],[],[]]

    th_tr_ids,th_tr_x, th_tr_y = train_data
    tr_ids = th_tr_ids.eval()
    tr_x = th_tr_x.get_value(borrow=True)
    tr_y = th_tr_y.eval()

    for i in range(len(tr_x)):
        # first 2 columns are ID and location
        output = int(tr_y[i])
        data_x_v2[output].append(tr_x[i])
        my_train_ids_v2[output].append(tr_ids[i])

    valid_size = 350
    full_rounds = 1
    orig_class_2_length = len(data_x_v2[2])
    for _ in range(orig_class_2_length):
        rand = np.random.random()
        if rand>=0.9 or len(valid_x)>valid_size:
            for _ in range(6) :
                data_x.append(data_x_v2[0][-1])
                data_x_v2[0].pop()
                data_y.append(0)
                my_train_ids.append(my_train_ids_v2[0][-1])
                my_train_ids_v2[0].pop()

            for _ in range(2):
                data_x.append(data_x_v2[1][-1])
                data_x_v2[1].pop()
                data_y.append(1)
                my_train_ids.append(my_train_ids_v2[1][-1])
                my_train_ids_v2[1].pop()

            data_x.append(data_x_v2[2][-1])
            data_x_v2[2].pop()
            data_y.append(2)
            my_train_ids.append(my_train_ids_v2[2][-1])
            my_train_ids_v2[2].pop()
            full_rounds += 1

        elif len(valid_x)<valid_size and rand<0.1:

            for _ in range(4):
                valid_x.append(data_x_v2[0][-1])
                data_x_v2[0].pop()
                valid_y.append(0)
                my_valid_ids.append(my_train_ids_v2[0][-1])
                my_train_ids_v2[0].pop()

            for _ in range(2):
                valid_x.append(data_x_v2[1][-1])
                data_x_v2[1].pop()
                valid_y.append(1)
                my_valid_ids.append(my_train_ids_v2[1][-1])
                my_train_ids_v2[1].pop()

            valid_x.append(data_x_v2[2][-1])
            data_x_v2[2].pop()
            valid_y.append(2)
            my_valid_ids.append(my_train_ids_v2[2][-1])
            my_train_ids_v2[2].pop()

            full_rounds += 1

    for j in range(len(data_x_v2[0])):
        data_x.append(data_x_v2[0][j])
        data_y.append(0)
        my_train_ids.append(my_train_ids_v2[0][j])

    for j in range(len(data_x_v2[1])):
        data_x.append(data_x_v2[1][j])
        data_y.append(1)
        my_train_ids.append(my_train_ids_v2[1][j])

    for j in range(len(data_x_v2[2])):
        data_x.append(data_x_v2[2][j])
        data_y.append(2)
        my_train_ids.append(my_train_ids_v2[2][j])

    train_set = (my_train_ids,data_x,data_y)
    valid_set = (my_valid_ids,valid_x,valid_y)

    print('Train: ',len(train_set[0]),' x ',len(train_set[1][0]))
    print('Valid: ',len(valid_set[0]),' x ',len(valid_set[1][0]))

    return get_shared_data(train_set),get_shared_data(valid_set)

import pandas as pd
def load_tensor_teslstra_data_v3(train_file,test_file,drop_col=None):

    tr_data = pd.read_csv(train_file)
    test_data = pd.read_csv(test_file)

    tr_ids = tr_data[['id']]
    tr_x = tr_data.ix[:,1:-1]
    tr_y = tr_data.ix[:,-1]

    test_ids = test_data[['id']]
    test_x = test_data.ix[:,1:]
    correct_order_test_ids = []
    import csv
    with open('test.csv', 'r',newline='') as f:
        reader = csv.reader(f)
        for i,row in enumerate(reader):
            if i==0:
                continue
            correct_order_test_ids.append(int(row[0]))

    if drop_col is not None:
        header = list(tr_x.columns.values)
        to_drop = [i for i,v in enumerate(header) if drop_col in v]

        tr_x.drop(tr_x.columns[to_drop],axis=1,inplace=True)
        test_x.drop(test_x.columns[to_drop],axis=1,inplace=True)

    return get_shared_data((tr_ids.as_matrix(),tr_x.as_matrix(),tr_y.as_matrix())), \
           get_shared_data((test_ids.as_matrix(),test_x.as_matrix())), correct_order_test_ids

class SDAEPretrainer(object):

    def __init__(self,param):

        self.batch_size = param['batch_size']
        self.iterations = param['iterations']
        self.in_size = param['in_size']
        self.out_size = param['out_size']
        self.hid_sizes = param['hid_sizes']
        self.learning_rate = param['learning_rate']
        self.pre_epochs = param['pre_epochs']
        self.finetune_epochs = param['fine_epochs']
        self.lam = param['lam']
        self.act = param['act']
        self.sdae = SDAE(self.in_size,self.out_size,self.hid_sizes,self.batch_size,self.learning_rate,self.lam,self.act,self.iterations)
        self.sdae.process()

        self.theano_tr_ids, self.tr_pred, self.tr_act = [],[],[]

    def pretrain(self,tr_all,v_all,ts_all,weights):

        tr_ids,tr_x,tr_y = tr_all
        v_ids,v_x,v_y = v_all
        ts_ids,ts_x = ts_all

        all_x = []
        all_x.extend(tr_x.get_value())
        all_x.extend(v_x.get_value())
        all_x.extend(ts_x.get_value())
        all_theano_x = shared(value=np.asarray(all_x,dtype=config.floatX),borrow=True)

        n_pretrain_batches = ceil(all_theano_x.get_value(borrow=True).shape[0] / self.batch_size)

        pretrain_func = self.sdae.pre_train(all_theano_x)

        for epoch in range(self.pre_epochs):
            pre_train_cost = []
            for b in range(n_pretrain_batches):
                pre_train_cost.append(pretrain_func(b))
            print('Pretrain cost ','(epoch ', epoch,'): ',np.mean(pre_train_cost))


    def get_features(self,tr_all,v_all,ts_all,layer_idx):

        tr_ids,tr_x,tr_y = tr_all
        v_ids,v_x,v_y = v_all
        ts_ids,ts_x = ts_all

        tr_feature_func = self.sdae.get_features(tr_x,tr_y,tr_ids,layer_idx,False)
        v_features_func = self.sdae.get_features(v_x,v_y,v_ids,layer_idx,False)
        ts_features_func = self.sdae.get_features(ts_x,None,ts_ids,layer_idx,True)

        n_train_batches = ceil(tr_x.get_value(borrow=True).shape[0] / self.batch_size)
        n_valid_batches = ceil(v_x.get_value(borrow=True).shape[0] / self.batch_size)
        n_test_batches = ceil(ts_x.get_value(borrow=True).shape[0] / self.batch_size)

        all_tr_features = []
        all_tr_outputs = []
        all_ts_features = []
        for b in range(n_train_batches):
            features, y, ids = tr_feature_func(b)

            temp = np.concatenate((np.reshape(ids,(ids.shape[0],1)),features),axis=1)

            all_tr_features.extend(temp.tolist())
            all_tr_outputs.extend(y)

        print('Size train features: ',len(all_tr_features),' x ',len(all_tr_features[0]))

        for b in range(n_valid_batches):
            features, y, ids = v_features_func(b)
            temp = [ids]
            temp = np.concatenate((np.reshape(ids,(ids.shape[0],1)),features),axis=1)

            all_tr_features.extend(temp.tolist())
            all_tr_outputs.extend(y)

        print('Size train+valid features: ',len(all_tr_features),' x ',len(all_tr_features[0]))

        for b in range(n_test_batches):
            features, y, ids = ts_features_func(b)
            
            temp = np.concatenate((np.reshape(ids,(ids.shape[0],1)),features),axis=1)
            all_ts_features.extend(temp.tolist())

        print('Size test features: ',len(all_ts_features),' x ',len(all_ts_features[0]))

        return (all_tr_features,all_tr_outputs),all_ts_features

    def get_labels(self):
        return self.theano_tr_ids,self.tr_pred,self.tr_act

    def get_test_results(self,ts_data):
        ts_ids, test_x = ts_data

        test_x = shared(value=np.asarray(test_x,dtype=config.floatX),borrow=True)

        test_func = self.sdae.test(test_x)
        n_test_batches = (test_x.get_value(borrow=True).shape[0])

        test_out_probs = []
        for b in range(n_test_batches):
            cls,probs = test_func(b)
            test_out_probs.append(probs[0])

        return ts_ids,test_out_probs

from OnlyXGBoost import MyXGBClassifier
if __name__ == '__main__':

    th_train,th_test,correct_ids = load_tensor_teslstra_data_v3('features_train.csv', 'features_test.csv',None)
    th_tr_slice,th_v_slice = tensor_divide_test_valid((th_train[0],th_train[1],th_train[2]))

    import collections
    dl_params_1 = collections.defaultdict()
    dl_params_1['batch_size'] = 100
    dl_params_1['iterations'] = 1
    dl_params_1['in_size'] = 398
    dl_params_1['out_size'] = 3
    dl_params_1['hid_sizes'] = [500,500,500]
    dl_params_1['learning_rate'] = 0.75
    dl_params_1['pre_epochs'] = 100
    dl_params_1['fine_epochs'] = 1
    dl_params_1['lam'] = 1e-5
    dl_params_1['act'] = 'relu'

    sdae_pre = SDAEPretrainer(dl_params_1)

    sdae_pre.pretrain(th_tr_slice,th_v_slice,th_test,None)

    xgbClassifiers = []
    xgbClassifiers.append(MyXGBClassifier(n_rounds=50,eta=0.2,max_depth=10,subsample=0.9,colsample_bytree=0.9))
    xgbClassifiers[-1].fit(th_tr_slice[1].get_value(borrow=True),th_tr_slice[2].eval())

    xgbLogLosses = []
    xgbLogLosses.append(xgbClassifiers[-1].logloss(th_v_slice[1].get_value(borrow=True),th_v_slice[2].eval()))

    xgbProbas = []
    for h_i in range(len(dl_params_1['hid_sizes'])):
        print('Getting features for ',h_i,' layer')
        (tr_feat,tr_out),ts_feat = sdae_pre.get_features(th_tr_slice,th_v_slice,th_test,h_i)
        tr_feat = np.asarray(tr_feat, dtype=config.floatX)
        tr_out = np.asarray(tr_out, dtype=config.floatX)

        th_tr_feat = get_shared_data((tr_feat[:,0],tr_feat[:,1:],tr_out))
        tr_slice_tmp,v_slice_tmp = tensor_divide_test_valid(th_tr_feat)
        th_tr_slice_ids_tmp,th_tr_slice_x_tmp,th_tr_slice_y_tmp = tr_slice_tmp
        th_v_slice_ids_tmp,th_v_slice_x_tmp,th_v_slice_y_tmp = v_slice_tmp

        tr_slice_ids_tmp = th_tr_slice_ids_tmp.eval()
        tr_slice_x_tmp = th_tr_slice_x_tmp.get_value(borrow=True)
        tr_slice_y_tmp = th_tr_slice_y_tmp.eval()

        v_slice_ids_tmp = th_v_slice_ids_tmp.eval()
        v_slice_x_tmp = th_v_slice_x_tmp.get_value(borrow=True)
        v_slice_y_tmp = th_v_slice_y_tmp.eval()

        print('XGBClassifier for ', h_i, ' layer ...')
        xgbClassifiers.append(MyXGBClassifier(n_rounds=50,eta=0.2,max_depth=10,subsample=0.9,colsample_bytree=0.9))
        xgbClassifiers[-1].fit(tr_slice_x_tmp,tr_slice_y_tmp)
        xgbProbas.append(xgbClassifiers[-1].predict_proba(tr_slice_x_tmp))
        xgbLogLosses.append(xgbClassifiers[-1].logloss(v_slice_x_tmp,v_slice_y_tmp))


    def logloss(probs, Y):
        assert len(probs)==len(Y)
        logloss = 0.0
        for i in range(len(Y)):
            tmp_y = [0.,0.,0.]
            tmp_y[Y[i]]=1.
            v_prob = probs[i]
            if any(v_prob)==1.:
                v_prob = np.asarray([np.max([np.min([p,1-1e-15]),1e-15]) for p in v_prob])

            logloss += np.sum(np.asarray(tmp_y)*np.log(np.asarray(v_prob)))
        logloss = -logloss/len(Y)

        return logloss

    print(xgbLogLosses)

    avg_result = np.zeros((th_tr_slice[0].eval().shape[0],dl_params_1['out_size']),dtype=config.floatX)
    weigh_avg_result = np.zeros((th_tr_slice[0].eval().shape[0],dl_params_1['out_size']),dtype=config.floatX)
    for r_i,result in enumerate(xgbProbas):
        if r_i == np.argmin(xgbLogLosses):
            weigh_avg_result = np.add(weigh_avg_result,0.3*np.asarray(result))
        else:
            weigh_avg_result = np.add(weigh_avg_result,((1-0.3)/len(xgbProbas))*np.asarray(result))

        avg_result = np.add(avg_result,np.asarray(result))

    print("Avg logloss: ",logloss(avg_result,th_v_slice[2].eval()))
    print("Weighted avg logloss: ",logloss(avg_result,th_v_slice[2].eval()))