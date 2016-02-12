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
    my_train_weights = []

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

    valid_size = 1000
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
                my_train_weights.append(0.152)

            for _ in range(2):
                data_x.append(data_x_v2[1][-1])
                data_x_v2[1].pop()
                data_y.append(1)
                my_train_ids.append(my_train_ids_v2[1][-1])
                my_train_ids_v2[1].pop()
                my_train_weights.append(0.388)

            data_x.append(data_x_v2[2][-1])
            data_x_v2[2].pop()
            data_y.append(2)
            my_train_ids.append(my_train_ids_v2[2][-1])
            my_train_ids_v2[2].pop()
            my_train_weights.append(1.0)

            full_rounds += 1

        elif len(valid_x)<valid_size and rand<0.5:

            for _ in range(1):
                valid_x.append(data_x_v2[0][-1])
                data_x_v2[0].pop()
                valid_y.append(0)
                my_valid_ids.append(my_train_ids_v2[0][-1])
                my_train_ids_v2[0].pop()

            for _ in range(1):
                valid_x.append(data_x_v2[1][-1])
                data_x_v2[1].pop()
                valid_y.append(1)
                my_valid_ids.append(my_train_ids_v2[1][-1])
                my_train_ids_v2[1].pop()

            for _ in range(1):
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
        my_train_weights.append(0.152)

    for j in range(len(data_x_v2[1])):
        data_x.append(data_x_v2[1][j])
        data_y.append(1)
        my_train_ids.append(my_train_ids_v2[1][j])
        my_train_weights.append(0.388)

    for j in range(len(data_x_v2[2])):
        data_x.append(data_x_v2[2][j])
        data_y.append(2)
        my_train_ids.append(my_train_ids_v2[2][j])
        my_train_weights.append(1.0)

    train_set = (my_train_ids,data_x,data_y)
    valid_set = (my_valid_ids,valid_x,valid_y)

    print('Train: ',len(train_set[0]),' x ',len(train_set[1][0]))
    print('Valid: ',len(valid_set[0]),' x ',len(valid_set[1][0]))

    return get_shared_data(train_set),get_shared_data(valid_set),np.asarray(my_train_weights).reshape(-1,1)

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

import xgboost as xgb
class MyXGBClassifier(object):

    def __init__(self, n_rounds=100, **params):
        self.params = params
        self.params.update({'booster':'gbtree'})
        self.params.update({'silent':1})
        self.params.update({'objective': 'multi:softprob'})
        self.params.update({'num_class': 3})
        self.params.update({'eval_metric':'mlogloss'})
        self.clf = None
        self.n_rounds = n_rounds
        self.dtrain = None

    def fit_with_early_stop(self, X, Y, V_X, V_Y, weights=None):
        num_boost_round = self.n_rounds
        self.dtrain = xgb.DMatrix(X, label=Y, weight=weights)
        dvalid = xgb.DMatrix(V_X, label=V_Y)
        evallist = [(self.dtrain,'train'), (dvalid, 'eval')]

        # don't use iterative train if using early_stop
        self.clf = xgb.train(self.params, self.dtrain, num_boost_round, evallist, early_stopping_rounds=10)

    def fit(self, X, Y, weights=None):
        num_boost_round = self.n_rounds
        self.dtrain = xgb.DMatrix(X, label=Y, weight=weights)

        # don't use iterative train if using early_stop
        self.clf = xgb.train(self.params, self.dtrain, num_boost_round)

    def fit_with_valid(self,V_X,V_Y,rounds):
        dvalid = xgb.DMatrix(V_X, label=V_Y)
        for idx in range(rounds):
            self.clf.update(dvalid,idx)

    def predict(self, X):
        Y = self.clf.predict_proba(X)
        y = np.argmax(Y, axis=1)
        return np.array(y)

    def predict_proba(self, X):
        dtest = xgb.DMatrix(X)
        return self.clf.predict(dtest)

    def get_params(self, deep=True):
        return self.params

    def set_params(self, **params):
        self.params.update(params)
        return self

    def logloss(self, X, Y):
        return logloss_xgb(self,X,Y)

    def score(self, X, Y):
        return 1 / logloss_xgb(self,X, Y)

def logloss_xgb(est, X, Y):

    probs = est.predict_proba(X)
    logloss = 0.0
    for i in range(len(Y)):
        tmp_y = [0.,0.,0.]
        tmp_y[Y[i]]=1.
        v_probs = probs[i]
        if any(v_probs)==1.:
            v_probs = np.asarray([np.max([np.min([p,1-1e-15]),1e-15]) for p in v_probs])

        logloss += np.sum(np.asarray(tmp_y)*np.log(np.asarray(v_probs)))
    logloss = -logloss/len(Y)

    return logloss

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
        self.denoise = param['denoise']
        self.corr_level = param['corr_level']
        self.sdae = SDAE(self.in_size,self.out_size,self.hid_sizes,self.batch_size,self.learning_rate,self.lam,self.act,self.iterations,self.denoise,self.corr_level)
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

        if weights is not None:
            all_weights = np.asarray(tr_weights)
            all_weights = np.append(all_weights, np.asarray([1 for _ in range(v_x.get_value().shape[0])]).reshape(-1,1),axis=0)
            all_weights = np.append(all_weights, np.asarray([1 for _ in range(ts_x.get_value().shape[0])]).reshape(-1,1),axis=0)
            print(all_weights.shape)
            print(all_theano_x.get_value().shape)
        else:
            all_weights = None
        n_pretrain_batches = ceil(all_theano_x.get_value(borrow=True).shape[0] / self.batch_size)

        pretrain_func = self.sdae.pre_train(all_theano_x,shared(all_weights))

        for epoch in range(self.pre_epochs):
            pre_train_cost = []
            b_indices = [i for i in range(n_pretrain_batches)]
            np.random.shuffle(b_indices)
            for b in b_indices:
                pre_train_cost.append(pretrain_func(b))
            print('Pretrain cost (Layer-wise) ','(epoch ', epoch,'): ',np.mean(pre_train_cost))

    def full_pretrain(self,tr_all,v_all,ts_all,weights):

        tr_ids,tr_x,tr_y = tr_all
        v_ids,v_x,v_y = v_all
        ts_ids,ts_x = ts_all

        all_x = []
        all_x.extend(tr_x.get_value())
        all_x.extend(v_x.get_value())
        all_x.extend(ts_x.get_value())
        all_theano_x = shared(value=np.asarray(all_x,dtype=config.floatX),borrow=True)

        if weights is not None:
            all_weights = np.asarray(tr_weights)
            all_weights = np.append(all_weights,np.asarray([1 for _ in range(v_x.get_value().shape[0])]).reshape(-1,1),axis=0)
            all_weights = np.append(all_weights,np.asarray([1 for _ in range(ts_x.get_value().shape[0])]).reshape(-1,1),axis=0)
        else:
            all_weights = None
        n_pretrain_batches = ceil(all_theano_x.get_value(borrow=True).shape[0] / self.batch_size)

        full_pretrain_func = self.sdae.full_pretrain(all_theano_x,shared(all_weights))

        for epoch in range(self.pre_epochs//2):
            full_pre_cost = []
            b_indices = [i for i in range(n_pretrain_batches)]
            np.random.shuffle(b_indices)
            for b in b_indices:
                full_pre_cost.append(full_pretrain_func(b))
            print('Pretrain cost (Full) ','(epoch ', epoch,'): ',np.mean(full_pre_cost))


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

def save_features(file_name, X, Y):
    header = 'id'
    for i in range(X.shape[1]-1):
        header+=',feat_'+str(i)
    if Y is not None:
        header+=',out'
        res = np.append(X,Y,axis=1)
    else:
        res = X

    np.savetxt(file_name, res, delimiter=",", header=header, comments='')

def logloss(probs, Y, n_classes = 3):
    #assert len(probs)==len(Y)
    logloss = 0.0
    for i in range(len(Y)):
        tmp_y = [0. for _ in range(n_classes)]
        tmp_y[Y[i]]=1.
        v_prob = probs[i]/np.sum(probs[i])
        assert np.sum(v_prob)>0.999 and np.sum(v_prob)<1.00001
        v_prob = np.asarray([np.max([np.min([p,1-1e-15]),1e-15]) for p in v_prob])
        logloss += np.sum(np.asarray(tmp_y)*np.log(np.asarray(v_prob)))
        assert np.sum(np.asarray(tmp_y)*np.log(np.asarray(v_prob)))<0
    logloss = -logloss/len(Y)

    return logloss

from sklearn.ensemble import ExtraTreesClassifier
if __name__ == '__main__':


    test_function = False

    if test_function:
        print(' ############################################',
              '\nWARNING: TESTING MODE!!!! . Will run quickly...',
              '\n############################################\n')

    early_stopping = True
    use_weights = True
    select_features = False # select features using extratrees (based on importance)
    train_with_valid = False # this specify if we want to finetune with the validation data
    persist_features = False
    tr_v_rounds = 10
    print('Select features with Extratrees: ',select_features)
    print('Train with validation set: ',train_with_valid)

    th_train,th_test,correct_ids = load_tensor_teslstra_data_v3('features_train.csv', 'features_test.csv','loc')
    th_tr_slice,th_v_slice,tr_weights = tensor_divide_test_valid((th_train[0],th_train[1],th_train[2]))

    if not use_weights:
        tr_weights = None

    ts_ids,ts_x = th_test[0].eval(),th_test[1].get_value(borrow=True)

    import collections
    dl_params_1 = collections.defaultdict()
    dl_params_1['batch_size'] = 10
    dl_params_1['iterations'] = 1
    dl_params_1['in_size'] = th_train[1].get_value(borrow=True).shape[1]
    dl_params_1['out_size'] = 3
    dl_params_1['hid_sizes'] = [100,100,100]
    dl_params_1['learning_rate'] = 0.1
    dl_params_1['pre_epochs'] = 15
    dl_params_1['fine_epochs'] = 1
    dl_params_1['lam'] = 1e-8
    dl_params_1['act'] = 'relu'
    dl_params_1['denoise'] = True
    dl_params_1['corr_level'] = 0.2
    if test_function:
        dl_params_1['pre_epochs'] = 2
        num_rounds = 10
    else:
        num_rounds = 100
    sdae_pre = SDAEPretrainer(dl_params_1)

    fimp_cutoff_thresh = [0.2,0.1,0.0,0.0]
    second_layer_classifiers = ['xgb','gbm']

    sdae_pre.pretrain(th_tr_slice,th_v_slice,th_test,tr_weights)
    sdae_pre.full_pretrain(th_tr_slice,th_v_slice,th_test,tr_weights)

    xgbProbas = []
    xgbTestProbas = []
    xgbClassifiers = []
    xgbLogLosses = []

    if not test_function:
        xgbClassifiers.append(MyXGBClassifier(n_rounds=num_rounds,eta=0.2,max_depth=8,subsample=0.9,colsample_bytree=0.9))
        if early_stopping:
            xgbClassifiers[-1].fit_with_early_stop(th_tr_slice[1].get_value(borrow=True), th_tr_slice[2].eval(),
                                                   th_v_slice[1].get_value(borrow=True), th_v_slice[2].eval(), tr_weights)
        else:
            xgbClassifiers[-1].fit(th_tr_slice[1].get_value(borrow=True), th_tr_slice[2].eval(),tr_weights)

        if train_with_valid:
            xgbClassifiers[-1].fit_with_valid(th_v_slice[1].get_value(borrow=True),th_v_slice[2].eval(),tr_v_rounds)


        xgbLogLosses.append(xgbClassifiers[-1].logloss(th_v_slice[1].get_value(borrow=True),th_v_slice[2].eval()))
        xgbProbas.append(xgbClassifiers[-1].predict_proba(th_v_slice[1].get_value(borrow=True)))
        xgbTestProbas.append(xgbClassifiers[-1].predict_proba(ts_x))

    for h_i in range(len(dl_params_1['hid_sizes'])):
        print('Getting features for ',h_i,' layer')
        (tr_feat,tr_out),ts_feat = sdae_pre.get_features(th_tr_slice,th_v_slice,th_test,h_i)
        tr_feat = np.asarray(tr_feat, dtype=config.floatX)
        tr_out = np.asarray(tr_out, dtype=config.floatX)
        ts_feat = np.asarray(ts_feat, dtype=config.floatX)

        th_ts_feat = get_shared_data((ts_feat[:,0],ts_feat[:,1:],None))
        ts_ids_tmp,ts_x_tmp = th_ts_feat[0].eval(),th_ts_feat[1].get_value(borrow=True)

        th_tr_feat = get_shared_data((tr_feat[:,0],tr_feat[:,1:],tr_out))
        tr_slice_tmp,v_slice_tmp,tr_weights_tmp = tensor_divide_test_valid(th_tr_feat)
        th_tr_slice_ids_tmp,th_tr_slice_x_tmp,th_tr_slice_y_tmp = tr_slice_tmp
        th_v_slice_ids_tmp,th_v_slice_x_tmp,th_v_slice_y_tmp = v_slice_tmp

        tr_slice_ids_tmp = th_tr_slice_ids_tmp.eval()
        tr_slice_x_tmp = th_tr_slice_x_tmp.get_value(borrow=True)
        tr_slice_y_tmp = th_tr_slice_y_tmp.eval()

        v_slice_ids_tmp = th_v_slice_ids_tmp.eval()
        v_slice_x_tmp = th_v_slice_x_tmp.get_value(borrow=True)
        v_slice_y_tmp = th_v_slice_y_tmp.eval()

        if 'xgb' in second_layer_classifiers:
            print('XGBClassifier for ', h_i, ' layer ...')
            xgbClassifiers.append(MyXGBClassifier(n_rounds=num_rounds,eta=0.2,max_depth=8,subsample=0.9,colsample_bytree=0.9))
        if select_features:
            forest = ExtraTreesClassifier(n_estimators=1000, max_features="auto", n_jobs=5, random_state=0)
            forest.fit(tr_slice_x_tmp, tr_slice_y_tmp)
            tr_feature_imp = forest.feature_importances_

            tr_new_features_tmp,v_new_features_tmp,ts_new_features_tmp = None,None,None
            for tmp_idx in np.argsort(tr_feature_imp).tolist()[int(tr_slice_x_tmp.shape[1]*fimp_cutoff_thresh[h_i]):]:
                tr_tmp = np.asarray(tr_slice_x_tmp[:,tmp_idx]).reshape(-1,1)
                v_tmp = np.asarray(v_slice_x_tmp[:,tmp_idx]).reshape(-1,1)
                ts_tmp = np.asarray(ts_x_tmp[:,tmp_idx]).reshape(-1,1)

                if tr_new_features_tmp is not None:
                    tr_new_features_tmp = np.append(tr_new_features_tmp,tr_tmp, axis=1)
                    v_new_features_tmp = np.append(v_new_features_tmp,v_tmp, axis=1)
                    ts_new_feature_tmp = np.append(ts_new_feature_tmp,v_tmp, axis=1)
                else:
                    tr_new_features_tmp = tr_tmp
                    v_new_features_tmp = v_tmp
                    ts_new_feature_tmp = ts_tmp

            print('Selected features size: ',tr_new_features_tmp.shape)
            if 'xgb' in second_layer_classifiers:
                if early_stopping:
                    xgbClassifiers[-1].fit_with_early_stop(tr_new_features_tmp, tr_slice_y_tmp,
                                                           v_new_features_tmp, v_slice_y_tmp, tr_weights_tmp)
                else:
                    xgbClassifiers[-1].fit(tr_new_features_tmp, tr_slice_y_tmp,tr_weights_tmp)

                if train_with_valid:
                    xgbClassifiers[-1].fit_with_valid(v_new_features_tmp,v_slice_y_tmp,tr_v_rounds)

                xgbProbas.append(xgbClassifiers[-1].predict_proba(v_new_features_tmp))
                xgbTestProbas.append(xgbClassifiers[-1].predict_proba(ts_new_feature_tmp))
                xgbLogLosses.append(xgbClassifiers[-1].logloss(v_new_features_tmp,v_slice_y_tmp))

        else: # if we're using all features
            if 'xgb' in second_layer_classifiers:
                if early_stopping:
                    xgbClassifiers[-1].fit_with_early_stop(tr_slice_x_tmp, tr_slice_y_tmp,
                                                           v_slice_x_tmp, v_slice_y_tmp, tr_weights_tmp)
                else:
                    xgbClassifiers[-1].fit(tr_slice_x_tmp, tr_slice_y_tmp,tr_weights_tmp)
                if train_with_valid:
                    xgbClassifiers[-1].fit_with_valid(th_v_slice[1].get_value(borrow=True),
                                                      th_v_slice[2].eval(),tr_v_rounds)

                xgbProbas.append(xgbClassifiers[-1].predict_proba(v_slice_x_tmp)) #append validation probabilities
                xgbTestProbas.append(xgbClassifiers[-1].predict_proba(ts_x_tmp)) #append test probabilities
                xgbLogLosses.append(xgbClassifiers[-1].logloss(v_slice_x_tmp,v_slice_y_tmp))

            if persist_features:
                print('Persisting features for ',h_i,' layer')
                # saving train+valid features

                tr_x_save_features = np.append(
                    np.append(tr_slice_ids_tmp.reshape(-1,1),tr_slice_x_tmp,axis=1),
                    np.append(v_slice_ids_tmp.reshape(-1,1),v_slice_x_tmp,axis=1),
                    axis=0)

                tr_y_save_features = np.append(tr_slice_y_tmp,v_slice_y_tmp,axis=0).reshape(-1,1)

                save_features('features_dl_'+str(h_i)+'_train.csv',tr_x_save_features,tr_y_save_features)

                # saving test features
                ts_x_save_features = np.append(ts_ids_tmp.reshape(-1,1),ts_x_tmp,axis=1)
                save_features('features_dl_'+str(h_i)+'_test.csv',ts_x_save_features,None)

    print(xgbLogLosses)

    avg_result = np.zeros((th_v_slice[0].eval().shape[0],dl_params_1['out_size']),dtype=config.floatX)
    weigh_avg_result = np.zeros((th_v_slice[0].eval().shape[0],dl_params_1['out_size']),dtype=config.floatX)
    alpha = 0.5
    exp_test_result = np.zeros((th_v_slice[0].eval().shape[0],dl_params_1['out_size']),dtype=config.floatX)

    avg_test_result = np.zeros((th_test[0].eval().shape[0],dl_params_1['out_size']),dtype=config.floatX)
    weigh_avg_test_result = np.zeros((th_test[0].eval().shape[0],dl_params_1['out_size']),dtype=config.floatX)
    best_test_result = np.zeros((th_test[0].eval().shape[0],dl_params_1['out_size']),dtype=config.floatX)


    for r_i,result in enumerate(xgbProbas):
        test_result = xgbTestProbas[r_i]
        if r_i == np.argmin(xgbLogLosses):
            weigh_avg_result = np.add(weigh_avg_result,0.3*np.asarray(result))
            weigh_avg_test_result = np.add(weigh_avg_test_result, 0.3*np.asarray(test_result))
        else:
            weigh_avg_result = np.add(weigh_avg_result,((1-0.3)/(len(xgbProbas)-1))*np.asarray(result))
            weigh_avg_test_result = np.add(weigh_avg_test_result,((1-0.3)/(len(xgbTestProbas)-1))*np.asarray(test_result))

        avg_result = np.add(avg_result,np.asarray(result))/len(xgbProbas)
        avg_test_result = np.add(avg_test_result,np.asarray(test_result))

    for r_i in np.argsort(xgbLogLosses):
        alpha *= 0.5
        exp_test_result = np.add(alpha*np.asarray(xgbProbas),(1-alpha)*np.asarray(exp_test_result))

    print("Avg logloss: ",logloss(avg_result,th_v_slice[2].eval()))
    print("Weighted avg logloss: ",logloss(weigh_avg_result,th_v_slice[2].eval()))
    print("Exp decay logloss: ", logloss(exp_test_result,th_v_slice[2].eval()))
    print("Best avg logloss: ",logloss(xgbProbas[np.argmin(xgbLogLosses)],th_v_slice[2].eval()))

    print('\n Saving out probabilities (test)')
    import csv
    with open('W_sdae_xgboost_output.csv', 'w',newline='') as f:

        writer = csv.writer(f)
        writer.writerow('id,predict_0,predict_1,predict_2')
        for id in correct_ids:
            c_id = ts_ids.flatten().tolist().index(id)
            probs = weigh_avg_test_result[int(c_id),:]
            row = [id,probs[0], probs[1], probs[2]]
            writer.writerow(row)

    import csv
    with open('best_sdae_xgboost_output.csv', 'w',newline='') as f:

        writer = csv.writer(f)
        writer.writerow(['id','predict_0','predict_1','predict_2'])
        for id in correct_ids:
            c_id = ts_ids.flatten().tolist().index(id)
            probs = xgbTestProbas[np.argmin(xgbLogLosses)][int(c_id),:]
            row = [id,probs[0], probs[1], probs[2]]
            writer.writerow(row)

