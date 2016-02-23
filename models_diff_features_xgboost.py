import xgboost as xgb
import numpy as np

import pandas as pd
def load_teslstra_data_v3(train_file,test_file,drop_cols=None):

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

    if drop_cols is not None:
        for drop_col in drop_cols:
            header = list(tr_x.columns.values)
            to_drop = [i for i,v in enumerate(header) if drop_col in v]

            tr_x.drop(tr_x.columns[to_drop],axis=1,inplace=True)
            test_x.drop(test_x.columns[to_drop],axis=1,inplace=True)

    return (tr_ids.as_matrix(),tr_x.as_matrix(),tr_y.as_matrix()), \
           (test_ids.as_matrix(),test_x.as_matrix()), correct_order_test_ids

def divide_test_valid(train_data,weights=None):

    '''
            0th class 6.58, 1st class 2.58, 2nd class 1 (ratios)
    '''
    import csv

    tmp_weights = [0.2,1.0,1.0]
    v_select_prob = 0.5
    my_train_ids = []
    my_valid_ids = []
    my_train_ids_v2 = [[],[],[]]
    my_train_weights = []

    data_x = []
    data_y = []
    valid_x,valid_y = [],[]
    data_x_v2 = [[],[],[]]

    tr_ids,tr_x, tr_y = train_data

    for i in range(len(tr_x)):
        # first 2 columns are ID and location
        output = int(tr_y[i])
        data_x_v2[output].append(tr_x[i])
        my_train_ids_v2[output].append(tr_ids[i])

    valid_size =500
    full_rounds = 1
    orig_class_2_length = len(data_x_v2[2])
    for _ in range(orig_class_2_length):
        rand = np.random.random()
        if rand>=v_select_prob or len(valid_x)>valid_size:
            for _ in range(4) :
                data_x.append(data_x_v2[0][-1])
                data_x_v2[0].pop()
                data_y.append(0)
                my_train_ids.append(my_train_ids_v2[0][-1])
                my_train_ids_v2[0].pop()
                if weights is None:
                    my_train_weights.append(tmp_weights[0])
                else:
                    my_train_weights.append(weights[0])

            for _ in range(2):
                data_x.append(data_x_v2[1][-1])
                data_x_v2[1].pop()
                data_y.append(1)
                my_train_ids.append(my_train_ids_v2[1][-1])
                my_train_ids_v2[1].pop()
                if weights is None:
                    my_train_weights.append(tmp_weights[1])
                else:
                    my_train_weights.append(weights[1])

            data_x.append(data_x_v2[2][-1])
            data_x_v2[2].pop()
            data_y.append(2)
            my_train_ids.append(my_train_ids_v2[2][-1])
            my_train_ids_v2[2].pop()
            if weights is None:
                my_train_weights.append(tmp_weights[2])
            else:
                my_train_weights.append(weights[2])

            full_rounds += 1

        elif len(valid_x)<valid_size and rand<v_select_prob:

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
        if weights is None:
            my_train_weights.append(tmp_weights[0])
        else:
            my_train_weights.append(weights[0])

    for j in range(len(data_x_v2[1])):
        data_x.append(data_x_v2[1][j])
        data_y.append(1)
        my_train_ids.append(my_train_ids_v2[1][j])
        if weights is None:
            my_train_weights.append(tmp_weights[1])
        else:
            my_train_weights.append(weights[1])

    for j in range(len(data_x_v2[2])):
        data_x.append(data_x_v2[2][j])
        data_y.append(2)
        my_train_ids.append(my_train_ids_v2[2][j])
        if weights is None:
            my_train_weights.append(tmp_weights[2])
        else:
            my_train_weights.append(weights[2])

    train_set = (my_train_ids,data_x,data_y)
    valid_set = (my_valid_ids,valid_x,valid_y)

    print('Train: ',len(train_set[0]),' x ',len(train_set[1][0]))
    print('Valid: ',len(valid_set[0]),' x ',len(valid_set[1][0]))

    return train_set,valid_set,np.asarray(my_train_weights).reshape(-1,1)


def normalize_data(tr_x,ts_x,normz=None,axis=0):
    if normz is 'scale':
        tr_x = scale(tr_x,axis=axis)
        ts_x = scale(ts_x,axis=axis)
    elif normz is 'minmax':
        minmax_scaler = MinMaxScaler()
        if axis==0:
            for c_i in range(tr_x.shape[1]):
                tr_x[:,c_i] = minmax_scaler.fit_transform(tr_x[:,c_i])
                ts_x[:,c_i] = minmax_scaler.fit_transform(ts_x[:,c_i])
        elif axis==1:
            for r_i in range(tr_x.shape[0]):
                tr_x[r_i,:] = minmax_scaler.fit_transform(tr_x[r_i,:])
                ts_x[r_i,:] = minmax_scaler.fit_transform(ts_x[r_i,:])
    elif normz is 'sigmoid':
        if axis==0:
            col_max = np.max(tr_x,axis=0)
            cols_non_norm = np.argwhere(col_max>1).tolist()
            tr_x[:,cols_non_norm] = -0.5 + (1 / (1 + np.exp(-tr_x[:,cols_non_norm])))
            # TODO: implement col_max col_non_norm for test set
            ts_x[:,cols_non_norm] = -0.5 + (1/(1+np.exp(-ts_x[:,cols_non_norm])))
        elif axis==1:
            row_max = np.max(tr_x,axis=1)
            rows_non_norm = np.argwhere(row_max>1).tolist()
            tr_x[rows_non_norm,:] = -0.5 + (1 / (1 + np.exp(-tr_x[rows_non_norm,:])))
            # TODO: implement row_max row_non_norm for test set
            ts_x[rows_non_norm,:] = -0.5 + (1/(1+np.exp(-ts_x[rows_non_norm,:])))

    return tr_x,ts_x

from numpy import linalg as LA
from sklearn.preprocessing import normalize
from sklearn.preprocessing import scale,MinMaxScaler
def get_scale_log_x_plus_1(tr_all,ts_all,normz=None,axis=0):
    tr_ids,tr_x_orig,tr_y = tr_all
    ts_ids,ts_x_orig = ts_all

    tr_x = np.copy(tr_x_orig)
    ts_x = np.copy(ts_x_orig)

    if np.min(tr_x,axis=0).all() and np.min(tr_x,axis=0).all()>=0:
        print('no minus')
        tr_x = np.log(tr_x+1)
        ts_x = np.log(ts_x+1)
    else:
        tr_x = np.log(tr_x+1+np.abs(np.min(tr_x,axis=0)))
        ts_x = np.log(ts_x+1+np.abs(np.min(ts_x,axis=0)))
    assert not np.isnan(tr_x).any()
    assert not np.isnan(ts_x).any()

    tr_x,ts_x = normalize_data(tr_x,ts_x,normz,axis)

    assert not np.isnan(tr_x).any()
    assert not np.isnan(ts_x).any()

    return (tr_ids,tr_x,tr_y),(ts_ids,ts_x)

from sklearn.ensemble import GradientBoostingClassifier
def get_exp_decay_fimp(tr_all,ts_all,decay=0.9,normz=None,axis=0):
    tr_ids,tr_x_orig,tr_y = tr_all
    ts_ids,ts_x_orig = ts_all

    tr_x = np.copy(tr_x_orig)
    ts_x = np.copy(ts_x_orig)

    clf = GradientBoostingClassifier(n_estimators=100,max_depth=5,learning_rate=0.1)
    clf = clf.fit(tr_x, tr_y)
    fimp = np.asarray(clf.feature_importances_)
    ord_feature_idx = list(reversed(np.argsort(fimp)))
    # need to sort, else hard to perform delete
    train_feature_idx = ord_feature_idx

    curr_decay = 1
    for fidx in train_feature_idx:
        curr_decay *= decay
        tr_x[:,fidx] *= curr_decay
        ts_x[:,fidx] *= curr_decay

    tr_x,ts_x = normalize_data(tr_x,ts_x,normz,axis)

    return (tr_ids,tr_x,tr_y),(ts_ids,ts_x)

def get_pow(tr_all,ts_all,n=2,normz=None,axis=0):
    tr_ids,tr_x_orig,tr_y = tr_all
    ts_ids,ts_x_orig = ts_all

    tr_x = np.copy(tr_x_orig)
    ts_x = np.copy(ts_x_orig)

    tr_x = tr_x**n
    ts_x = ts_x**n

    tr_x,ts_x = normalize_data(tr_x,ts_x,normz,axis)

    return (tr_ids,tr_x,tr_y),(ts_ids,ts_x)

from scipy.stats.stats import pearsonr
def get_correlated_removed(tr_all,ts_all,thresh=0.95,normz=None,axis=0):
    tr_ids,tr_x_orig,tr_y = tr_all
    ts_ids,ts_x_orig = ts_all

    tr_x = np.copy(tr_x_orig)
    ts_x = np.copy(ts_x_orig)

    uncorr_idx = []
    for c_i in range(tr_x.shape[1]):
        for c_i_2 in range(c_i+1,tr_x.shape[1]):
            coeff = pearsonr(np.asarray(tr_x[:,c_i]).reshape(-1,1),np.asarray(tr_x[:,c_i_2]).reshape(-1,1))[0]
            if coeff > thresh and c_i not in uncorr_idx:
                uncorr_idx.append(c_i)

    print("Found ",len(uncorr_idx)," uncorrelated features")
    uncorr_idx.sort()
    for idx in reversed(uncorr_idx):
        tr_x = np.delete(tr_x,idx,axis=1)
        ts_x = np.delete(ts_x,idx,axis=1)

    tr_x, ts_x = normalize_data(tr_x,ts_x,normz,axis)

    return (tr_ids,tr_x,tr_y),(ts_ids,ts_x)

from sklearn.cluster import KMeans
def get_kmeans_features(tr_all,ts_all,n_clusters=3,normz=None,axis=0):
    tr_ids,tr_x_orig,tr_y = tr_all
    ts_ids,ts_x_orig = ts_all

    tr_x = np.copy(tr_x_orig)
    ts_x = np.copy(ts_x_orig)

    kmeans = KMeans(n_clusters)
    kmeans.fit(np.append(tr_x,ts_x,axis=0))

    tf_tr_x = kmeans.transform(tr_x)
    tf_ts_x = kmeans.transform(ts_x)

    tf_tr_x,tf_ts_x = normalize_data(tf_tr_x,tf_ts_x,normz,axis)

    return (tr_ids,tf_tr_x,tr_y),(ts_ids,tf_ts_x)


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

from sklearn.cross_validation import KFold
def avg_cross_validator(tr_all_all, ts_all_all, estimators, avg_type='weighted'):

    logloss_everything = []
    for f_i,(tr_all,ts_all) in enumerate(zip(tr_all_all,ts_all_all)):
        print('Running features set: ',f_i)
        tr_ids,tr_x,tr_y = tr_all
        ts_ids,ts_x = ts_all
        kf = KFold(tr_x.shape[0],n_folds=5)

        num_diff_est = len(estimators[0])
        print(num_diff_est, ' different estimators detected')
        if avg_type is 'mean':
            weights = [1/num_diff_est for _ in range(num_diff_est)]
        elif avg_type is 'weighted':
            if num_diff_est>1:
                weights = [0.7]
                for e_i in range(num_diff_est-1):
                    weights.append((1-0.3)/(num_diff_est-1))
            else:
                weights = [1.0]
        logloss_for_all_folds = []
        for kf_i,(tr_i,v_i) in enumerate(kf):
            print('\t Running for fold: ',kf_i)
            x_train, x_test = tr_x[tr_i],tr_x[v_i]
            y_train, y_test = tr_y[tr_i],tr_y[v_i]


            probs = None
            e_i = 0
            for k,v in estimators[f_i].items():
                if k is 'xgb':
                    est = MyXGBClassifier(n_rounds=v['n_rounds'],eta=v['eta'],max_depth=v['max_depth'])
                elif k is 'knn':
                    est = KNeighborsClassifier(n_neighbors=v['n_neighbors'])
                print('\t\t Running estimator (',k,')')
                est.fit(x_train,y_train)
                print('\t\t Logloss (',k,'): ',logloss(est.predict_proba(x_test),y_test))
                if probs is None:
                    probs = weights[e_i]*est.predict_proba(x_test)
                else:
                    probs = np.add(probs,weights[e_i]*est.predict_proba(x_test))
                e_i += 1
            print('\t Mean logloss for fold ',kf_i,': ',logloss(probs,y_test))
            logloss_for_all_folds.append(logloss(probs,y_test))
        print('\t mean logloss for all folds: ',np.mean(logloss_for_all_folds))
        logloss_everything.append(np.mean(logloss_for_all_folds))

    print('Mean logloss for Cross Validate: ',np.mean(logloss_everything))

import xgboost as xgb
class MyXGBClassifier(object):

    def __init__(self, n_rounds=100, **params):
        self.params = params
        self.params.update({'booster':'gbtree'})
        self.params.update({'silent':1})
        self.params.update({'objective': 'multi:softprob'})
        self.params.update({'num_class': 3})
        self.params.update({'eval_metric':'mlogloss'})
        self.params.update({'lambda':0.9})
        self.params.update({'alpha':0.9})
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
        dtest = xgb.DMatrix(X)
        Y = self.clf.predict(dtest)
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


from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import cross_val_score
if __name__ == '__main__':

    #f_names = ['features_2','features_dl_all','features_2_cat_tree']
    normz = None
    axis = 0
    cv_type = 'individual' # individual or all
    print('Normalizing with ',normz)
    f_names = ['features_2']

    if cv_type == 'individual':

        for fn in f_names:
            print('Loading data ... from: ',fn)

            tr_v_all,ts_all,correct_ids =load_teslstra_data_v3(fn+'_train.csv',fn+'_test.csv',None)

            #tr_v_log,ts_log = get_scale_log_x_plus_1(tr_v_all,ts_all,normz,axis)
            #tr_v_exp,ts_exp = get_exp_decay_fimp(tr_v_all,ts_all,0.95,normz,axis)
            #tr_v_sqr,ts_sqr = get_pow(tr_v_all,ts_all,2,normz,axis)
            #tr_v_corr,ts_corr = get_correlated_removed(tr_v_all,ts_all,0.9,normz,axis)
            #tr_v_8km,ts_8km = get_kmeans_features(tr_v_all,ts_all,8,normz)
            #tr_v_128km,ts_128km = get_kmeans_features(tr_v_all,ts_all,128,normz)
            #tr_v_1024km,ts_1024km = get_kmeans_features(tr_v_all,ts_all,512,normz)


            #transformed_features = [('original',tr_v_all,ts_all),('log',tr_v_log,ts_log),('corr',tr_v_corr,ts_corr)]
            transformed_features = [('original',tr_v_all,ts_all)]
            param = {}
            param['objective'] = 'multi:softprob'
            param['booster'] = 'gbtree'
            param['eta'] = 0.03  # high eta values give better perf
            param['max_depth'] = 10
            param['silent'] = 1
            param['lambda'] = 0.95
            param['alpha'] = 0.95
            #param['nthread'] = 4
            param['subsample']=0.9
            param['colsample_bytree']=0.9
            param['num_class'] = 3
            param['eval_metric']='mlogloss'
            param['num_rounds'] = 600

            for tf in transformed_features:
                print('Cross validation for: ', tf[0])
                tr_x_tf = tf[1][1]
                tr_y_tf = tf[1][2]
                print('Train (x,y): ',tr_x_tf.shape,',',tr_y_tf.shape)

                dtrain = xgb.DMatrix(tr_x_tf, label=tr_y_tf)
                history = xgb.cv(param,dtrain,param['num_rounds'],nfold=5,metrics={'mlogloss'})
                print('Xgboost',np.min(history['test-mlogloss-mean']),' at ',np.argmin(history['test-mlogloss-mean']),' iteration')

                #knn_cross_vals = cross_val_score(KNeighborsClassifier(n_neighbors=256),tr_x_tf,tr_y_tf,'log_loss',5)
                #print('KNN: ',-np.mean(knn_cross_vals))

                print()

    elif cv_type == 'all':

        tr_transformed_features = []
        ts_transformed_features = []
        for fn in f_names:
            tr_v_all,ts_all,correct_ids =load_teslstra_data_v3(fn+'_train.csv',fn+'_test.csv',None)
            tr_v_log,ts_log = get_scale_log_x_plus_1(tr_v_all,ts_all,normz)
            tr_v_exp,ts_exp = get_exp_decay_fimp(tr_v_all,ts_all,0.95,normz)
            #tr_v_corr,ts_corr = get_correlated_removed(tr_v_all,ts_all,0.9,normz)
            tr_transformed_features.extend(
                    [tr_v_all,tr_v_log])
            ts_transformed_features.extend(
                    [ts_all,ts_log]
            )

        param = {}
        param['eta'] = 0.05  # high eta values give better perf
        param['max_depth'] = 8
        param['subsample']=0.9
        param['n_rounds'] = 150

        knn_param = {'n_neighbors':256}

        #estimators = [(MyXGBClassifier(n_rounds=150,eta=0.05,max_depth=8),
        #               KNeighborsClassifier(n_neighbors=256)) for _ in range(len(tr_transformed_features))]
        estimators = [{'xgb':param} for _ in range(len(tr_transformed_features))]
        print('Built ',len(estimators),' estimators')
        avg_cross_validator(tr_transformed_features,ts_transformed_features,estimators,'weighted')
    #tr_all,v_all,weigts = divide_test_valid(tr_v_all,None)
