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


def weighted_softmax_obj(weights, preds, dtrain):

    labels = np.asarray(dtrain.get_label()).reshape((len(preds),1))
    lbl_mat = np.zeros((len(labels),3),dtype=np.float32)

    for i,j in zip(range(len(labels)),labels):
        lbl_mat[int(i)][int(j)] = 1.
    #print(preds)
    preds = 1.0 / (1.0 + np.exp(-preds))
    #preds = np.exp(preds)/np.sum(np.exp(preds))

    pred_val_correct_class = np.asarray([np.argmax(preds[int(k)]) for k in labels]).reshape((len(preds),1))

    grad = (np.asarray(weights).reshape((len(labels),1))/np.max(weights))*\
           (np.asarray(preds) - np.asarray(lbl_mat))

    #print('grad info')
    #print('gsize: ',grad.shape)
    #print('gmean: ',np.mean(grad,axis=0))
    #print('gmin: ',np.min(grad,axis=0))
    #print('gmax: ',np.max(grad,axis=0))
    grad = np.asarray(grad,dtype=np.float32).flatten()

    hess = np.ones((len(labels)*3,1))/300
    #hess = (np.asarray(weights).reshape((len(labels),1))/np.max(weights)) *np.asarray(preds) * (1.0-np.asarray(preds))
    #print('hess info')
    #print('hmean: ',np.mean(hess,axis=0))
    #print('hsize: ',hess.shape)
    #print('hmin: ',np.min(hess,axis=0))
    #print('hmax: ',np.max(hess,axis=0))
    #hess = np.asarray(hess,dtype=np.float32).flatten()


    # need to have # of data points * num of classes amount of entries in each list grad,hess
    return grad,hess

def weighted_eval_metric(weights,preds,dtrain):
    labels = dtrain.get_label()
    # return a pair metric_name, result
    # since preds are margin(before logistic transformation, cutoff at 0)
    pred_val_correct_class = [np.argmax(preds[k]) for k in labels]

    if len(weights)==len(preds):
        temp = np.multiply(np.asarray(weights)/np.max(weights),
                    [1 if labels[j] != pred_val_correct_class[j] else 0 for j in range(len(labels))])
        return 'werror', np.mean(temp)
    else:
        return 'werror', np.mean(np.asarray([1 if labels[j] != pred_val_correct_class[j] else 0 for j in range(len(labels))]))

import functools

class XGBoost(object):

    def __init__(self,params):
        self.param = params
        self.bst = None

    def logloss(self,ids,probs,actuals,weights=None):
        if weights is not None:
            weights = np.asarray(weights)/np.max(weights)

        logloss = 0.0
        for i,id in enumerate(ids):
            tmp_y = [0.,0.,0.]
            tmp_y[actuals[i]]=1.
            norm_v_probs = np.asarray(probs[i])
            if any(norm_v_probs)==1.:
                norm_v_probs = np.asarray([np.max([np.min([p,1-1e-15]),1e-15]) for p in norm_v_probs])
            if weights is not None:
                logloss += np.sum(np.asarray(weights[i])*np.asarray(tmp_y)*np.log(np.asarray(norm_v_probs)))
            else:
                logloss += np.sum(np.asarray(tmp_y)*np.log(np.asarray(norm_v_probs)))

        logloss = -logloss/len(ids)
        return logloss

    def train_v1(self,tr_all,v_all,weights=None,eps=3):

        tr_ids,tr_x,tr_y = tr_all
        if v_all is not None:
            v_ids,v_x,v_y = v_all

        num_round = self.param['num_rounds']
        if weights is not None:
            xg_train = xgb.DMatrix(np.asarray(tr_x,dtype=np.float32), label=np.asarray(tr_y,dtype=np.float32),weight=np.asarray(weights)/np.max(weights))
        else:
            xg_train = xgb.DMatrix(np.asarray(tr_x,dtype=np.float32), label=np.asarray(tr_y,dtype=np.float32))

        print('\nTraining ...')
        if v_all is not None:
            xg_valid = xgb.DMatrix( np.asarray(v_x,dtype=np.float32), label=np.asarray(v_y,dtype=np.float32))
            #eval list is used to keep track of performance
            evallist = [(xg_train,'train'), (xg_valid, 'eval')]

            epochs = eps
            for ep in range(epochs):
                self.bst = xgb.train(self.param, xg_train, 100, evallist,early_stopping_rounds=5)

                #self.bst = xgb.train(self.param, xg_train, num_round, evallist,early_stopping_rounds = 10)
                pred_train = self.bst.predict(xg_train,output_margin=True)
                # get prediction
                pred_valid = self.bst.predict(xg_valid,output_margin=True)

                xg_valid.set_base_margin(pred_valid.flatten())
                xg_train.set_base_margin(pred_train.flatten())

        else:
            for idx in range(10):
                self.bst.update(xg_train, idx)
                pred_train = self.bst.predict(xg_train)
                #xg_train.set_base_margin(pred_train.flatten())

                v_ids = tr_ids
                v_x = tr_x
                v_y = tr_y
                #final_pred_valid = self.bst.predict(xg_valid)
                self.logloss(v_x, v_y)

        y_mat = []
        for i in range(len(v_ids)):
            temp = [0,0,0]
            temp[v_y[i]] = 1.
            y_mat.append(temp)

        pred_y = [np.argmax(arr) for arr in pred_train]

        return tr_ids,pred_y,tr_y

    def train_v2(self,tr_all,v_all,num_boost_rounds):

        tr_ids,tr_x,tr_y = tr_all
        if v_all is not None:
            v_ids,v_x,v_y = v_all

        print(tr_x.shape,', ',tr_y.shape)
        dtrain = xgb.DMatrix(tr_x, label=tr_y)
        dvalid = xgb.DMatrix(v_x, label=v_y)
        evallist = [(dtrain,'train'), (dvalid, 'eval')]

        # don't use iterative train if using early_stop
        self.clf = xgb.train(self.param, dtrain, num_boost_rounds, evallist, early_stopping_rounds=10)

    def get_probs(self,x):
        y = np.asarray([1 for _ in range(x.shape[0])])
        dtest = xgb.DMatrix(x,label=y)
        return self.clf.predict(dtest)

    def cross_validate(self,param, tr_x,tr_y,v_x,v_y,num_round):

        best_eta,best_depth,best_reg = 0,0,0
        best_eta_round,best_depth_round,best_reg_round = 0,0,0

        xg_train = xgb.DMatrix( np.asarray(tr_x,dtype=np.float32), label=np.asarray(tr_y,dtype=np.float32))
        xg_valid = xgb.DMatrix( np.asarray(v_x,dtype=np.float32), label=np.asarray(v_y,dtype=np.float32))
        xg_test = xgb.DMatrix( np.asarray(test_x,dtype=np.float32), label=np.asarray(test_dummy_y,dtype=np.float32))

        ('\nCross validation ...')
        #history = xgb.cv(param, xg_train, num_round, nfold=5,metrics={'mlogloss'}, seed = 4675)
        # hitory._get_values is a num_round x 4 matrix
        #print(history)

        # Cross validate eta
        etas = [0.2, 0.5, 0.9]
        min_idx_eta = 0
        min_mean = np.inf
        for i,e in enumerate(etas):
            param['eta'] = e
            history = xgb.cv(param, xg_train, num_boost_round=num_round,nfold=5,metrics={'mlogloss'}, seed = 4675)

            for h_i,h in enumerate(history):
                str_test_loss = h.split('\t')[1]
                str_val = str_test_loss.split(':')[1]
                curr_mean = float(str_val.split('+')[0])
                curr_stdev = float(str_val.split('+')[1])
                if curr_mean<min_mean:
                    min_idx_eta = i
                    min_mean = curr_mean
                    best_eta_round = h_i

        best_eta = etas[min_idx_eta]
        param['eta']=best_eta

            # Cross validate max_depth
        max_depths = [3, 4, 5]
        min_idx_depth = 0
        min_mean = np.inf
        for i,d in enumerate(max_depths):
            param['max_depth'] = d
            history = xgb.cv(param, xg_train, num_boost_round=num_round, nfold=5,metrics={'mlogloss'}, seed = 4675)

            for h_i,h in enumerate(history):
                str_test_loss = h.split('\t')[1]
                str_val = str_test_loss.split(':')[1]
                mean_val=float(str_val.split('+')[0])
                stdev_val=float(str_val.split('+')[1])
                if mean_val<min_mean:
                    min_idx_depth = i
                    min_mean = mean_val
                    best_depth_round = h_i

        best_depth = max_depths[min_idx_depth]
        param['max_depth']= best_depth
        # Cross validate alpha lambda

        reg = [[0.5,0.1],[0.5,0.5],[0.1,0.5]]
        min_idx_reg = 0
        min_mean = np.inf
        for i,arr in enumerate(reg):
            param['alpha'] = arr[0]
            param['lambda'] = arr[1]
            history = xgb.cv(param, xg_train, num_boost_round=num_round, nfold=5,metrics={'mlogloss'}, seed = 4675)

            for h_i,h in enumerate(history):
                str_test_loss = h.split('\t')[1]
                str_val = str_test_loss.split(':')[1]
                mean_val=float(str_val.split('+')[0])
                stdev_val=float(str_val.split('+')[1])
                if mean_val<min_mean:
                    min_idx_reg = i
                    min_mean = mean_val
                    best_reg_round = h_i

        best_reg = reg[min_idx_reg]
        param['alpha']=best_reg[0]
        param['lambda']=best_reg[1]

        print('Best eta: ',best_eta,' Round: ',best_eta_round)
        print('Best depth: ',best_depth, ' Round: ',best_depth_round)
        print('Best reg (alpha lambda): ',best_reg,' Round: ',best_reg_round)

    def logloss(self,X, Y):
        xg_X = xgb.DMatrix(np.asarray(X,dtype=np.float32), label=np.asarray(Y,dtype=np.float32))
        probs = self.bst.predict(xg_X)


        logloss = 0.0
        for i in range(len(Y)):
            tmp_y = [0.,0.,0.]
            tmp_y[Y[i]]=1.
            v_probs = probs[i]
            if any(v_probs)==1.:
                v_probs = np.asarray([np.max([np.min([p,1-1e-15]),1e-15]) for p in v_probs])

            logloss += np.sum(np.asarray(tmp_y)*np.log(np.asarray(v_probs)))
        logloss = -logloss/len(Y)
        print(logloss)
        return logloss

    def test(self,test_x):
        test_dummy_y = [0 for _ in range(len(test_x))]
        xg_test = xgb.DMatrix( np.asarray(test_x,dtype=np.float32), label=np.asarray(test_dummy_y,dtype=np.float32))

        pred_test = self.bst.predict(xg_test)

        return pred_test

    def test_clf(self,test_x):
        pred_test = self.clf.predict_proba(np.asarray(test_x))
        return pred_test



from sklearn.ensemble import ExtraTreesClassifier
if __name__ == '__main__':
    print('Loading data ...')
    tr_v_all,ts_all,correct_ids =load_teslstra_data_v3('features_2_train.csv','features_2_test.csv',None)
    tr_all,v_all,weigts = divide_test_valid(tr_v_all,None)
    #tr,v,test,my_ids,correct_ids =load_teslstra_data_v2('deepnet_features_train_0.csv','deepnet_features_test_0.csv',False,1)

    tr_ids,tr_x,tr_y = tr_all
    v_ids,v_x,v_y = v_all
    ts_ids,test_x = ts_all

    tr_x = np.asarray(tr_x)
    v_x = np.asarray(v_x)
    test_x = np.asarray(test_x)

    clf = ExtraTreesClassifier()
    clf = clf.fit(tr_v_all[1], tr_v_all[2])
    fimp = np.asarray(clf.feature_importances_)
    ord_feature_idx = list(reversed(np.argsort(fimp)))
    fimp_thresh = 0.6
    # need to sort, else hard to perform delete
    train_feature_idx = ord_feature_idx[int(fimp_thresh*len(ord_feature_idx)):]
    train_feature_idx.sort()
    for fidx in reversed(train_feature_idx):
        tr_x = np.delete(tr_x,fidx,axis=1)
        v_x = np.delete(v_x,fidx,axis=1)
        test_x = np.delete(test_x,fidx,axis=1)

    print("After Transformation ...")
    print('Train: ',tr_x.shape)
    print('Valid: ',v_x.shape)
    print('Test: ',test_x.shape)

    print('Defining parameters ...')
    param = {}
    # use softmax multi-class classification
    param['objective'] = 'multi:softprob'
    # scale weight of positive examples
    param['booster'] = 'gbtree'
    param['eta'] = 0.1  # high eta values give better perf
    param['max_depth'] = 10
    param['silent'] = 1
    param['lambda'] = 0.9
    param['alpha'] = 0.9
    param['nthread'] = 4
    param['num_class'] = 3
    param['eval_metric']='mlogloss'
    param['num_rounds'] = 200

    xgboost = XGBoost(param)
    #xgboost2 = XGBoost(param)

    #weights = [0.4 if tr_y[i]==2 else 0.3 for i in range(len(tr_y))]
    xgboost.train_v2((tr_ids,np.asarray(tr_x),np.asarray(tr_y)),(v_ids,np.asarray(v_x),np.asarray(v_y)),200)

    pred_test = xgboost.get_probs(np.asarray(test_x))

    print('\n Saving out probabilities (test)')
    import csv
    with open('xgboost_output.csv', 'w',newline='') as f:
        class_dist = [0,0,0]
        writer = csv.writer(f)
        for id in correct_ids:
            c_id = ts_ids.index(id)
            probs = pred_test[int(c_id)]
            row = [id,probs[0], probs[1], probs[2]]
            class_dist[np.argmax(probs)] += 1
            writer.writerow(row)

