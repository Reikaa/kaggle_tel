import xgboost as xgb
import numpy as np

def load_teslstra_data(remove_header=False,start_idx=0):
    import csv
    train_set = []
    valid_set = []
    test_set = []
    my_test_ids = []
    correct_order_test_ids = []
    with open('features_modified_train.csv', 'r',newline='') as f:
        reader = csv.reader(f)
        data_x = []
        data_y = []
        valid_x = []
        valid_y = []
        valid_idx = np.random.randint(0,7000,size=(100,)).tolist()

        for i,row in enumerate(reader):
            if remove_header and i==0:
                continue
            if not i in valid_idx:
                data_x.append(np.asarray(row[start_idx:-1],dtype=np.float32).tolist())
                data_y.append(int(row[-1]))
            else:
                valid_x.append(np.asarray(row[start_idx:-1],dtype=np.float32).tolist())
                valid_y.append(int(row[-1]))

        train_set = (data_x,data_y)
        valid_set = (valid_x,valid_y)

    with open('features_modified_test.csv', 'r',newline='') as f:
        reader = csv.reader(f)
        data_x = []
        for i,row in enumerate(reader):
            if remove_header and i==0:
                continue
            data_x.append(np.asarray(row[start_idx:-1],dtype=np.float32).tolist())
            my_test_ids.append(int(row[0]))
        test_set = [data_x]

    with open('test.csv', 'r',newline='') as f:
        reader = csv.reader(f)

        for i,row in enumerate(reader):
            if i==0:
                continue
            correct_order_test_ids.append(int(row[0]))

    train_x,train_y = train_set
    valid_x,valid_y = valid_set
    test_x = test_set

    all_data = [(train_x,train_y),(valid_x,valid_y),(test_x),my_test_ids,correct_order_test_ids]

    return all_data

def load_teslstra_data_v2(train_file,test_file,remove_header=False,start_col=1):

    '''
            0th class 6.58, 1st class 2.58, 2nd class 1 (ratios)
    '''
    import csv
    train_set = []
    valid_set = []
    test_set = []
    my_train_ids = []
    my_valid_ids = []
    my_train_ids_v2 = [[],[],[]]
    my_test_ids = []
    correct_order_test_ids = []

    with open(train_file, 'r',newline='') as f:
        reader = csv.reader(f)
        data_x = []
        data_y = []
        valid_x,valid_y = [],[]
        data_x_v2 = [[],[],[]]


        for i,row in enumerate(reader):
            if remove_header and i==0:
                continue

            # first 2 columns are ID and location
            output = int(row[-1])
            data_x_v2[output].append(row[start_col:-1])
            my_train_ids_v2[output].append(row[0])

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

        train_set = (data_x,data_y)
        valid_set = (valid_x,valid_y)

    with open(test_file, 'r',newline='') as f:
        reader = csv.reader(f)
        data_x = []
        for i,row in enumerate(reader):
            if remove_header and i==0:
                continue
            # first 2 columns are ID and location
            data_x.append(row[start_col:])
            my_test_ids.append(int(row[0]))
        test_set = [data_x]

    with open('test.csv', 'r',newline='') as f:
        reader = csv.reader(f)

        for i,row in enumerate(reader):
            if i==0:
                continue
            correct_order_test_ids.append(int(row[0]))


    print('Train: ',len(train_set[0]),' x ',len(train_set[0][0]))
    print('Valid: ',len(valid_set[0]),' x ',len(valid_set[0][0]))
    print('Test: ',len(test_set[0]),' x ',len(test_set[0][0]))

    all_data = [train_set,valid_set,test_set[0],my_test_ids,correct_order_test_ids,my_train_ids,my_valid_ids]

    return all_data

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
        self.clf = None

    def train_clf(self,tr_all,v_all,weights=None):

        tr_ids,tr_x,tr_y = tr_all
        v_ids,v_x,v_y = v_all

        #eval list is used to keep track of performance

        print('\nTraining ...')

        self.clf = xgb.XGBClassifier(max_depth=self.param['max_depth'], learning_rate=self.param['learning_rate'],
                                     n_estimators=self.param['n_estimators'], silent=True, objective='multi:softprob',
                                     nthread=self.param['nthread'], gamma=0, min_child_weight=1, max_delta_step=0,
                                     subsample=1, colsample_bytree=1, base_score=0.5, seed=0)

        if weights is not None:
            self.clf.fit(np.asarray(tr_x), np.asarray(tr_y),np.asarray(weights)/np.max(weights))
        else:
            self.clf.fit(np.asarray(tr_x), np.asarray(tr_y))


        #self.bst = xgb.train(self.param, xg_train, num_round, evallist,early_stopping_rounds = 10)
        pred_train = self.clf.predict_proba(np.asarray(tr_x))

        # get prediction
        pred_valid = self.clf.predict_proba(np.asarray(v_x))

        pred_y = [np.argmax(arr) for arr in pred_train]


        logloss_tr = self.logloss(tr_ids,pred_train,tr_y,weights)
        logloss_v = self.logloss(v_ids,pred_valid,v_y)
        print('XGB logloss (train): ',logloss_tr)
        print('XGB logloss (valid): ',logloss_v,'\n')
        return tr_ids,pred_y,tr_y

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

    def train(self,tr_all,v_all,weights=None):

        tr_ids,tr_x,tr_y = tr_all
        v_ids,v_x,v_y = v_all

        num_round = self.param['num_rounds']
        if weights is not None:
            xg_train = xgb.DMatrix( np.asarray(tr_x,dtype=np.float32), label=np.asarray(tr_y,dtype=np.float32),weight=np.asarray(weights)/np.max(weights))
        else:
            xg_train = xgb.DMatrix( np.asarray(tr_x,dtype=np.float32), label=np.asarray(tr_y,dtype=np.float32))
        xg_valid = xgb.DMatrix( np.asarray(v_x,dtype=np.float32), label=np.asarray(v_y,dtype=np.float32))

        #eval list is used to keep track of performance
        evallist = [(xg_train,'train'), (xg_valid, 'eval')]

        print('\nTraining ...')

        epochs = 10
        for ep in range(epochs):
            self.bst = xgb.train(self.param, xg_train, 10, evallist)

            #self.bst = xgb.train(self.param, xg_train, num_round, evallist,early_stopping_rounds = 10)
            pred_train = self.bst.predict(xg_train,output_margin=True)
            # get prediction
            pred_valid = self.bst.predict(xg_valid,output_margin=True)

            xg_valid.set_base_margin(pred_valid.flatten())
            xg_train.set_base_margin(pred_train.flatten())


        final_pred_valid = self.bst.predict(xg_valid)

        y_mat = []
        for i in range(len(v_ids)):
            temp = [0,0,0]
            temp[v_y[i]] = 1.
            y_mat.append(temp)

        logloss = self.logloss(v_ids, final_pred_valid, v_y)
        print('Valid: ',logloss)
        pred_y = [np.argmax(arr) for arr in pred_train]

        return tr_ids,pred_y,tr_y


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

    def test(self,test_x):
        test_dummy_y = [0 for _ in range(len(test_x))]
        xg_test = xgb.DMatrix( np.asarray(test_x,dtype=np.float32), label=np.asarray(test_dummy_y,dtype=np.float32))

        pred_test = self.bst.predict(xg_test)

        return pred_test

    def test_clf(self,test_x):
        pred_test = self.clf.predict_proba(np.asarray(test_x))
        return pred_test




if __name__ == '__main__':
    print('Loading data ...')
    tr,v,test,ts_ids,correct_ids,tr_ids,v_ids =load_teslstra_data_v2('features_modified_2_train.csv','features_modified_2_test.csv',True,2)
    #tr,v,test,my_ids,correct_ids =load_teslstra_data_v2('deepnet_features_train_0.csv','deepnet_features_test_0.csv',False,1)

    tr_x,tr_y = tr
    v_x,v_y = v
    tr_big_x = []
    tr_big_x.extend(v_x)
    tr_big_x.extend(tr_x)

    tr_big_y =[]
    tr_big_y.extend(v_y)
    tr_big_y.extend(tr_y)

    test_x = test


    print('Defining parameters ...')
    param = {}
    # use softmax multi-class classification
    param['objective'] = 'multi:softprob'
    # scale weight of positive examples
    param['booster'] = 'gbtree'
    param['eta'] = 0.1  # high eta values give better perf
    param['max_depth'] = 10
    param['silent'] = 1
    param['lambda'] = 0.1
    param['alpha'] = 0.1
    param['nthread'] = 4
    param['num_class'] = 3
    param['eval_metric']='mlogloss'
    param['num_rounds'] = 2500

    param2= {}
    param2['learning_rate'] = 0.1
    param2['n_estimators'] = 100
    param2['max_depth'] = 10
    param2['nthread'] = 4
    xgboost = XGBoost(param)
    xgboost2 = XGBoost(param2)

    #weights = [0.4 if tr_y[i]==2 else 0.3 for i in range(len(tr_y))]
    xgboost.train((tr_ids,tr_x,tr_y),(v_ids,v_x,v_y),None)
    pred_test = xgboost.test(test_x)

    xgboost2.train_clf((tr_ids,tr_x,tr_y),(v_ids,v_x,v_y),None)
    pred_test = xgboost2.test_clf(test_x)

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

