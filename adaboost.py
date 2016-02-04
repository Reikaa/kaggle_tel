__author__ = 'Thushan Ganegedara'

import numpy as np

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

        valid_idx = np.random.randint(0,len(data_x_v2[2]),size=(100,)).tolist()

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

    train_x,train_y = train_set
    valid_x,valid_y = valid_set
    test_x = np.asarray(test_set[0])

    print('Train: ',len(train_set[0]),' x ',len(train_set[0][0]))
    print('Valid: ',len(valid_set[0]),' x ',len(valid_set[0][0]))
    print('Test: ',len(test_set[0]),' x ',len(test_set[0][0]))

    all_data = [(train_x,train_y),(valid_x,valid_y),(test_x),my_test_ids,correct_order_test_ids,my_train_ids,my_valid_ids]

    return all_data

tr_data, v_data, test_x, ts_ids, correct_ts_ids, tr_ids, v_ids = load_teslstra_data_v2('features_modified_train.csv','features_modified_test.csv',True,1)
tr_x,tr_y  = tr_data
v_x,v_y = v_data

# a model should take 2 np array tuples as (tr_x,tr_y),(v_x,v_y)
# and output pred labels, actual labels

from adaboost_classifiers import UseSDAE,UseXGBoost
import collections


params_deeplearn = collections.defaultdict()
params_deeplearn['batch_size'] = 1
params_deeplearn['in_size'] = 254
params_deeplearn['out_size'] = 3
params_deeplearn['hid_sizes'] = [100]
params_deeplearn['learning_rate'] = 0.25
params_deeplearn['pre_epochs'] = 5
params_deeplearn['fine_epochs'] = 100
params_deeplearn['lam'] = 0.0
params_deeplearn['act'] = 'sigmoid'

xgb_param = {}
# use softmax multi-class classification
xgb_param['objective'] = 'multi:softprob'
# scale weight of positive examples
xgb_param['booster'] = 'gbtree'
xgb_param['eta'] = 0.2  # high eta values give better perf
xgb_param['max_depth'] = 5
xgb_param['silent'] = 1
xgb_param['lambda'] = 0.5
xgb_param['alpha'] = 0.5
xgb_param['nthread'] = 4
xgb_param['num_class'] = 3
xgb_param['eval_metric']= 'mlogloss'
xgb_param['num_rounds'] = 300

sdae = UseSDAE(params_deeplearn)
xgboost = UseXGBoost(xgb_param)
models_funcs = [[sdae.train,sdae.get_labels],[xgboost.train,xgboost.get_labels]] # length m
alpha = [0 for _ in range(len(models_funcs))] # m long list
num_classes = 3
thresh = 0.001

w = [1./len(tr_y) for _ in range(len(tr_y))]
for m, funcs in enumerate(models_funcs):

    train,get_labels = funcs
    if m == 0:
        selected_x = tr_x
        selected_y = tr_y
    else:
        w_thresh = [i for i in range(len(w)) if w[i]> thresh]
        print('Selecting a subset of ',len(w_thresh),' examples')
        selected_x = [tr_x[i] for i in w_thresh]
        selected_y = [tr_y[i] for i in w_thresh]

    train((tr_ids,selected_x,selected_y),(v_ids,v_x,v_y))

    # shoud return a tuple (ids, pred, actual)
    ids, pred_y, act_y = get_labels()
    print(len(ids))
    print(len(pred_y))
    print(len(act_y))
    vec = np.multiply(w,[1 if pred_y[i]!=act_y[i] else 0 for i in range(len(pred_y))])
    err_m = np.sum(vec)/np.sum(w)
    print('Err for the ',m,' th model: ',err_m)
    alpha[m] = np.log((1-err_m)/err_m) + np.log(3-1)
    print('Weight for the ',m,' th model: ',alpha[m])

    exp_term = np.exp([1*alpha[m] if pred_y[i]!=act_y[i] else 0 for i in range(len(pred_y))])
    w = np.multiply(w,exp_term)

    # renormalize
    w = np.asarray(w)*1.0/np.sum(w)




