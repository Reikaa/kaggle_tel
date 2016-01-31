import xgboost as xgb
import numpy as np

def load_teslstra_data():
    import csv
    train_set = []
    valid_set = []
    test_set = []
    with open('deepnet_features_train.csv', 'r',newline='') as f:
        reader = csv.reader(f)
        data_x = []
        data_y = []
        for i,row in enumerate(reader):
            data_x.append(row[:-2])
            data_y.append(row[-1])
        train_set = (data_x,data_y)

    with open('deepnet_features_valid.csv', 'r',newline='') as f:
        reader = csv.reader(f)
        data_x = []
        data_y = []
        for i,row in enumerate(reader):
            data_x.append(row[:-2])
            data_y.append(row[-1])
        valid_set = (data_x,data_y)

    with open('deepnet_features_test.csv', 'r',newline='') as f:
        reader = csv.reader(f)
        data_x = []
        for i,row in enumerate(reader):
            data_x.append(row[:-1])
        test_set = [data_x]

    train_x,train_y = train_set
    valid_x,valid_y = valid_set
    test_x = test_set

    all_data = [(train_x,train_y),(valid_x,valid_y),(test_x)]

    return all_data

def cross_validate(param,values,xg_train,model,params,num_rounds):

    for v in values:
        params[param] = v
        history = xgb.cv(param, xg_train, num_round, nfold=5,metrics={'mlogloss'}, seed = 4675)

        arr = history._get_values



if __name__ == '__main__':
    print('Loading data ...')
    tr,v,test =load_teslstra_data()

    tr_x,tr_y = tr
    v_x,v_y = v
    tr_big_x = []
    tr_big_x.extend(tr_x)
    tr_big_x.extend(v_x)
    tr_big_y =[]
    tr_big_y.extend(tr_y)
    tr_big_y.extend(v_y)

    test_x = test[0]
    test_dummy_y = [0 for _ in range(len(test_x))]
    xg_train = xgb.DMatrix( np.asarray(tr_x,dtype=np.float32), label=np.asarray(tr_y,dtype=np.float32))
    xg_valid = xgb.DMatrix( np.asarray(v_x,dtype=np.float32), label=np.asarray(v_y,dtype=np.float32))
    xg_test = xgb.DMatrix( np.asarray(test_x,dtype=np.float32), label=np.asarray(test_dummy_y,dtype=np.float32))

    print('Defining parameters ...')
    param = {}
    # use softmax multi-class classification
    param['objective'] = 'multi:softmax'
    # scale weight of positive examples
    param['booster'] = 'gbtree'
    param['eta'] = 0.9  # high eta values give better perf
    param['max_depth'] = 20
    param['silent'] = 1
    param['lambda'] = 0.001
    param['alpha'] = 0.001
    param['nthread'] = 4
    param['num_class'] = 3
    param['eval_metric']='mlogloss'

    #eval list is used to keep track of performance
    evallist = [ (xg_train,'train'), (xg_valid, 'valid') ]
    epochs = 10
    num_round = 500

    print('\nCross validation ...')
    #history = xgb.cv(param, xg_train, num_round, nfold=5,metrics={'mlogloss'}, seed = 4675)
    # hitory._get_values is a num_round x 4 matrix
    #print(history)


    print('\nTraining ...')
    for _ in range(epochs):
        bst = xgb.train(param, xg_train, num_round, evallist,early_stopping_rounds=100)
        ptrain = bst.predict(xg_train,output_margin=True)
        ptrain_flat =np.asarray(ptrain).flatten().tolist()  #multiclass need the margin array flattened
        xg_train.set_base_margin(ptrain_flat)
        '''
        if np.random.random() < 0.1:
            print('\nValid ...')
            bst = xgb.train(param, xg_valid, num_round, evallist,early_stopping_rounds=100)
            pvalid = bst.predict(xg_valid,output_margin=True)
            pvalid_flat = np.asarray(pvalid).flatten().tolist()
            xg_valid.set_base_margin(pvalid_flat)'''

    # get prediction

    pred = bst.predict( xg_valid );
    pred_test = bst.predict(xg_test)

    print(pred_test)
    import csv
    with open('output.csv', 'w',newline='') as f:
        writer = csv.writer(f)
        for p in pred_test:
            row = [0, 0, 0]
            row[int(p)] = 1
            writer.writerow(row)

