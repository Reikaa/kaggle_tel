import xgboost as xgb
import numpy as np

def load_teslstra_data():
    import csv
    train_set = []
    valid_set = []
    test_set = []
    with open('features_train.csv', 'r',newline='') as f:
        reader = csv.reader(f)
        data_x = []
        data_y = []
        valid_x = []
        valid_y = []
        valid_idx = np.random.randint(0,7000,size=(100,)).tolist()
        for i,row in enumerate(reader):
            if i==0:
                continue
            if not i in valid_idx:
                data_x.append(row[2:-1])
                data_y.append(row[-1])
            else:
                valid_x.append(row[2:-1])
                valid_y.append(row[-1])

        train_set = (data_x,data_y)
        valid_set = (valid_x,valid_y)

    with open('features_test.csv', 'r',newline='') as f:
        reader = csv.reader(f)
        data_x = []
        for i,row in enumerate(reader):
            if i==0:
                continue
            data_x.append(row[2:-1])
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
    tr_big_x.extend(v_x)
    tr_big_x.extend(tr_x)

    tr_big_y =[]
    tr_big_y.extend(v_y)
    tr_big_y.extend(tr_y)

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
    param['eta'] = 0.5  # high eta values give better perf
    param['max_depth'] = 3
    param['silent'] = 1
    param['lambda'] = 0.5
    param['alpha'] = 0.5
    param['nthread'] = 4
    param['num_class'] = 3
    param['eval_metric']='mlogloss'

    #eval list is used to keep track of performance
    evallist = [ (xg_train,'train'), (xg_valid, 'valid') ]
    epochs = 1
    num_round = 1000

    best_eta,best_depth,best_reg = 0,0,0

    print('\nCross validation ...')
    #history = xgb.cv(param, xg_train, num_round, nfold=5,metrics={'mlogloss'}, seed = 4675)
    # hitory._get_values is a num_round x 4 matrix
    #print(history)

    # Cross validate eta
    etas = [0.01, 0.5, 0.9]
    min_idx_eta = 0
    prev_mean = np.inf
    for i,e in enumerate(etas):
        param['eta'] = e
        history = xgb.cv(param, xg_train, num_boost_round=num_round,nfold=5,metrics={'mlogloss'}, seed = 4675)
        mean_vals = []
        stdev_vals = []
        for h in history:
            str_test_loss = h.split('\t')[1]
            str_val = str_test_loss.split(':')[1]
            mean_vals.append(float(str_val.split('+')[0]))
            stdev_vals.append(float(str_val.split('+')[1]))
            if np.min(mean_vals)<prev_mean:
                min_idx_eta = i
                prev_mean = np.min(mean_vals)

    best_eta = etas[min_idx_eta]
    param['eta']=best_eta

        # Cross validate max_depth
    max_depths = [2, 10, 100]
    min_idx_depth = 0
    prev_mean = np.inf
    for i,d in enumerate(max_depths):
        param['max_depth'] = d
        history = xgb.cv(param, xg_train, num_boost_round=num_round, nfold=5,metrics={'mlogloss'}, seed = 4675)
        mean_vals = []
        stdev_vals = []
        for h in history:
            str_test_loss = h.split('\t')[1]
            str_val = str_test_loss.split(':')[1]
            mean_vals.append(float(str_val.split('+')[0]))
            stdev_vals.append(float(str_val.split('+')[1]))
            if np.min(mean_vals)<prev_mean:
                min_idx_depth = i
                prev_mean = np.min(mean_vals)

    best_depth = max_depths[min_idx_depth]
    param['max_depth']= best_depth
    # Cross validate alpha lambda

    reg = [[0.95,0.95], [0.1,0.1], [0.5,0.5]]
    min_idx_reg = 0
    prev_mean = np.inf
    for i,arr in enumerate(reg):
        param['alpha'] = arr[0]
        param['lambda'] = arr[1]
        history = xgb.cv(param, xg_train, num_boost_round=num_round, nfold=5,metrics={'mlogloss'}, seed = 4675)
        mean_vals = []
        stdev_vals = []
        for h in history:
            str_test_loss = h.split('\t')[1]
            str_val = str_test_loss.split(':')[1]
            mean_vals.append(float(str_val.split('+')[0]))
            stdev_vals.append(float(str_val.split('+')[1]))
            if np.min(mean_vals)<prev_mean:
                min_idx_reg = i
                prev_mean = np.min(mean_vals)

    best_reg = reg[min_idx_reg]
    param['alpha']=best_reg[0]
    param['lambda']=best_reg[1]

    print('Best eta: ',best_eta)
    print('Best depth: ',best_depth)
    print('Best reg (alpha lambda): ',best_reg)


    '''print('\nTraining ...')

    bst = xgb.train(param, xg_train, num_round, evallist)
    ptrain = bst.predict(xg_train,output_margin=True)
    ptrain_flat =np.asarray(ptrain).flatten().tolist()  #multiclass need the margin array flattened
    xg_train.set_base_margin(ptrain_flat)

    pvalid = bst.predict(xg_valid,output_margin=True)
    pvalid_flat =np.asarray(pvalid).flatten().tolist()  #multiclass need the margin array flattened
    xg_valid.set_base_margin(pvalid_flat)

    # get prediction

    pred = bst.predict(xg_valid);
    pred_test = bst.predict(xg_test)

    print(pred_test)
    import csv
    with open('output.csv', 'w',newline='') as f:
        writer = csv.writer(f)
        for p in pred_test:
            row = [0, 0, 0]
            row[int(p)] = 1
            writer.writerow(row)'''

