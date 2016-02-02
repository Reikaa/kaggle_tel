import xgboost as xgb
import numpy as np

def load_teslstra_data(remove_header=False,start_idx=0):
    import csv
    train_set = []
    valid_set = []
    test_set = []
    my_test_ids = []
    correct_order_test_ids = []
    with open('deepnet_features.csv', 'r',newline='') as f:
        reader = csv.reader(f)
        data_x = []
        data_y = []
        valid_x = []
        valid_y = []
        #valid_idx = np.random.randint(0,7000,size=(100,)).tolist()
        valid_idx = [i for i in range(400)]
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

    '''with open('deepnet_features.csv', 'r',newline='') as f:
        reader = csv.reader(f)
        data_x = []
        for i,row in enumerate(reader):
            if remove_header and i==0:
                continue
            data_x.append(np.asarray(row[start_idx:-1],dtype=np.float32).tolist())
            my_test_ids.append(int(row[0]))
        test_set = [data_x]'''

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


if __name__ == '__main__':
    print('Loading data ...')
    tr,v,test,my_ids,correct_ids =load_teslstra_data(False,0)

    tr_x,tr_y = tr
    v_x,v_y = v
    tr_big_x = []
    tr_big_x.extend(v_x)
    tr_big_x.extend(tr_x)

    tr_big_y =[]
    tr_big_y.extend(v_y)
    tr_big_y.extend(tr_y)

    #test_x = test[0]
    #test_dummy_y = [0 for _ in range(len(test_x))]
    xg_train = xgb.DMatrix( np.asarray(tr_x,dtype=np.float32), label=np.asarray(tr_y,dtype=np.float32))
    xg_valid = xgb.DMatrix( np.asarray(v_x,dtype=np.float32), label=np.asarray(v_y,dtype=np.float32))
    #xg_test = xgb.DMatrix( np.asarray(test_x,dtype=np.float32), label=np.asarray(test_dummy_y,dtype=np.float32))

    print('Defining parameters ...')
    param = {}
    # use softmax multi-class classification
    param['objective'] = 'multi:softprob'
    # scale weight of positive examples
    param['booster'] = 'gbtree'
    param['eta'] = 0.1  # high eta values give better perf
    param['max_depth'] = 5
    param['silent'] = 1
    param['lambda'] = 0.99
    param['alpha'] = 0.99
    param['nthread'] = 4
    param['num_class'] = 3
    param['eval_metric']='mlogloss'

    #eval list is used to keep track of performance
    evallist = [ (xg_train,'train'), (xg_valid, 'valid') ]
    epochs = 1
    num_round = 500

    best_eta,best_depth,best_reg = 0,0,0
    best_eta_round,best_depth_round,best_reg_round = 0,0,0
    print('\nCross validation ...')
    #history = xgb.cv(param, xg_train, num_round, nfold=5,metrics={'mlogloss'}, seed = 4675)
    # hitory._get_values is a num_round x 4 matrix
    #print(history)

    # Cross validate eta
    etas = [0.01, 0.1, 0.2]
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
    max_depths = [5, 10, 20]
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

    reg = [[0.9,0.9], [0.95,0.95], [0.99,0.99]]
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


    '''print('\nTraining ...')

    bst = xgb.train(param, xg_train, num_round, evallist)
    ptrain = bst.predict(xg_train)
    #ptrain_flat =np.asarray(ptrain).flatten().tolist()  #multiclass need the margin array flattened
    #xg_train.set_base_margin(ptrain_flat)

    #pvalid = bst.predict(xg_valid,output_margin=True)
    #pvalid_flat =np.asarray(pvalid).flatten().tolist()  #multiclass need the margin array flattened
    #xg_valid.set_base_margin(pvalid_flat)

    # get prediction

    pred_test = bst.predict(xg_test)


    import csv
    with open('xgboost_output.csv', 'w',newline='') as f:
        writer = csv.writer(f)
        for id in correct_ids:
            c_id = my_ids.index(id)
            probs = pred_test[int(c_id)]
            row = [id,probs[0], probs[1], probs[2]]
            writer.writerow(row)'''


