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
        #valid_idx = [i for i in range(400)]
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

class XGBoost(object):

    def __init__(self,params):
        self.param = params
        self.bst = None

    def train(self,tr_all,v_all):

        tr_ids,tr_x,tr_y = tr_all
        v_ids,v_x,v_y = v_all

        num_round = self.param['num_rounds']
        xg_train = xgb.DMatrix( np.asarray(tr_x,dtype=np.float32), label=np.asarray(tr_y,dtype=np.float32))
        xg_valid = xgb.DMatrix( np.asarray(v_x,dtype=np.float32), label=np.asarray(v_y,dtype=np.float32))


        #eval list is used to keep track of performance
        evallist = [ (xg_train,'train'), (xg_valid, 'valid') ]

        print('\nTraining ...')

        self.bst = xgb.train(self.param, xg_train, num_round, evallist)
        pred_train = self.bst.predict(xg_train)

        # get prediction
        pred_valid = self.bst.predict(xg_valid)

        pred_y = [np.argmax(arr) for arr in pred_train]

        return tr_ids,pred_y,tr_y

        '''
        print('\n Saving out probabilities (valid)')
        with open('xgboost_valid_probs.csv', 'w',newline='') as f:
            import csv
            writer = csv.writer(f)

            row = ['id','pred_0','pred_1','pred_2','act_0','act_1','act_2']
            writer.writerow(row)

            labels = xg_valid.get_label()
            for i,id in enumerate(v_ids):
                row = [id]
                row.extend(pred_valid[i])
                temp = [0,0,0]
                temp[int(labels[i])]=1
                row.extend(temp)
                writer.writerow(row)'''

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

    def test(self):
        xg_test = xgb.DMatrix( np.asarray(test_x,dtype=np.float32), label=np.asarray(test_dummy_y,dtype=np.float32))
        pred_test = self.bst.predict(xg_test)
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


if __name__ == '__main__':
    print('Loading data ...')
    tr,v,test,ts_ids,correct_ids,tr_ids,v_ids =load_teslstra_data_v2('features_modified_train.csv','features_modified_test.csv',True,2)
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
    test_dummy_y = [0 for _ in range(len(test_x))]

    print('Defining parameters ...')
    param = {}
    # use softmax multi-class classification
    param['objective'] = 'multi:softprob'
    # scale weight of positive examples
    param['booster'] = 'gbtree'
    param['eta'] = 0.2  # high eta values give better perf
    param['max_depth'] = 5
    param['silent'] = 1
    param['lambda'] = 0.5
    param['alpha'] = 0.5
    param['nthread'] = 4
    param['num_class'] = 3
    param['eval_metric']='mlogloss'
    param['num_rounds'] = 500






