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

    all_data = [(my_train_ids,train_x,train_y),(my_valid_ids,valid_x,valid_y),(my_test_ids,test_x),correct_order_test_ids]

    return all_data

tr_data, v_data, test_data, correct_ts_ids = load_teslstra_data_v2('features_modified_2_train.csv','features_modified_2_test.csv',True,13)

tr_ids,tr_x,tr_y  = tr_data
v_ids,v_x,v_y = v_data
ts_ids, test_x = test_data

# a model should take 2 np array tuples as (tr_x,tr_y),(v_x,v_y)
# and output pred labels, actual labels

from adaboost_classifiers import UseSDAE,UseXGBoost,UseSVM,UseRF
import collections


dl_params_1 = collections.defaultdict()
dl_params_1['batch_size'] = 50
dl_params_1['iterations'] = 1
dl_params_1['in_size'] = 398
dl_params_1['out_size'] = 3
dl_params_1['hid_sizes'] = [500]
dl_params_1['learning_rate'] = 0.75
dl_params_1['pre_epochs'] = 5
dl_params_1['fine_epochs'] = 100
dl_params_1['lam'] = 0.0
dl_params_1['act'] = 'sigmoid'

dl_params_2 = collections.defaultdict()
dl_params_2['batch_size'] = 50
dl_params_2['in_size'] = 398
dl_params_2['out_size'] = 3
dl_params_2['hid_sizes'] = [500]
dl_params_2['iterations'] = 3
dl_params_2['learning_rate'] = 0.25
dl_params_2['pre_epochs'] = 5
dl_params_2['fine_epochs'] = 100
dl_params_2['lam'] = 0.0
dl_params_2['act'] = 'sigmoid'


dl_params_3 = collections.defaultdict()
dl_params_3['batch_size'] = 50
dl_params_3['in_size'] = 398
dl_params_3['out_size'] = 3
dl_params_3['hid_sizes'] = [1500]
dl_params_3['iterations'] = 3
dl_params_3['learning_rate'] = 0.25
dl_params_3['pre_epochs'] = 5
dl_params_3['fine_epochs'] = 100
dl_params_3['lam'] = 0.0
dl_params_3['act'] = 'sigmoid'

xgb_param_1 = {}
xgb_param_1['objective'] = 'multi:softprob'
xgb_param_1['booster'] = 'gbtree'
xgb_param_1['eta'] = .5  # less eta values give better perf
xgb_param_1['max_depth'] = 5
xgb_param_1['silent'] = 1
xgb_param_1['lambda'] = 0.5
xgb_param_1['alpha'] = 0.5
xgb_param_1['nthread'] = 4
xgb_param_1['num_class'] = 3
xgb_param_1['eval_metric']= 'mlogloss'
xgb_param_1['num_rounds'] = 300
xgb_param_1['learning_rate'] = 0.1
xgb_param_1['n_estimators'] = 100

xgb_param_2 = {}
xgb_param_2['objective'] = 'multi:softprob'
xgb_param_2['booster'] = 'gblinear'
xgb_param_2['eta'] = 0.6  # less eta values give better perf
xgb_param_2['max_depth'] = 5
xgb_param_2['silent'] = 1
xgb_param_2['lambda'] = 0.5
xgb_param_2['alpha'] = 0.5
xgb_param_2['nthread'] = 4
xgb_param_2['num_class'] = 3
xgb_param_2['eval_metric']= 'mlogloss'
xgb_param_2['num_rounds'] = 300
xgb_param_2['learning_rate'] = 0.3
xgb_param_2['n_estimators'] = 100

xgb_param_3 = {}
xgb_param_3['objective'] = 'multi:softprob'
xgb_param_3['booster'] = 'gbtree'
xgb_param_3['eta'] = 0.8 # high eta values give better perf
xgb_param_3['max_depth'] = 10
xgb_param_3['silent'] = 1
xgb_param_3['lambda'] = 0.5
xgb_param_3['alpha'] = 0.5
xgb_param_3['nthread'] = 4
xgb_param_3['num_class'] = 3
xgb_param_3['eval_metric']= 'mlogloss'
xgb_param_3['num_rounds'] = 300
xgb_param_3['learning_rate'] = 0.01
xgb_param_3['n_estimators'] = 500

svm_params_1 = {}
svm_params_1['kernel'] = 'rbf'

svm_params_2 = {}
svm_params_2['kernel'] = 'wexp'
svm_params_2['gamma'] = 0.0

rf_params = {}
rf_params['max_depth'] = 10

models_available = {}

#models_available['svm_1'] = svm_params_1

#models_available['sdae_1'] = dl_params_1
#models_available['sdae_2'] = dl_params_2
#models_available['sdae_3'] = dl_params_3

<<<<<<< HEAD
#models_available['xgb_1'] = xgb_param_1
models_available['xgb_2'] = xgb_param_2
#models_available['xgb_3'] = xgb_param_3

=======
models_available['xgb_1'] = xgb_param_1
#models_available['xgb_2'] = xgb_param_2
#models_available['xgb_3'] = xgb_param_3
#models_available['rf_1'] = rf_params
>>>>>>> 503d1ff916a0bd203e0b35b32613f4a3fd4fcf84


M = 500
alpha_M = [] # m long list
alpha_M_v2 = []
acc_per_class_M = []
models_M = []
model_names_M = []

num_classes = 3

def calc_loss(w, p_y, a_y):

    loss = np.sum(np.multiply(w, [1 if p_y[w_i] != a_y[w_i] else 0 for w_i in range(len(w))]))
    return loss

def test_model(alpha_m, model, ts_ids, test_x):
    #norm_alpha_m is a 3 element array
    ids,probs = model.get_test_results((ts_ids,test_x))

    w_probs = []
    for i,p in enumerate(probs):
        p_class = np.argmax(p)

        if alpha_m is np.float:
            p[p_class] = np.asarray(p) * alpha_m
        else:
            p = np.asarray(p) * alpha_m[p_class]
        w_probs.append(p.tolist())

    #return (np.asarray(probs) * np.asarray(alpha_m)).tolist()
    return w_probs

def test_model_exact(alpha_m, model, ts_ids, test_x):
    #norm_alpha_m is a 3 element array
    ids,probs = model.get_test_results((ts_ids,test_x))

    w_act = []
    for i,p in enumerate(probs):
        tmp = [0,0,0]
        p_class = np.argmax(p)
        tmp[p_class] = alpha_m
        w_act.append(tmp)

    return w_act

# Using alpha #
min_log_loss = np.inf
best_M = 0
good_m_vals = []
type = 'probs' # 'actual' or 'prob'

for m in range(M):

    best_k,best_model,min_loss = None,None,np.inf
    best_ids,best_pred,best_act = [],[],[]

    model_losses = []
    for k,v in models_available.items():
        if 'sdae' in k:
            model = UseSDAE(models_available[k])
            train = model.train
            get_labels = model.get_labels
        elif 'xgb' in k:
            model = UseXGBoost(models_available[k])
            train = model.train
            get_labels = model.get_labels
        elif 'svm' in k:
            model = UseSVM(models_available[k])
            train= model.train
            get_labels = model.get_labels
        elif 'rf' in k:
            model = UseRF(models_available[k])
            train= model.train
            get_labels = model.get_labels

        if m == 0:
            w = [1./len(tr_y) for _ in range(len(tr_y))]

        train((tr_ids,tr_x,tr_y),(v_ids,v_x,v_y),w)
        # shoud return a tuple (ids, pred, actual)
        ids, pred_y, act_y = get_labels()
        loss_i = calc_loss(w,pred_y,act_y)
        model_losses.append((k,loss_i))
        if loss_i < min_loss:
            min_loss = loss_i
            best_k = k
            best_model = model
            best_ids = ids
            best_pred = pred_y
            best_act = act_y


    print('Model losses: ',model_losses)
    print('Model: ',best_model.__class__.__name__)
    print('Selected best model: ', best_k)
    models_M.append(best_model)
    model_names_M.append(best_k)

    vec = np.multiply(w,[1 if best_pred[i]!=best_act[i] else 0 for i in range(len(best_pred))])

    vecs_v2 = []
    weight_means = []
    for i in range(num_classes):
        temp = [(j, 1) if best_pred[j] != best_act[j] and best_act[j] == i else (j, 0) for j in range(len(best_pred))]
        print('Temp: ',len(temp))
        vec_v2 =[]
        tmp_weights = []
        for j,val in temp:
            if val != 0:
                vec_v2.append(w[j]*val)

        for j in range(len(best_act)):
            if best_act[j]==i:
                tmp_weights.append(w[j])

        weight_means.append((len(tmp_weights),np.mean(tmp_weights)))
        vecs_v2.append(vec_v2)

    print('Weigts: ',weight_means)

    err_m = np.sum(vec)/np.sum(w)

    err_m_v2 = []
    for tmp_idx in range(len(vecs_v2)):
        err_m_v2.append(np.sum(vecs_v2[tmp_idx])/(weight_means[tmp_idx][0]*weight_means[tmp_idx][1]))

    if err_m<=0 or err_m >= (1-(1/num_classes)):
        print('Best err: ',err_m,' reached 0 or went too high')
        M=m
        break

    print('Err for the (m=',m,'): ',err_m)
    err_m = np.max([np.min([err_m,1-1e-15]),1e-15])
    alpha_M.append(np.log((1-err_m)/err_m) + np.log(num_classes-1))

    err_m_v2 = [np.max([np.min([e,1-1e-10]),1e-15]) for e in err_m_v2]
    print('Err_m : ',err_m)
    print('Err_m_v2: ',err_m_v2)
    tmp_alpha = np.log(np.divide(np.asarray(1-np.asarray(err_m_v2)),np.asarray(err_m_v2))) + np.log(num_classes-1)

    if np.min(tmp_alpha)<0:
        tmp_alpha = -np.min(tmp_alpha) + tmp_alpha
    alpha_M_v2.append(tmp_alpha*alpha_M[-1]/np.sum(tmp_alpha))

    print('Alpha for the ',m,' th model: ',alpha_M[m])
    print('Alpha v2 for the ',m,' th model: ',alpha_M_v2[m])

    exp_term = np.exp([1*alpha_M[m] if best_pred[i]!=best_act[i] else 0 for i in range(len(best_pred))])
    w = np.multiply(w,exp_term)

    # renormalize
    w = np.asarray(w)*1.0/np.sum(w)

    print('Calculating multiclass log loss')

    norm_alpha = np.asarray(alpha_M)
    #norm_alpha[1:] = norm_alpha[1:]/np.sum(norm_alpha)
    tmp_good_m = []
    tmp_good_m.extend(good_m_vals)
    tmp_good_m.append(m)
    for m_2 in tmp_good_m:

        if type == 'actual':
            valid_probs_m = test_model_exact(list(norm_alpha[m_2]),models_M[m_2],v_ids,v_x)
        elif type == 'probs':
            valid_probs_m = test_model(alpha_M_v2[m_2],models_M[m_2],v_ids,v_x)

        if m_2 == 0:
            valid_probs = np.asarray(valid_probs_m)
        else:
            valid_probs += np.asarray(valid_probs_m)

    logloss = 0.0
    for v_i,id in enumerate(v_ids):
        tmp_y = [0.,0.,0.]
        tmp_y[v_y[v_i]]=1.
        if type=='actual':
            valid_exacts = np.asarray(valid_probs[v_i],dtype=np.float32)
            norm_v_probs = [0,0,0]
            norm_v_probs[np.argmax(valid_exacts)] = 1.
        elif type == 'probs':
            norm_v_probs = np.asarray(valid_probs[v_i],dtype=np.float32)*1.0/np.sum(valid_probs[v_i])
        if any(norm_v_probs)==1.:
            norm_v_probs = np.asarray([np.max([np.min(p,1-1e-15),1e-15]) for p in norm_v_probs])
        logloss += np.sum(np.asarray(tmp_y)*np.log(np.asarray(norm_v_probs)))

    logloss = -logloss/len(v_ids)
    print('Multi class log loss (valid) (alpha) (m=',m_2,'): ',logloss)
    if logloss < min_log_loss:
        min_log_loss = logloss
        best_M = m_2
        good_m_vals.append(best_M)
    #elif logloss*1.01 > min_log_loss:
    #    break

    print('Multi class log loss (valid) (alpha) (m=',best_M,'): ',min_log_loss)

for m in range(best_M+1):

    print('Testing phase for M=',m,'\n')
    weigh_probs_m = test_model(alpha_M[m],models_M[m],ts_ids,test_x)
    print('Model: ',models_M[m].__class__.__name__,' (',model_names_M[m],')')
    print('Alpha: ',alpha_M_v2[m])
    if m == 0:
        weigh_test_probs = weigh_probs_m
    else:
        weigh_test_probs += weigh_probs_m

print('\n Saving out probabilities (test)')
import csv
with open('adaboost_output.csv', 'w',newline='') as f:
    class_dist = [0,0,0]
    writer = csv.writer(f)
    for id in correct_ts_ids:
        c_id = ts_ids.index(id)
        prob = weigh_test_probs[int(c_id)]
        row = [id,prob[0]/np.sum(prob), prob[1]/np.sum(prob), prob[2]/np.sum(prob)]
        writer.writerow(row)

