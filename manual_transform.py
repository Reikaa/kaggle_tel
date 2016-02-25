__author__ = 'Thushan Ganegedara'

import numpy as np

import pandas as pd
def load_teslstra_data_manual(train_file,test_file,drop_cols=None):

    tr_data = pd.read_csv(train_file)
    test_data = pd.read_csv(test_file)

    tr_ids = tr_data[['id']]
    tr_x = tr_data.ix[:,1:-1]
    tr_y = tr_data.ix[:,-1]

    test_ids = test_data[['id']]
    test_x = test_data.ix[:,1:]
    correct_order_test_ids = []

    #selected_features = ['feat_'+str(fi) for fi in range(70,90)]
    #selected_features.extend(['feat_'+str(fi) for fi in range(190,230)])
    #selected_features.extend(['feat_'+str(fi) for fi in range(300,310)])
    #selected_features.extend(['feat_'+str(fi) for fi in range(360,370)])
    selected_features = ['feat_'+str(fi) for fi in range(0,380)]
    #selected_events = ['eve_'+str(ei) for ei in [0,2,3,5,6,13,14,19,21,26,28,29,30,34,35,42,43,44,47,49,53,54]]
    selected_events = ['eve_'+str(ei) for ei in range(0,54)]

    selected_resources = ['res_'+str(ri) for ri in range(0,10)]

    selected_locations = ['loc_'+str(li) for li in range(0,1126)]

    selected_severity = ['sev_'+str(si) for si in range(1,5)]
    new_tr_x = tr_data[['id']]
    new_ts_x = test_data[['id']]

    new_location_x = pd.DataFrame()

    header = list(tr_x.columns.values)

    for li,l in enumerate(selected_locations):
        #new_tr_x[l] = tr_x[l]
        #new_ts_x[l] = test_x[l]

        new_location_x[l] = tr_x[l]

    idx_0 = np.argwhere(tr_y==0).flatten()
    new_location_x_0 = new_location_x.as_matrix()[idx_0.tolist()]

    idx_1 = np.argwhere(tr_y==1).flatten()
    new_location_x_1 = new_location_x.as_matrix()[idx_1.tolist()]

    idx_2 = np.argwhere(tr_y==2).flatten()
    new_location_x_2 = new_location_x.as_matrix()[idx_2.tolist()]


    from sklearn import manifold, datasets
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    n_points = 1000
    n_neighbors = 50
    n_components = 2

    '''Y0 = manifold.TSNE(n_neighbors, n_components).fit_transform(new_location_x_0)
    Y1 = manifold.TSNE(n_neighbors, n_components).fit_transform(new_location_x_1)
    Y2 = manifold.TSNE(n_neighbors, n_components).fit_transform(new_location_x_2)

    fig = plt.figure(1)
    plt.suptitle("Manifold Learning Location", fontsize=14)
    plt.scatter(Y0[:, 0], Y0[:, 1], c='r')
    plt.scatter(Y1[:, 0], Y1[:, 1],c='g')
    plt.scatter(Y2[:, 0], Y2[:, 1], c='b')'''

    new_feature_x = pd.DataFrame()
    for fi,f in enumerate(selected_features):
        #new_tr_x[f] = tr_x[f]
        #new_ts_x[f] = test_x[f]

        new_feature_x[f] = tr_x[f]

    new_loc_feat_x = pd.DataFrame()
    loc_bin_size = 100
    feat_bin_size = 10
    from math import floor,ceil
    for fi,f in enumerate(selected_features):
        for li,l in enumerate(selected_locations):
            li_bin_idx = floor(li/loc_bin_size)
            loc_bins_per_feature = ceil(len(selected_locations)/loc_bin_size)
            idx = loc_bins_per_feature*floor(fi/feat_bin_size) + li_bin_idx
            #print('fi ',fi,', li',li, ', ',idx)
            f_id = 'new_feat_'+str(idx)

            if not f_id in new_loc_feat_x.columns:
                new_loc_feat_x[f_id] = tr_x[f]*tr_x[l]
                nonz_idx = new_loc_feat_x.loc[f_id]>0

                new_loc_feat_x.iloc[f_id,nonz_idx[0]] = 1
            else:
                new_loc_feat_x[f_id] += (tr_x[f]*tr_x[l])
                nonz_idx = new_loc_feat_x.loc[f_id]>0

                new_loc_feat_x.iloc[f_id,nonz_idx[0]] = 1

    print('Feature visualization')
    idx_0 = np.argwhere(tr_y==0).flatten()
    new_feature_x_0 = new_loc_feat_x.as_matrix()[idx_0.tolist()]

    idx_1 = np.argwhere(tr_y==1).flatten()
    new_feature_x_1 = new_loc_feat_x.as_matrix()[idx_1.tolist()]

    idx_2 = np.argwhere(tr_y==2).flatten()
    new_feature_x_2 = new_loc_feat_x.as_matrix()[idx_2.tolist()]

    F_Y0 = manifold.MDS(n_neighbors, n_components).fit_transform(new_feature_x_0)
    F_Y1 = manifold.MDS(n_neighbors, n_components).fit_transform(new_feature_x_1)
    F_Y2 = manifold.MDS(n_neighbors, n_components).fit_transform(new_feature_x_2)

    fig = plt.figure(2)
    plt.suptitle("Manifold Learning Features", fontsize=14)
    plt.scatter(F_Y0[:, 0], F_Y0[:, 1], c='r')
    plt.scatter(F_Y1[:, 0], F_Y1[:, 1], c='g')
    plt.scatter(F_Y2[:, 0], F_Y2[:, 1], c='b')

    '''print('Feature visualization')
    idx_0 = np.argwhere(tr_y==0).flatten()
    new_feature_x_0 = new_feature_x.as_matrix()[idx_0.tolist()]

    idx_1 = np.argwhere(tr_y==1).flatten()
    new_feature_x_1 = new_feature_x.as_matrix()[idx_1.tolist()]

    idx_2 = np.argwhere(tr_y==2).flatten()
    new_feature_x_2 = new_feature_x.as_matrix()[idx_2.tolist()]

    F_Y0 = manifold.TSNE(n_neighbors, n_components).fit_transform(new_feature_x_0)
    F_Y1 = manifold.TSNE(n_neighbors, n_components).fit_transform(new_feature_x_1)
    F_Y2 = manifold.TSNE(n_neighbors, n_components).fit_transform(new_feature_x_2)

    fig = plt.figure(2)
    plt.suptitle("Manifold Learning Features", fontsize=14)
    plt.scatter(F_Y0[:, 0], F_Y0[:, 1], c='r')
    plt.scatter(F_Y1[:, 0], F_Y1[:, 1], c='g')
    plt.scatter(F_Y2[:, 0], F_Y2[:, 1], c='b')'''


    '''for fi in range(200,210):
        for fi2 in range(0,100):
            feature_id = 'feat_' + str(fi) +"_add_" + str(fi2)
            new_tr_x[feature_id] = tr_x['feat_'+str(fi)] + tr_x['feat_'+str(fi2)]
            new_ts_x[feature_id] = test_x['feat_'+str(fi)] + test_x['feat_'+str(fi2)]'''

    '''for fi in range(200,240):
        for ei in range(30,54):
            feature_id = 'feat_eve_' + str(fi) +"_add_" + str(ei)
            new_tr_x[feature_id] = tr_x['feat_'+str(fi)] * tr_x['eve_'+str(ei)]
            new_ts_x[feature_id] = test_x['feat_'+str(fi)] * test_x['eve_'+str(ei)]'''

    '''for fi in range(200,240):
        for li in range(0,46):
            feature_id = 'feat_loc_' + str(fi) +"_add_" + str(li)
            new_tr_x[feature_id] = tr_x['feat_'+str(fi)] * tr_x['loc_'+str(li)]
            new_ts_x[feature_id] = test_x['feat_'+str(fi)] * test_x['loc_'+str(li)]'''

    '''for fi in range(200,240):
        for ri in range(0,10):
            feature_id = 'feat_loc_' + str(fi) +"_add_" + str(ri)
            new_tr_x[feature_id] = tr_x['feat_'+str(fi)] * tr_x['res_'+str(ri)]
            new_ts_x[feature_id] = test_x['feat_'+str(fi)] * test_x['res_'+str(ri)]'''

    for si,s in enumerate(selected_severity):
        new_tr_x[s] = tr_x[s]
        new_ts_x[s] = test_x[s]

    new_events_x =pd.DataFrame()
    for ei,e in enumerate(selected_events):
        new_tr_x[e] = tr_x[e]
        new_ts_x[e] = test_x[e]

        new_events_x[e] = tr_x[e]

    '''idx_0 = np.argwhere(tr_y==0).flatten()
    new_event_x_0 = new_events_x.as_matrix()[idx_0.tolist()]

    idx_1 = np.argwhere(tr_y==1).flatten()
    new_event_x_1 = new_events_x.as_matrix()[idx_1.tolist()]

    idx_2 = np.argwhere(tr_y==2).flatten()
    new_event_x_2 = new_events_x.as_matrix()[idx_2.tolist()]

    E_Y0 = manifold.TSNE(n_neighbors, n_components).fit_transform(new_event_x_0)
    E_Y1 = manifold.TSNE(n_neighbors, n_components).fit_transform(new_event_x_1)
    E_Y2 = manifold.TSNE(n_neighbors, n_components).fit_transform(new_event_x_2)

    fig = plt.figure(3)
    plt.suptitle("Manifold Learning Events", fontsize=14)
    plt.scatter(E_Y0[:, 0], E_Y0[:, 1],  c='r')
    plt.scatter(E_Y1[:, 0], E_Y1[:, 1], c='g')
    plt.scatter(E_Y2[:, 0], E_Y2[:, 1],  c='b')'''


    '''new_loc_eve_x_0 = np.empty((new_location_x_0.shape[0],len(selected_locations)*len(selected_events)))
    new_loc_eve_x_1 = np.empty((new_location_x_1.shape[0],len(selected_locations)*len(selected_events)))
    new_loc_eve_x_2 = np.empty((new_location_x_2.shape[0],len(selected_locations)*len(selected_events)))
    lei = 0
    for li,l in enumerate(selected_locations):
        for ei,e in enumerate(selected_events):
            new_loc_eve_x_0[:,lei] = new_location_x_0[:,li]*new_event_x_0[:,ei]
            new_loc_eve_x_1[:,lei] = new_location_x_1[:,li]*new_event_x_1[:,ei]
            new_loc_eve_x_2[:,lei] = new_location_x_2[:,li]*new_event_x_2[:,ei]
            lei += 1

    LE_Y0 = manifold.Isomap(n_neighbors, n_components).fit_transform(new_loc_eve_x_0)
    LE_Y1 = manifold.Isomap(n_neighbors, n_components).fit_transform(new_loc_eve_x_1)
    LE_Y2 = manifold.Isomap(n_neighbors, n_components).fit_transform(new_loc_eve_x_2)

    fig = plt.figure(4)
    plt.suptitle("Manifold Learning Location + Events", fontsize=14)
    plt.scatter(LE_Y0[:, 0], LE_Y0[:, 1],  c='r')
    plt.scatter(LE_Y1[:, 0], LE_Y1[:, 1], c='g')
    plt.scatter(LE_Y2[:, 0], LE_Y2[:, 1],  c='b')'''
    #plt.show()

    for ri,r in enumerate(selected_resources):
        new_tr_x[r] = tr_x[r]
        new_ts_x[r] = test_x[r]


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

    return (tr_ids.as_matrix(),new_loc_feat_x.as_matrix(),tr_y.as_matrix()), \
           (test_ids.as_matrix(),new_ts_x.as_matrix()), correct_order_test_ids

def sort_by_id(drop_cols=None):

    train_file = 'features_2_singles_train.csv'

    tr_data = pd.read_csv(train_file)


    tr_ids = tr_data[['id']].as_matrix().flatten()
    tr_loc = tr_data[['loc_0']].as_matrix().flatten()
    tr_sev = tr_data[['sev_0']].as_matrix().flatten()

    tr_x = tr_data.ix[:,1:-1]
    tr_y = tr_data.ix[:,-1].as_matrix()


    idxs = np.argsort(tr_ids)

    dict_by_class = {}
    dividor = 50
    import csv
    with open('ordered_by_id.csv', 'w',newline='') as f2:
        writer2 = csv.writer(f2)
        writer2.writerow(['id','loc','sev','out'])
        for i in idxs:
            if tr_y[i]==0:
                if not dict_by_class[idxs[i]%dividor]:
                    dict_by_class[idxs[i]%dividor] = [1,0,0]
                else:
                    dict_by_class[idxs[i]%dividor].append([1,0,0])
            if tr_y[i]==1:
                if not dict_by_class[idxs[i]%dividor]:
                    dict_by_class[idxs[i]%dividor] = [0,1,0]
                else:
                    dict_by_class[idxs[i]%dividor].append([0,1,0])
            if tr_y[i]==2:
                if not dict_by_class[idxs[i]%dividor]:
                    dict_by_class[idxs[i]%dividor] = [0,0,1]
                else:
                    dict_by_class[idxs[i]%dividor].append([0,0,1])
            r = [tr_ids[i],tr_loc[i],tr_sev[i],tr_y[i]]
            writer2.writerow(r)

    for j in range(dividor):
        print(np.asarray( dict_by_class[j]).mean(axis=0))


from numpy import linalg as LA
from sklearn.preprocessing import normalize
from sklearn.preprocessing import scale,MinMaxScaler

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

import xgboost as xgb
if __name__ == '__main__':

    sort_by_id()
    '''fn = 'features_2'
    tr_v_all,ts_all,correct_ids = load_teslstra_data_manual(fn+'_train.csv',fn+'_test.csv',None)


    param = {}
    param['objective'] = 'multi:softprob'
    param['booster'] = 'gbtree'
    param['eta'] = 0.08  # high eta values give better perf
    param['max_depth'] = 8
    param['silent'] = 1
    param['lambda'] = 0.9
    param['alpha'] = 0.9
    #param['nthread'] = 4
    param['subsample']=0.9
    param['colsample_bytree']=0.9
    param['num_class'] = 3
    param['eval_metric']='mlogloss'
    param['num_rounds'] = 600

    tr_x_tf,ts_x_tf = normalize_data(tr_v_all[1],ts_all[1],None,0)
    tr_y_tf = tr_v_all[2]
    print('Train (x,y): ',tr_x_tf.shape,',',tr_y_tf.shape)

    dtrain = xgb.DMatrix(tr_x_tf, label=tr_y_tf)
    history = xgb.cv(param,dtrain,param['num_rounds'],nfold=5,metrics={'mlogloss'})'''