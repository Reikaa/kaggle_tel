__author__ = 'Thushan Ganegedara'

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

    header = list(tr_x.columns.values)
    return (tr_ids.as_matrix(),tr_x.as_matrix(),tr_y.as_matrix()), \
           (test_ids.as_matrix(),test_x.as_matrix()), correct_order_test_ids,header




from sklearn.ensemble import GradientBoostingClassifier

def calc_feature_imp(header,tr_x,tr_y):
    clf = GradientBoostingClassifier()
    clf = clf.fit(tr_x, tr_y)
    fimp = np.asarray(clf.feature_importances_)
    # this gives feature improtance high to low
    ord_feature_idx = list(reversed(np.argsort(fimp)))
    print('[calc_feature_imp (gbm)] Done')
    return ord_feature_idx

def is_aggregated_feature_better(new_coeff,fidx_coeff,fidx2_coeff,threshold=None):

    thresh = threshold if threshold is not None else 0
    if new_coeff>thresh:
        if (fidx_coeff > 0 and fidx2_coeff <0) and new_coeff > fidx_coeff:
            return True
        if (fidx_coeff < 0 and fidx2_coeff > 0 ) and new_coeff > fidx2_coeff:
            return True
        if (fidx_coeff > 0 and fidx2_coeff > 0 ) and new_coeff > np.max([fidx_coeff,fidx2_coeff]):
            return True
        return False

    if new_coeff<-thresh:
        if (fidx_coeff < 0 and fidx2_coeff > 0) and new_coeff < fidx_coeff:
            return True
        if (fidx_coeff > 0 and fidx2_coeff < 0 ) and new_coeff < fidx2_coeff:
            return True
        if (fidx_coeff < 0 and fidx2_coeff < 0 ) and new_coeff < np.min([fidx_coeff,fidx2_coeff]):
            return True
        return False

    return False

def transform_with_gbm(header,tr_x,tr_y,ts_x,fimp_idx,agg_types=['mul'],threshold=0.1,n_est=100,learning_rate=0.1,max_depth=5):

    features = []
    for agg in agg_types:
        i=0
        for fimp_i,fidx in enumerate(fimp_idx):
            print('[gbm] Gradient Boosting for ',i)
            if i>tr_x.shape[1]//2:
                break
            new_tr_x = None
            tmp_features = []
            for fidx_2 in fimp_idx[fimp_i:]:
                if agg == 'mul':
                    new_feat = tr_x[:,fidx]*tr_x[:,fidx_2]
                if agg == 'add':
                    new_feat = tr_x[:,fidx]+tr_x[:,fidx_2]
                if agg == 'sqr-mul':
                    new_feat = tr_x[:,fidx]*(tr_x[:,fidx_2]**2)

                tmp_features.append((fidx,fidx_2,agg))
                if new_tr_x is None:
                    new_tr_x = np.asarray(new_feat).reshape(-1,1)
                else:
                    new_tr_x = np.append(new_tr_x,new_feat.reshape(-1,1),axis=1)

            i += 1

            clf = GradientBoostingClassifier(n_estimators=n_est,learning_rate=learning_rate,max_depth=max_depth)
            clf = clf.fit(new_tr_x, tr_y)
            fimp = np.asarray(clf.feature_importances_)

            # this gives feature improtance high to low
            ord_feature_idx = list(reversed(np.argsort(fimp)))

            for idx in ord_feature_idx:
                if fimp[idx]<threshold:
                    break
                # if above threshold add the features
                print('[gbm_single] important feature: ',tmp_features[idx],', (',fimp[idx],')')
                features.append(tmp_features[idx])

    new_header,new_tr_x,new_ts_x = get_new_data_with_features(header,tr_x,ts_x,features,False)
    return new_header,new_tr_x,new_ts_x

from sklearn.preprocessing import OneHotEncoder
def transform_with_gbm_to_categorical(header,tr_x,tr_y,ts_x,n_est=100,learning_rate=0.1,max_depth=5):

    clf = GradientBoostingClassifier(n_estimators=n_est,learning_rate=learning_rate,max_depth=max_depth)
    clf = clf.fit(tr_x, tr_y)

    ''' #Node count
    estimators = clf.estimators_
    for row in estimators:
        for e in row:
            print(e.tree_.node_count)'''
    leaf_indices = clf.apply(tr_x)
    leaf_indices = leaf_indices.reshape(leaf_indices.shape[0],-1)

    ts_leaf_indices = clf.apply(ts_x)
    ts_leaf_indices = ts_leaf_indices.reshape(ts_leaf_indices.shape[0],-1)

    enc = OneHotEncoder()
    enc.fit(np.append(leaf_indices,ts_leaf_indices,axis=0))

    tr_cat_features = enc.transform(leaf_indices).toarray()
    ts_cat_features = enc.transform(ts_leaf_indices).toarray()

    header = ['cat_'+str(i) for i in range(ts_cat_features.shape[1])]
    print('[gbm_cat] Features size: ',len(header))
    return header,tr_cat_features,ts_cat_features


def transform_with_pearson(header,tr_x,tr_y,ts_x,fimp_idx):

    threshold = 0.3
    features = []
    i=0
    for fidx in reversed(fimp_idx):
        #if i>10:
        #    break;
        fidx_coeff = get_pearson_coeff(tr_x[:,fidx],tr_y)
        for fidx_2 in range(tr_x.shape[1]):
            fidx2_coeff = get_pearson_coeff(tr_x[:,fidx_2],tr_y)
            new_mul_feat = tr_x[:,fidx]*tr_x[:,fidx_2]
            new_add_feat = tr_x[:,fidx]+tr_x[:,fidx_2]
            new_sqr_mul_feat = tr_x[:,fidx]*(tr_x[:,fidx_2]**2)
            new_mul_coeff = get_pearson_coeff(new_mul_feat,tr_y)
            new_add_coeff = get_pearson_coeff(new_add_feat,tr_y)
            new_sqr_mul_coeff = get_pearson_coeff(new_sqr_mul_feat,tr_y)

            all_good_agg_features = {}
            if is_aggregated_feature_better(new_mul_coeff,fidx_coeff,fidx2_coeff,threshold):
                print(fidx,'*',fidx_2,': ',new_mul_coeff)
                all_good_agg_features['mul'] = new_mul_coeff
            if is_aggregated_feature_better(new_add_coeff,fidx_coeff,fidx2_coeff,threshold):
                print(fidx,'+',fidx_2,': ',new_add_coeff)
                all_good_agg_features['add'] = new_add_coeff
            if is_aggregated_feature_better(new_sqr_mul_coeff,fidx_coeff,fidx2_coeff,threshold):
                print(fidx,'*',fidx_2,'**2: ',new_sqr_mul_coeff)
                all_good_agg_features['sqr-mul'] = new_sqr_mul_coeff

            max = 0
            best_op = None
            for k,v in all_good_agg_features.items():
                if np.abs(v)>max:
                    best_op = k

            if best_op is not None:
                features.append((fidx,fidx_2,best_op))
        i += 1

    new_header,new_tr_x,new_ts_x = get_new_data_with_features(header,tr_x,ts_x,features,False)
    return new_header,new_tr_x,new_ts_x

from sklearn.feature_selection import SelectKBest,chi2
def transform_with_chi(header,tr_x,tr_y,ts_x,fimp_idx,feature_count=1000):


    fidx = fimp_idx[int(len(fimp_idx)//2)]

    for fidx_2 in range(fidx,tr_x.shape[1]):
    #for fidx_2 in range(fidx,np.min([fidx+50,tr_x.shape[1]])):
        new_feats = np.append((tr_x[:,fidx]*tr_x[:,fidx_2]).reshape(-1,1),(tr_x[:,fidx]+tr_x[:,fidx_2]).reshape(-1,1),axis=1)
        ts_new_feats = np.append((ts_x[:,fidx]*ts_x[:,fidx_2]).reshape(-1,1),(ts_x[:,fidx]+ts_x[:,fidx_2]).reshape(-1,1),axis=1)

        header.extend([str(fidx)+'_mul_'+str(fidx_2),str(fidx)+'_add_'+str(fidx_2)])
        if (np.where(np.max(new_feats,axis=0)>1)[0]).size > 0:
            norm_idx = np.argwhere(np.max(new_feats,axis=0)>1)
            new_feats[:,norm_idx] = np.divide(new_feats[:,norm_idx],np.max(new_feats,axis=0)[norm_idx])
        if (np.where(np.max(ts_new_feats,axis=0)>1)[0]).size > 0:
            norm_idx = np.argwhere(np.max(ts_new_feats,axis=0)>1)
            ts_new_feats[:,norm_idx] = np.divide(ts_new_feats[:,norm_idx],np.max(ts_new_feats,axis=0)[norm_idx])

        tr_x = np.append(tr_x,new_feats,axis=1)
        ts_x = np.append(ts_x,ts_new_feats,axis=1)
        if np.isnan(tr_x).any():
            print(header[-1])
        if np.isnan(ts_x).any():
            print(header[-1])
        if np.max(tr_x[:,-1])>1.:
            print('max: ',np.max(tr_x[:,-1]))
        assert np.isnan(tr_x).all()==False
        assert np.isnan(ts_x).all()==False

    if tr_x.shape[1]>feature_count:
        ch2 = SelectKBest(chi2, k=feature_count)
        ch2.fit(tr_x, tr_y)
        tr_x = ch2.transform(tr_x)
        ts_x = ch2.transform(ts_x)
        header = [header[h_i] for h_i in ch2.get_support(indices=True)]
        new_features = [h for h in header if ('mul' in h or 'add' in h or 'sqr-mul' in h)]
        print('[chi] Added ',len(new_features),' new features ...')

    return header,tr_x,ts_x

import collections
# here the header is taken from tr_x, so we don't have 'id' and 'out'
def get_new_data_with_features(header_orig,tr_x,ts_x,features,delete_original=False):
    header = header_orig

    features.sort(key=lambda tup: tup[0],reverse=True)
    deleted = [] # need this coz fidx can repeat
    for v in features:
        tr_new_col = None
        ts_new_col = None
        new_header_col = None
        print(v[0],',',v[1],',',v[2])
        k = v[0]

        if v[2] == 'mul':
            new_header_col = str(k)+'_mul_'+str(v[1])
            tr_new_col = tr_x[:,k]*tr_x[:,v[1]]
            ts_new_col = ts_x[:,k]*ts_x[:,v[1]]
        elif v[2] == 'add':
            new_header_col = str(k)+'_add_'+str(v[1])
            tr_new_col = tr_x[:,k]+tr_x[:,v[1]]
            ts_new_col = ts_x[:,k]+ts_x[:,v[1]]
        elif v[2] == 'sqr-mul':
            new_header_col = str(k)+'_sqrmul_'+str(v[1])
            tr_new_col = tr_x[:,k]*(tr_x[:,v[1]]**2)
            ts_new_col = ts_x[:,k]*(ts_x[:,v[1]]**2)

        header.append(new_header_col)
        tr_x = np.append(tr_x,tr_new_col.reshape(-1,1),axis=1)
        ts_x = np.append(ts_x,ts_new_col.reshape(-1,1),axis=1)

        if delete_original:
            if k not in deleted:
                print('Deleting column ',k)
                del header[k]
                tr_x = np.delete(tr_x,k,axis=1)
                ts_x = np.delete(ts_x,k,axis=1)
                deleted.append(k)

    assert len(header) == tr_x.shape[1]
    assert tr_x.shape[1] == ts_x.shape[1]
    return header,tr_x,ts_x

def get_pearson_coeff(d1,d2):
    from scipy.stats.stats import pearsonr
    coeff = pearsonr(np.asarray(d1).reshape(-1,1),np.asarray(d2).reshape(-1,1))[0]
    return coeff

def write_data(head,fn,tr_all,ts_all):
    tr_ids,tr_x,tr_y = tr_all
    ts_ids,ts_x = ts_all
    header = ['id']
    header.extend(head)
    header.append('out')
    import csv
    with open(fn+"_train.csv", 'w',newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for r_i in range(tr_x.shape[0]):
            row_data = [tr_ids[r_i][0]]
            row_data.extend(tr_x[r_i,:].tolist())
            row_data.append(tr_y[r_i])
            writer.writerow(row_data)

    with open(fn+"_test.csv", 'w',newline='') as f:
        writer = csv.writer(f)
        del header[-1]
        writer.writerow(header)
        for r_i in range(ts_x.shape[0]):
            row_data = [ts_ids[r_i][0]]
            row_data.extend(ts_x[r_i,:].tolist())
            writer.writerow(row_data)

from numpy import linalg as LA
from sklearn.preprocessing import normalize
from sklearn.preprocessing import scale,MinMaxScaler
def normalize_data(tr_x,ts_x,normz=None,axis=0):
    if normz == 'scale':
        tr_x = scale(tr_x,axis=axis)
        ts_x = scale(ts_x,axis=axis)
    elif normz == 'minmax':
        minmax_scaler = MinMaxScaler()
        if axis==0:
            for c_i in range(tr_x.shape[1]):
                tr_x[:,c_i] = minmax_scaler.fit_transform(tr_x[:,c_i])
                ts_x[:,c_i] = minmax_scaler.fit_transform(ts_x[:,c_i])
        elif axis==1:
            for r_i in range(tr_x.shape[0]):
                tr_x[r_i,:] = minmax_scaler.fit_transform(tr_x[r_i,:])
                ts_x[r_i,:] = minmax_scaler.fit_transform(ts_x[r_i,:])
    elif normz == 'log':
        if axis==0:
            col_max = np.max(tr_x,axis=0)
            cols_non_norm = np.argwhere(col_max>1).tolist()
            tr_x[:,cols_non_norm] = np.log(tr_x[:,cols_non_norm]+1)
            ts_x[:,cols_non_norm] = np.log(ts_x[:,cols_non_norm]+1)

        if axis==1:
            row_max = np.max(tr_x,axis=0)
            rows_non_norm = np.argwhere(row_max>1).tolist()
            tr_x[rows_non_norm,:] = np.log(tr_x[rows_non_norm,:]+1)
            ts_x[rows_non_norm,:] = np.log(ts_x[rows_non_norm,:]+1)
    elif normz == 'sigmoid':
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

if __name__ == '__main__':

    fn = 'features_2'
    tr_v_all,ts_all,correct_ids,header =load_teslstra_data_v3(fn+'_train.csv',fn+'_test.csv',None)

    normaliz = True
    norm_type = 'minmax'

    normaliz_token = '_non_norm'
    tr_x,ts_x = tr_v_all[1],ts_all[1]
    if normaliz:
        print("Normalizing with ",norm_type)
        normaliz_token = '_norm'
        tr_x,ts_x = normalize_data(tr_v_all[1],ts_all[1],norm_type,axis=0)
    select_features_with_global_gbm = True
    tasks = ['analyse-agg-gbm','analyse-agg-pearson','analyse-tree-cat','analyse-chi-sqr']
    task = tasks[3]
    print('Task: ',task)
    if task == tasks[0]:
        n_est = 65
        lr = 0.1
        max_depth = 5
        threshold = 0.1
        fimp_idx = calc_feature_imp(header,tr_x,tr_v_all[2])
        all_agg_features = []
        new_header,new_tr_x,new_ts_x = transform_with_gbm(
                header,tr_x,tr_v_all[2],ts_x,fimp_idx,['mul','add'],threshold,n_est,lr,max_depth)
        print('Adding ',len(all_agg_features),' features')

        # this is if we perform an extra global faeture selection step with another gbm
        if select_features_with_global_gbm:
            n_est_glb,lr_glb,max_depth_glb = 150,0.1,7
            clf2 = GradientBoostingClassifier(n_estimators=n_est_glb,learning_rate=lr_glb,max_depth=max_depth_glb)
            clf2 = clf2.fit(new_tr_x, tr_v_all[2])
            fimp2 = np.asarray(clf2.feature_importances_)

            ord_feature_idx = list(reversed(np.argsort(fimp2)))
            features_to_keep=[]
            for idx in ord_feature_idx[:len(ord_feature_idx)*3//4]:
                # if above threshold add the features
                print('[agg] important feature: ',new_header[idx],', (',fimp2[idx],')')
                features_to_keep.append(idx)

            #get features chosen by above for loop
            features_to_remove = list(set([tmp_i for tmp_i in range(new_tr_x.shape[1])])-set(features_to_keep))
            features_to_remove.sort(reverse=True)

            print('Removing ',len(features_to_remove),' features ...')
            for f_r in features_to_remove:
                del new_header[f_r]
                new_tr_x = np.delete(new_tr_x,f_r,axis=1)
                new_ts_x = np.delete(new_ts_x,f_r,axis=1)


        filename = fn + "_gbm" + normaliz_token
        write_data(new_header,filename,(tr_v_all[0],new_tr_x,tr_v_all[2]),(ts_all[0],new_ts_x))

    if task == tasks[1]:
        new_header,new_tr_x,new_ts_x = transform_with_pearson(
                header,tr_x,tr_v_all[2],ts_x,fimp_idx)
        filename = fn + "_pearson" + normaliz_token
        write_data(new_header,filename,(tr_v_all[0],new_tr_x,tr_v_all[2]),(ts_all[0],new_ts_x))

    if task == tasks[2]:
        n_est,lr,max_depth = 25,0.1,5
        new_header,new_tr_x, new_ts_x = transform_with_gbm_to_categorical(header,tr_x,tr_v_all[2],ts_x,n_est,lr,max_depth)
        filename = fn + "_cat_tree" + normaliz_token
        write_data(new_header,filename,(tr_v_all[0],new_tr_x,tr_v_all[2]),(ts_all[0],new_ts_x))

    if task == tasks[3]:
        if norm_type == 'scale':
            print("WRONG NORMALIZATION TYPE FOR CHI-SQR")
        iterations = 50
        new_header,new_tr_x,new_ts_x = header,tr_x,ts_x
        for it in range(iterations):
            print('Iteration ',it)
            fimp_idx = calc_feature_imp(new_header,new_tr_x,tr_v_all[2])
            new_header,new_tr_x, new_ts_x = transform_with_chi(header,new_tr_x,tr_v_all[2],new_ts_x,fimp_idx,800)
            print('tr_x: ',new_tr_x.shape)
            print('ts_x: ',new_ts_x.shape)
        filename = fn + "_chi2" + normaliz_token
        write_data(new_header,filename,(tr_v_all[0],new_tr_x,tr_v_all[2]),(ts_all[0],new_ts_x))

