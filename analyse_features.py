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

def analyse_feature_imp(header,tr_x,tr_y):
    clf = GradientBoostingClassifier()
    clf = clf.fit(tr_x, tr_y)
    fimp = np.asarray(clf.feature_importances_)
    # this gives feature improtance high to low
    ord_feature_idx = list(reversed(np.argsort(fimp)))

    for idx in ord_feature_idx[:25]:
        print(fimp[idx])

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

def analyse_aggregated_features_gbm(header,tr_x,tr_y,ts_x,fimp_idx,agg_type='mul',threshold=0.1,n_est=100,learning_rate=0.1,max_depth=5):

    i=0
    features = []
    for fidx in reversed(fimp_idx):
        print('Gradient Boosting for ',i)
        if i>tr_x.shape[1]//20:
            break;
        new_tr_x = None
        tmp_features = []
        for fidx_2 in range(fidx,tr_x.shape[1]):
            if agg_type is 'mul':
                new_feat = tr_x[:,fidx]*tr_x[:,fidx_2]
            if agg_type is 'add':
                new_feat = tr_x[:,fidx]+tr_x[:,fidx_2]
            if agg_type is 'sqr-mul':
                new_feat = tr_x[:,fidx]*(tr_x[:,fidx_2]**2)

            tmp_features.append((fidx,fidx_2,agg_type))
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
            print('[single] important feature: ',tmp_features[idx],', (',fimp[idx],')')
            features.append(tmp_features[idx])

    return features

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
    print('Features size: ',len(header))
    return header,tr_cat_features,ts_cat_features


def analyse_aggregated_features_pearson(header,tr_x,tr_y,ts_x,fimp_idx):

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
    return features

import collections
# here the header is taken from tr_x, so we don't have 'id' and 'out'
def get_new_data_with_agg(header_orig,tr_x,ts_x,features,delete_original=False):
    header = header_orig

    features.sort(key=lambda tup: tup[0],reverse=True)
    deleted = [] # need this coz fidx can repeat
    for v in features:
        tr_new_col = None
        ts_new_col = None
        new_header_col = None
        print(v[0],',',v[1],',',v[2])
        k = v[0]

        if v[2] is 'mul':
            new_header_col = str(k)+'_mul_'+str(v[1])
            tr_new_col = tr_x[:,k]*tr_x[:,v[1]]
            ts_new_col = ts_x[:,k]*ts_x[:,v[1]]
        elif v[2] is 'add':
            new_header_col = str(k)+'_add_'+str(v[1])
            tr_new_col = tr_x[:,k]+tr_x[:,v[1]]
            ts_new_col = ts_x[:,k]+ts_x[:,v[1]]
        elif v[2] is 'sqr-mul':
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

if __name__ == '__main__':

    fn = 'features_2'
    tr_v_all,ts_all,correct_ids,header =load_teslstra_data_v3(fn+'_train.csv',fn+'_test.csv',None)

    select_features_with_global_gbm = True
    tasks = ['analyse-agg-gbm','analyse-agg-pearson','analyse-tree-cat']
    task = tasks[0]

    if task == tasks[0]:
        n_est = 100
        lr = 0.1
        max_depth = 5
        threshold = 0.1
        fimp_idx = analyse_feature_imp(header,tr_v_all[1],tr_v_all[2])
        all_agg_features = []
        agg_features = analyse_aggregated_features_gbm(
                header,tr_v_all[1],tr_v_all[2],ts_all[1],fimp_idx,'mul',threshold,n_est,lr,max_depth)
        all_agg_features.extend(agg_features)
        agg_features = analyse_aggregated_features_gbm(
                header,tr_v_all[1],tr_v_all[2],fimp_idx,'add',threshold,n_est,lr,max_depth)
        all_agg_features.extend(agg_features)
        print('Adding ',len(all_agg_features),' features')
        new_header,new_tr_x,new_ts_x = get_new_data_with_agg(header,tr_v_all[1],ts_all[1],all_agg_features,False)

        # this is if we perform an extra global faeture selection step with another gbm
        if select_features_with_global_gbm:
            clf2 = GradientBoostingClassifier(n_estimators=n_est*2,learning_rate=lr,max_depth=max_depth)
            clf2 = clf2.fit(new_tr_x, tr_v_all[2])
            fimp2 = np.asarray(clf2.feature_importances_)

            ord_feature_idx = list(reversed(np.argsort(fimp2)))
            features_to_keep=[]
            for idx in ord_feature_idx:
                if fimp2[idx]<threshold:
                    break
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


        filename = fn + "_gbm"
        write_data(new_header,filename,(tr_v_all[0],new_tr_x,tr_v_all[2]),(ts_all[0],new_ts_x))

    if task == tasks[1]:
        agg_features = analyse_aggregated_features_pearson(
                header,tr_v_all[1],tr_v_all[2],ts_all[1],fimp_idx)
        new_header,new_tr_x,new_ts_x = get_new_data_with_agg(header,tr_v_all[1],ts_all[1],all_agg_features,False)
        filename = fn + "_pearson"
        write_data(new_header,filename,(tr_v_all[0],new_tr_x,tr_v_all[2]),(ts_all[0],new_ts_x))

    if task == tasks[2]:
        new_header,new_tr_x, new_ts_x = transform_with_gbm_to_categorical(header,tr_v_all[1],tr_v_all[2],ts_all[1])
        filename = fn + "_cat_tree"
        write_data(new_header,filename,(tr_v_all[0],new_tr_x,tr_v_all[2]),(ts_all[0],new_ts_x))



