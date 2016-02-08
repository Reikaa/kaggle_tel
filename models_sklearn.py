
import numpy as np
from sklearn import svm
from sklearn.metrics import log_loss

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

def weighted_lin_kernel(weights,X,Y):

    return np.multiply(weights,np.dot(X,Y.T))

def weighted_exp_kernel(weights,gamma,X,Y):
    euc = np.zeros((X.shape[0],Y.shape[0]),dtype=np.float32)
    for i,y in enumerate(Y):
        euc[i,:] = np.linalg.norm((X-y),ord=2,axis=1).reshape((1,-1))

    return np.exp(-gamma*np.multiply(weights,euc))


from functools import  partial

class SVM(object):

    def __init__(self,params):
        self.params = params
        self.svm = None

    def train(self,tr_all,v_all,weights=None):

        tr_ids,tr_x,tr_y = tr_all
        v_ids,v_x,v_y = v_all

        weight_means = []
        for i in range(3):
            tmp_weights = []
            for j in range(len(tr_y)):
                if tr_y[j]==i:
                    tmp_weights.append(weights[j])
            weight_means.append(np.mean(tmp_weights))

        weight_means = np.asarray(weight_means)/np.sum(weight_means)
        class_weights = {0:weight_means[0],1:weight_means[1],2:weight_means[2]}
        if self.params['kernel'] == 'wlinear':
            self.svm = svm.SVC(kernel=partial(weighted_lin_kernel,weights))
        elif self.params['kernel'] == 'wexp':
            self.svm = svm.SVC(kernel=partial(weighted_exp_kernel,weights,self.params['gamma']))
        elif self.params['kernel'] == 'rbf':
            self.svm = svm.SVC(kernel='rbf',class_weight=class_weights)

        print('Fitting the model ...')
        #self.svm.fit(v_x,v_y)
        self.svm.fit(np.asarray(tr_x,dtype=np.float32),tr_y)

        print('Predict ...')
        pred_tr = self.svm.predict(np.asarray(tr_x,dtype=np.float32))
        pred_valid = self.svm.predict(np.asarray(v_x,dtype=np.float32))

        logloss = 0.0
        for i,id in enumerate(v_ids):
            tmp_y = [0.,0.,0.]
            tmp_y[v_y[i]]=1.
            norm_v_probs = [0.,0.,0.]
            norm_v_probs[pred_valid[i]] = 1.0
            if any(norm_v_probs)==1.:
                norm_v_probs = np.asarray([np.max([np.min(p,1-1e-15),1e-15]) for p in norm_v_probs])
            logloss += np.sum(np.asarray(tmp_y)*np.log(np.asarray(norm_v_probs)))
        logloss = -logloss/len(v_ids)
        print('SVM logloss (valid): ',logloss)
        return tr_ids,pred_tr,tr_y

    def test(self,test_x):

        pred_y = self.svm.predict(np.asarray(test_x,dtype=np.float32))

        ind_pred = []
        for p in pred_y:
            tmp = [0,0,0]
            tmp[p] = 1.
            ind_pred.append(tmp)

        return ind_pred
if __name__ == '__main__':

    svm = SVM()
    svm.train()