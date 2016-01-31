
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

    def get_shared_data(data_xy):
        data_x,data_y = data_xy
        shared_x = shared(value=np.asarray(data_x,dtype=config.floatX),borrow=True)
        shared_y = shared(value=np.asarray(data_y,dtype=config.floatX),borrow=True)

        return shared_x,T.cast(shared_y,'int32')

    train_x,train_y = train_set
    valid_x,valid_y = valid_set
    test_x = test_set

    all_data = [(train_x,train_y),(valid_x,valid_y),(test_x)]

    return all_data

from sklearn import svm
class SVM(object):

    def __init__(self):
        self.svm = svm.LinearSVC()
        print('Loading data ...')
        self.tr, self.v, self.test = load_teslstra_data()

    def train(self):
        tr_x,tr_y = self.tr
        v_x,v_y = self.v
        print('Fitting the model ...')
        self.svm.fit(v_x,v_y)
        self.svm.fit(tr_x,tr_y)

        print('Predict ...')
        pred_v_x = self.svm.predict(v_x)

        print('Validation Accuracy: ',np.count_nonzero(pred_v_x==v_y)/len(v_y))

if __name__ == '__main__':

    svm = SVM()
    svm.train()