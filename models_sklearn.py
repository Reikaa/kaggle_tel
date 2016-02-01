
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
        valid_idx = np.random.randint(0,7000,size=(500,)).tolist()
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
            data_x.append(row[:-1])
        test_set = [data_x]

    train_x,train_y = train_set
    valid_x,valid_y = valid_set
    test_x = test_set

    all_data = [(train_x,train_y),(valid_x,valid_y),(test_x)]

    return all_data

from sklearn import svm
class SVM(object):

    def __init__(self):
        self.svm = svm.SVC()
        print('Loading data ...')
        self.tr, self.v, self.test = load_teslstra_data()

    def train(self):
        tr_x,tr_y = self.tr
        v_x,v_y = self.v
        print('Fitting the model ...')
        #self.svm.fit(v_x,v_y)
        self.svm.fit(tr_x,tr_y)

        print('Predict ...')
        pred_v_x = self.svm.predict(v_x)

        print('Validation Accuracy: ',np.count_nonzero(pred_v_x==v_y)/len(v_y))

if __name__ == '__main__':

    svm = SVM()
    svm.train()