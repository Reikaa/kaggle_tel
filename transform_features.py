__author__ = 'Thushan Ganegedara'
import numpy as np
import matplotlib.pyplot as plt

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
                # first 2 columns are ID and location
                data_x.append(np.asarray(row[2:-1],dtype=np.float32).tolist())
                data_y.append(int(row[-1]))
            else:
                # first 2 columns are ID and location
                valid_x.append(np.asarray(row[2:-1],dtype=np.float32).tolist())
                valid_y.append(int(row[-1]))

        train_set = (data_x,data_y)
        valid_set = (valid_x,valid_y)

    with open('features_test.csv', 'r',newline='') as f:
        reader = csv.reader(f)
        data_x = []
        for i,row in enumerate(reader):
            if i==0:
                continue
            # first 2 columns are ID and location
            data_x.append(np.asarray(row[2:],dtype=np.float32))
        test_set = [data_x]

    print('Train: ',len(train_set[0]),' x ',len(train_set[0][0]))
    print('Valid: ',len(valid_set[0]),' x ',len(valid_set[0][0]))
    print('Test: ',len(test_set[0]),' x ',len(test_set[0][0]))

    all_data = [train_set,valid_set,test_set]

    return all_data


def plot_mean_against_output(train_x,train_y,valid_x,valid_y):

    all_tr_x = []
    for i in range(3):
        tr_i_indexes = [idx for idx,val in enumerate(train_y) if val==i]
        v_i_indexes = [idx for idx,val in enumerate(valid_y) if val==i]

        tr_x_rows_for_i = []
        for tr_idx in tr_i_indexes:
            tr_x_rows_for_i.append(train_x[tr_idx])

        v_x_rows_for_i = []
        for v_idx in v_i_indexes:
            v_x_rows_for_i.append(valid_x[v_idx])

        tr_x_means_for_i = np.mean(np.asarray(tr_x_rows_for_i),axis=0)
        v_x_means_for_i = np.mean(np.asarray(v_x_rows_for_i),axis=0)
        all_means_for_i = (np.add(tr_x_means_for_i,v_x_means_for_i)/2.).tolist()
        all_tr_x.append(all_means_for_i)

    x_axis = np.linspace(1,98,98)

    plot(1,1,3,[x_axis,x_axis,x_axis],all_tr_x,['0','1','2'])

def plot(fig_id,sub_row,sub_col,X,Y,titles,fontsize='large'):
    fig = plt.figure(fig_id)
    import matplotlib.gridspec as gridspec
    gs1 = gridspec.GridSpec(sub_row, sub_col)
    axes = []
    for i in range(sub_row*sub_col):
        ax = fig.add_subplot(gs1[i])

        ax.plot(X[i],Y[i])
            #ax.set_xlabel(x_label, fontsize='medium')
            #ax.set_ylabel(y_label, fontsize=fontsize)
        ax.set_title(titles[i], fontsize=fontsize)
        axes.append(ax)

    plt.show()
if __name__ == '__main__':

    all_data = load_teslstra_data()
    plot_mean_against_output(all_data[0][0],all_data[0][1],all_data[1][0],all_data[1][1])