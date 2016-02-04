from models_deeplearn import SDAE
from theano import function, config, shared
import theano.tensor as T
import numpy as np

class UseSDAE(object):

    def __init__(self,param):

        self.batch_size = param['batch_size']
        self.in_size = param['in_size']
        self.out_size = param['out_size']
        self.hid_sizes = param['hid_sizes']
        self.learning_rate = param['learning_rate']
        self.pre_epochs = param['pre_epochs']
        self.finetune_epochs = param['fine_epochs']
        self.lam = param['lam']
        self.act = param['act']

        self.sdae = SDAE(self.in_size,self.out_size,self.hid_sizes,self.batch_size,self.learning_rate,self.lam,self.act)
        self.sdae.process()

        self.theano_tr_ids, self.tr_pred, self.tr_act = [],[],[]

    def train(self,tr_all,v_all):

        tr_ids,tr_x,tr_y = tr_all
        v_ids,v_x,v_y = v_all

        def get_shared_data(data_xy):
            data_x,data_y = data_xy
            shared_x = shared(value=np.asarray(data_x,dtype=config.floatX),borrow=True)
            shared_y = shared(value=np.asarray(data_y,dtype=config.floatX),borrow=True)

            return shared_x,T.cast(shared_y,'int32')

        train = get_shared_data((tr_x,tr_y))
        valid = get_shared_data((v_x,v_y))

        n_train_batches = int(train[0].get_value(borrow=True).shape[0] / self.batch_size)
        n_valid_batches = int(valid[0].get_value(borrow=True).shape[0] / self.batch_size)


        pretrain_func = self.sdae.pre_train(train[0],train[1])
        finetune_func = self.sdae.fine_tune(train[0],train[1])


        my_valid_id_tensor = shared(value=np.asarray(v_ids,dtype=config.floatX),borrow=True)
        my_valid_id_int_tensor = T.cast(my_valid_id_tensor,'int32')
        validate_func = self.sdae.validate(valid[0],valid[1],my_valid_id_int_tensor)

        my_train_id_tensor = shared(value=np.asarray(tr_ids,dtype=config.floatX),borrow=True)
        my_train_id_int_tensor = T.cast(my_train_id_tensor,'int32')
        tr_validate_func = self.sdae.validate(train[0],train[1],my_train_id_int_tensor)

        for epoch in range(self.pre_epochs):
            pre_train_cost = []
            for b in range(n_train_batches):
                pre_train_cost.append(pretrain_func(b))
            print('Pretrain cost ','(epoch ', epoch,'): ',np.mean(pre_train_cost))

        min_valid_err = np.inf
        for epoch in range(self.finetune_epochs):
            from random import shuffle
            finetune_cost = []

            b_idx =[i for i in range(0,n_train_batches)]
            shuffle(b_idx)
            for b in b_idx:
                finetune_cost.append(finetune_func(b))
            print('Finetune cost: ','(epoch ', epoch,'): ',np.mean(finetune_cost))

            if epoch%25==0:
                valid_cost = []
                for b in range(n_valid_batches):
                    ids,errs,pred_y,act_y = validate_func(b)
                    valid_cost.append(errs)


                curr_valid_err = np.mean(valid_cost)
                print('Validation error: ',np.mean(valid_cost))
                if curr_valid_err*0.95>min_valid_err:
                    break
                elif  curr_valid_err<min_valid_err:
                    min_valid_err = curr_valid_err


        for b in range(n_train_batches):
            t_ids,t_errs,t_pred_y,t_act_y = tr_validate_func(b)
            self.theano_tr_ids.extend(t_ids)
            self.tr_pred.extend([np.argmax(arr) for arr in t_pred_y])
            self.tr_act.extend([np.argmax(arr) for arr in t_act_y])

    def test(self,ts_all):
        ts_ids,ts_x = ts_all
        test_x = shared(value=np.asarray(ts_x,dtype=config.floatX),borrow=True)

        test_func = self.sdae.test(test_x)
        n_test_batches = int(test_x.get_value(borrow=True).shape[0])

    def get_labels(self):
        return self.theano_tr_ids,self.tr_pred,self.tr_act

from models_xgboost import XGBoost
class UseXGBoost(object):

    def __init__(self,params):
        self.xgboost = XGBoost(params)
        self.tr_ids,self.tr_pred,self.tr_act = [],[],[]

    def train(self,tr_all,v_all):
        ids, pred, act = self.xgboost.train(tr_all,v_all)
        self.tr_ids = ids
        self.tr_pred = pred
        self.tr_act = act

    def get_labels(self):
        return self.tr_ids,self.tr_pred,self.tr_act
