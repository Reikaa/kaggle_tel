__author__ = 'Thushan Ganegedara'

import numpy as np
from SparseAutoencoderGPU import SparseAutoencoder
from SoftmaxClassifierGPU import SoftmaxClassifier

from scipy import optimize
from scipy import misc
from numpy import linalg as LA

import os
from numpy import linalg as LA
from math import sqrt

from theano import function, config, shared, sandbox, Param
import theano.tensor as T
import time

import sys,getopt

from utils import tile_raster_images


class StackedAutoencoder(object):


    def __init__(self,in_size=1585, hidden_size = [500, 500, 250], out_size = 3, batch_size = 100, corruption_levels=[0.1, 0.1, 0.1],dropout=True,drop_rates=[0.5,0.2,0.2]):
        self.i_size = in_size
        self.h_sizes = hidden_size
        self.o_size = out_size
        self.batch_size = batch_size

        self.n_layers = len(hidden_size)
        self.sa_layers = []
        self.sa_activations_train = []
        self.sa_activations_test = []
        self.thetas = []
        self.thetas_as_blocks = []

        self.dropout = dropout
        self.drop_rates = drop_rates

        #check if there are layer_count+1 number of dropout rates (extra one for softmax)
        if dropout:
            assert self.n_layers+1 == len(self.drop_rates)

        self.corruption_levels = corruption_levels

        #check if there are layer_count number of corruption levels
        if denoising:
            assert self.n_layers == len(self.corruption_levels)

        self.cost_fn_names = ['sqr_err', 'neg_log']

        self.x = T.matrix('x')  #store the inputs
        self.y = T.ivector('y') #store the labels for the corresponding inputs

        self.fine_cost = T.dscalar('fine_cost') #fine tuning cost
        self.error = T.dscalar('test_error')    #test error value

        #print network info
        print("Network Info:")
        print("Layers: " ,self.n_layers)
        print("Layer sizes: ",        self.h_sizes)

        #intializing the network.
        #crating SparseAutoencoders and storing them in sa_layers
        #calculating hidden activations (symbolic) and storing them in sa_activations_train/test
        #there are two types of activations as the calculations are different for train and test with dropout
        for i in range(self.n_layers):

            if i==0:
                curr_input_size = self.i_size
            else:
                curr_input_size = self.h_sizes[i-1]

            #if i==0 input is the raw input
            if i==0:
                curr_input_train = self.x
                curr_input_test = self.x
            #otherwise input is the previous layer's hidden activation
            else:
                a2_train = self.sa_layers[-1].get_hidden_act(training=True)
                a2_test = self.sa_layers[-1].get_hidden_act(training=False)
                self.sa_activations_train.append(a2_train)
                self.sa_activations_test.append(a2_test)
                curr_input_train = self.sa_activations_train[-1]
                curr_input_test = self.sa_activations_test[-1]

            sa = SparseAutoencoder(n_inputs=curr_input_size, n_hidden=self.h_sizes[i],
                                   x_train=curr_input_train, x_test=curr_input_test,
                                   dropout=dropout, dropout_rate=self.drop_rates[i])
            self.sa_layers.append(sa)
            self.thetas.extend(self.sa_layers[-1].get_params())
            self.thetas_as_blocks.append(self.sa_layers[-1].get_params())

        #-1 index gives the last element
        a2_train = self.sa_layers[-1].get_hidden_act(training=True)
        a2_test = self.sa_layers[-1].get_hidden_act(training=False)
        self.sa_activations_train.append(a2_train)
        self.sa_activations_test.append(a2_test)

        self.softmax = SoftmaxClassifier(n_inputs=self.h_sizes[-1], n_outputs=self.o_size,
                                         x_train=self.sa_activations_train[-1], x_test = self.sa_activations_test[-1],
                                         y=self.y, dropout=self.dropout, dropout_rate=self.drop_rates[-1])
        self.lam_fine_tune = T.scalar('lam')
        self.fine_cost = self.softmax.get_cost(self.lam_fine_tune,cost_fn=self.cost_fn_names[1])

        self.thetas.extend(self.softmax.theta)

        #measure test performance
        self.error = self.softmax.get_error(self.y)
        self.predicted = self.softmax.pred

    def load_data(self):
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


        train_x,train_y = get_shared_data(train_set)
        valid_x,valid_y = get_shared_data(valid_set)
        test_test = np.asarray(test_set[0],dtype=config.floatX)
        test_x = shared(value=np.asarray(test_set[0],dtype=config.floatX),borrow=True)


        all_data = [(train_x,train_y),(valid_x,valid_y),(test_x)]

        return all_data

    def greedy_pre_training(self, train_x, batch_size=1, pre_lr=0.25,denoising=False):

        pre_train_fns = []
        index = T.lscalar('index')
        lam = T.scalar('lam')
        beta = T.scalar('beta')
        rho = T.scalar('rho')

        i = 0
        print("\nCompiling functions for DA layers...")
        for sa in self.sa_layers:


            cost, updates = sa.get_cost_and_updates(l_rate=pre_lr, lam=lam, beta=beta, rho=rho, cost_fn=self.cost_fn_names[1],
                                                    corruption_level=self.corruption_levels[i], denoising=denoising)

            #the givens section in this line set the self.x that we assign as input to the initial
            # curr_input value be a small batch rather than the full batch.
            # however, we don't need to set subsequent inputs to be an only a minibatch
            # because if self.x is only a portion, you're going to get the hidden activations
            # corresponding to that small batch of inputs.
            # Therefore, setting self.x to be a mini-batch is enough to make all the subsequents use
            # hidden activations corresponding to that mini batch of self.x
            sa_fn = function(inputs=[index, Param(lam, default=0.25), Param(beta, default=0.25), Param(rho, default=0.2)], outputs=cost, updates=updates, givens={
                self.x: train_x[index * batch_size: (index+1) * batch_size]
                }
            )

            pre_train_fns.append(sa_fn)
            i = i+1

        return pre_train_fns

    def fine_tuning(self, datasets, batch_size=1, fine_lr=0.2):
        (train_set_x, train_set_y) = datasets[0]
        (valid_set_x, valid_set_y) = datasets[1]
        test_set_x = datasets[2]

        n_valid_batches = int(valid_set_x.get_value(borrow=True).shape[0]/batch_size)


        index = T.lscalar('index')  # index to a [mini]batch

        gparams = T.grad(self.fine_cost, self.thetas)

        updates = [(param, param - gparam*fine_lr)
                   for param, gparam in zip(self.thetas,gparams)]

        fine_tuen_fn = function(inputs=[index, Param(self.lam_fine_tune,default=0.25)],outputs=self.fine_cost, updates=updates, givens={
            self.x: train_set_x[index * self.batch_size: (index+1) * self.batch_size],
            self.y: train_set_y[index * self.batch_size: (index+1) * self.batch_size]
        })

        validation_fn = function(inputs=[index],outputs=self.error, givens={
            self.x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            self.y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        },name='valid')

        def valid_score():
            return [validation_fn(i) for i in range(n_valid_batches)]
        return fine_tuen_fn, valid_score

    def train_model(self, datasets=None, pre_epochs=5, fine_epochs=300, pre_lr=0.25, fine_lr=0.2, batch_size=1, lam=0.0001, beta=0.25, rho = 0.2,denoising=False):

        print("Training Info...")
        print("Batch size: ",batch_size)
        print("Pre-training: ",pre_lr," (lr) ",pre_epochs," (epochs)")
        print("Fine-tuning: ",fine_lr," (lr) ",fine_epochs," (epochs)")
        print("Corruption: ",denoising,self.corruption_levels)
        print("Weight decay: ",lam)
        print("Dropout: ",self.dropout, self.drop_rates)
        print("Sparcity: (beta) ",beta," (rho)", rho)

        (train_set_x, train_set_y) = datasets[0]
        (valid_set_x, valid_set_y) = datasets[1]


        n_train_batches = int(train_set_x.get_value(borrow=True).shape[0] / batch_size)

        pre_train_fns = self.greedy_pre_training(train_set_x, batch_size=self.batch_size,pre_lr=pre_lr,denoising=denoising)

        start_time = time.clock()
        for i in range(self.n_layers):

            print("\nPretraining layer ",i)
            for epoch in range(pre_epochs):
                c=[]
                for batch_index in range(n_train_batches):
                    c.append(pre_train_fns[i](index=batch_index, lam=lam, beta=beta, rho=rho))

                print('Training epoch',epoch,', cost ',np.mean(c))

            end_time = time.clock()
            training_time = (end_time - start_time)

            print("Training time: %f" %training_time)

        #########################################################################
        #####                          Fine Tuning                          #####
        #########################################################################
        print("\nFine tuning...")

        fine_tune_fn,valid_model = self.fine_tuning(datasets,batch_size=self.batch_size,fine_lr=fine_lr)

        #########################################################################
        #####                         Early-Stopping                        #####
        #########################################################################
        patience = 10 * n_train_batches # look at this many examples
        patience_increase = 2.
        improvement_threshold = 0.995
        #validation frequency - the number of minibatches to go through before checking validation set
        validation_freq = min(n_train_batches,patience/2)

        #we want to minimize best_valid_loss, so we shoudl start with largest
        best_valid_loss = np.inf
        test_score = 0.

        done_looping = False
        epoch = 0

        while epoch < fine_epochs and (not done_looping):
            epoch = epoch + 1
            fine_tune_cost = []
            for mini_index in range(n_train_batches):
                cost = fine_tune_fn(index=mini_index,lam=lam)
                fine_tune_cost.append(cost)
                #what's the role of iter? iter acts as follows
                #in first epoch, iter for minibatch 'x' is x
                #in second epoch, iter for minibatch 'x' is n_train_batches + x
                #iter is the number of minibatches processed so far...
                iter = (epoch-1) * n_train_batches + mini_index

                # this is an operation done in cycles. 1 cycle is iter+1/validation_freq
                # doing this every epoch
                if (iter+1) % validation_freq == 0:
                    validation_losses = valid_model()
                    curr_valid_loss = np.mean(validation_losses)
                    print('epoch',epoch,' minibatch ',mini_index+1 ,' validation error is ',curr_valid_loss)

                    if curr_valid_loss < best_valid_loss:

                        if (
                            curr_valid_loss < best_valid_loss * improvement_threshold
                        ):
                            patience = max(patience, iter * patience_increase)

                        best_valid_loss = curr_valid_loss
                        best_iter = iter

            print('Fine tune cost for epoch ',epoch+1,' is ',np.mean(fine_tune_cost))
            #patience is here to check the maximum number of iterations it should check
            #before terminating
            if patience <= iter:
                done_looping = True
                break


    def test_model(self,test_set_x,batch_size= 1):
        batch_size = 1
        print('\nTesting the model...')
        n_test_batches = int(test_set_x.get_value(borrow=True).shape[0] / batch_size)

        index = T.lscalar('index')

        #no update parameters, so this just returns the values it calculate
        #without objetvie function minimization
        test_fn = function(inputs=[index], outputs=self.predicted, givens={
            self.x: test_set_x[
                index * batch_size: (index + 1) * batch_size
            ]
        }, name='test')

        pred_out = []
        for batch_index in range(n_test_batches):
            out = test_fn(batch_index)
            pred_out.append(out[0])


        print("Predicted Values \n")
        print(pred_out)

        with open('deepnet_out.csv', 'w',newline='') as f:
            import csv
            writer = csv.writer(f)
            for p in pred_out:
                row = [0,0,0]
                row[p] = 1
                writer.writerow(row)

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def lin_rec(self,x):
        return np.max([0,x])
    def cost(self,input,theta_as_blocks,layer_idx,index):

        layer_input = input
        for i in range(layer_idx):
            a = self.sigmoid(np.dot(layer_input,theta_as_blocks[i][0]) + theta_as_blocks[i][1])
            layer_input = a

        cost = self.sigmoid(np.dot(layer_input,theta_as_blocks[layer_idx][0]) + theta_as_blocks[layer_idx][1])[index]
        #print "         Cost for node %i in layer %i is %f" %(index,layer_idx,cost)
        return -cost


if __name__ == '__main__':
    #sys.argv[1:] is used to drop the first argument of argument list
    #because first argument is always the filename
    try:
        opts,args = getopt.getopt(sys.argv[1:],"h:p:f:b:d:",["w_decay=","early_stopping=","dropout=","corruption=","beta=","rho="])
    except getopt.GetoptError:
        print('<filename>.py -h [<hidden values>] -p <pre-epochs> -f <fine-tuning-epochs> -b <batch_size> -d <data_folder>')
        sys.exit(2)

    #when I run in command line
    if len(opts)!=0:
        lam = 0.0
        dropout = True
        drop_rates = [0.1,0.1,0.1]
        corr_level = [0.1, 0.2, 0.3]
        denoising = False
        beta = 0.0
        rho = 0.0

        for opt,arg in opts:
            if opt == '-h':
                hid_str = arg
                hid = [int(s.strip()) for s in hid_str.split(',')]
            elif opt == '-p':
                pre_ep = int(arg)
            elif opt == '-f':
                fine_ep = int(arg)
            elif opt == '-b':
                b_size = int(arg)
            elif opt == '-d':
                data_dir = arg
            elif opt == '--w_decay':
                lam = float(arg)
            elif opt == '--dropout':
                drop_rate_str = arg.split(',')[0]
                if drop_rate_str=='y':
                    dropout = True
                    drop_rates = [float(s.strip()) for s in arg.split(',')[1:]]
                elif drop_rate_str == 'n':
                    dropout = False
            elif opt == '--corruption':
                corr_str = arg
                denoise_str = corr_str.split(',')[0]
                if denoise_str=='y':
                    denoising = True
                    corr_level = [float(s.strip()) for s in corr_str.split(',')[1:]]
                else:
                    denoising = False
            elif opt == '--beta':
                beta = float(arg)
            elif opt == '--rho':
                rho = float(arg)

    #when I run in Pycharm
    else:
        lam = 0.0
        hid = [500,500]
        pre_ep = 20
        fine_ep = 100
        b_size = 50
        dropout = True
        drop_rates = [0.2,0.2,0.2]
        corr_level = [0.2,0.2]
        denoising=True
        beta = 0.0
        rho = 0.6
    sae = StackedAutoencoder(hidden_size=hid, batch_size=b_size, corruption_levels=corr_level,dropout=dropout,drop_rates=drop_rates)
    all_data = sae.load_data()
    sae.train_model(datasets=all_data, pre_epochs=pre_ep, fine_epochs=fine_ep, batch_size=sae.batch_size, lam=lam, beta=beta, rho=rho, denoising=denoising)
    sae.test_model(all_data[2],batch_size=sae.batch_size)
    #max_inp = sae.get_input_threshold(all_data[0][0])
    #sae.visualize_hidden(max_inp)