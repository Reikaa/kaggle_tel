from theano import function, config, shared
import numpy as np
import theano.tensor as T

def load_teslstra_data(train_file,test_file,remove_header=False,start_col=1):
    import csv
    train_set = []
    valid_set = []
    test_set = []
    my_test_ids = []
    correct_order_test_ids = []
    with open(train_file, 'r',newline='') as f:
        reader = csv.reader(f)
        data_x = []
        data_y = []
        valid_x = []
        valid_y = []

        valid_idx = np.random.randint(0,7000,size=(100,)).tolist()
        for i,row in enumerate(reader):
            if remove_header and i==0:
                continue
            if not i in valid_idx:
                # first 2 columns are ID and location
                data_x.append(row[start_col:-1])
                data_y.append(row[-1])
            else:
                # first 2 columns are ID and location
                valid_x.append(row[start_col:-1])
                valid_y.append(row[-1])

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

    def get_shared_data(data_xy):
        data_x,data_y = data_xy
        shared_x = shared(value=np.asarray(data_x,dtype=config.floatX),borrow=True)
        shared_y = shared(value=np.asarray(data_y,dtype=config.floatX),borrow=True)

        return shared_x,T.cast(shared_y,'int32')


    train_x,train_y = get_shared_data(train_set)
    valid_x,valid_y = get_shared_data(valid_set)
    test_x = shared(value=np.asarray(test_set[0],dtype=config.floatX),borrow=True)

    print('Train: ',len(train_set[0]),' x ',len(train_set[0][0]))
    print('Valid: ',len(valid_set[0]),' x ',len(valid_set[0][0]))
    print('Test: ',len(test_set[0]),' x ',len(test_set[0][0]))

    all_data = [(train_x,train_y),(valid_x,valid_y),(test_x),my_test_ids,correct_order_test_ids]

    return all_data

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

    def get_shared_data(data_xy):
        data_x,data_y = data_xy
        shared_x = shared(value=np.asarray(data_x,dtype=config.floatX),borrow=True)
        shared_y = shared(value=np.asarray(data_y,dtype=config.floatX),borrow=True)

        return shared_x,T.cast(shared_y,'int32')


    train_x,train_y = get_shared_data(train_set)
    valid_x,valid_y = get_shared_data(valid_set)
    test_x = shared(value=np.asarray(test_set[0],dtype=config.floatX),borrow=True)

    print('Train: ',len(train_set[0]),' x ',len(train_set[0][0]))
    print('Valid: ',len(valid_set[0]),' x ',len(valid_set[0][0]))
    print('Test: ',len(test_set[0]),' x ',len(test_set[0][0]))

    all_data = [(train_x,train_y),(valid_x,valid_y),(test_x),my_test_ids,correct_order_test_ids,my_train_ids,my_valid_ids]

    return all_data

def relu(x):
    return T.switch(x>=0, x, 0.)

def softplus(x):
    return T.log(1+T.exp(x))

class Layer(object):

    def __init__(self,in_size,out_size,activation='sigmoid'):

        rng = np.random.RandomState(0)
        init = 4 * np.sqrt(6.0 / (in_size + out_size))
        initial = np.asarray(rng.uniform(low=-init, high=init, size=(in_size, out_size)), dtype=config.floatX)
        self.act = activation
        self.W = shared(value=initial,name='W_'+str(in_size)+'->'+str(out_size))
        self.b = shared(np.ones(out_size,dtype=config.floatX)*0.01,name='b_'+str(out_size))
        self.b_prime = shared(np.ones(in_size,dtype=config.floatX)*0.01,name='b_prime_'+str(out_size))
        self.out = None
        self.in_hat = None
        self.params = [self.W, self.b, self.b_prime]

    def encode(self,x,train_type='fintune'):
        if self.act == 'relu' and train_type=='pretrain':
            self.out = softplus(T.dot(x,self.W)+self.b)
        else:
            self.out = T.nnet.sigmoid(T.dot(x,self.W)+self.b)

        return self.out

    def decode(self,out,train_type='fintune'):
        if self.act == 'relu' and train_type=='pretrain':
            self.in_hat = softplus(T.dot(out,self.W.T)+self.b_prime)
        else:
            self.in_hat = T.nnet.sigmoid(T.dot(out,self.W.T)+self.b_prime)

        return self.in_hat

class SoftMax(object):

    def __init__(self,in_size,out_size,act):

        rng = np.random.RandomState(0)
        init = 4 * np.sqrt(6.0 / (in_size + out_size))
        initial = np.asarray(rng.uniform(low=-init, high=init, size=(in_size, out_size)), dtype=config.floatX)
        self.W = shared(value=initial,name='W_'+str(in_size)+'->'+str(out_size))
        self.b = shared(np.ones(out_size,dtype=config.floatX)*0.01,name='b_'+str(out_size))

        self.p_y_given_x = None
        self.y_pred = None
        self.error = None
        self.logloss = None
        self.neg_log = None
        self.params = [self.W, self.b]
        self.cost = None
        self.y_mat_update = None
        self.act = act

    def process(self,sym_chained_x,sym_y):
        self.p_y_given_x = T.nnet.softmax(T.dot(sym_chained_x, self.W) + self.b)
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        self.error = T.mean(T.neq(self.y_pred, sym_y))

        cost_vector = -T.log(self.p_y_given_x)[T.arange(sym_y.shape[0]), sym_y]
        self.cost = T.mean(cost_vector)

        if self.act == 'sigmoid':
            y_mat = T.zeros((sym_y.shape[0],3),dtype=config.floatX)
        elif self.act == 'relu':
            y_mat = T.ones((sym_y.shape[0],3),dtype=config.floatX)

        self.y_mat_update = T.set_subtensor(y_mat[T.arange(sym_y.shape[0]),sym_y], 1)

        self.logloss = T.mean(T.nnet.categorical_crossentropy(self.p_y_given_x,self.y_mat_update))
        self.neg_log = -T.mean(T.log(self.p_y_given_x)[T.arange(sym_y.shape[0]), sym_y])

class LogisticRegression(object):

    def __init__(self,in_size,batch_size):
        self.batch_size = batch_size
        self.in_size = in_size
        self.out_size = 3
        self.learn_rate = 0.1
        self.sym_x = T.dmatrix('x')
        self.sym_y = T.ivector('y')

        rng = np.random.RandomState(0)
        init = 4 * np.sqrt(6.0 / (self.in_size + self.out_size))
        initial = np.asarray(rng.uniform(low=-init, high=init, size=(self.in_size, self.out_size)), dtype=config.floatX)
        self.W = shared(value=initial,name='W_'+str(self.in_size)+'->'+str(self.out_size))
        self.b = shared(np.ones(self.out_size,dtype=config.floatX)*0.01,name='b_'+str(self.out_size))

        self.p_y_given_x = None
        self.y_pred = None
        self.error = None
        self.cost = None
        self.out = None

    def process(self):

        self.p_y_given_x = T.nnet.softmax(T.dot(self.sym_x, self.W) + self.b)
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        self.error = T.mean(T.neq(self.y_pred, self.sym_y))
        self.cost = -T.mean(T.log(self.p_y_given_x)[T.arange(self.sym_y.shape[0]), self.sym_y])

    def train(self,x,y):
        idx = T.iscalar('idx')
        params = [self.W,self.b]
        updates = [(param,param-self.learn_rate*grad) for param,grad in zip(params,T.grad(self.cost,wrt=params))]

        theano_train_fn = function(inputs=[idx],outputs=self.error,updates=updates,
                                   givens = {
                                       self.sym_x: x[idx * self.batch_size:(idx+1) * self.batch_size],
                                       self.sym_y: y[idx * self.batch_size:(idx+1) * self.batch_size]
                                   })

        def train_fn(batch_id):
            return theano_train_fn(batch_id)

        return train_fn

    def validate(self,x,y):
        idx = T.iscalar('idx')
        theano_validate_fn = function(inputs=[idx],outputs=self.error,updates=None,
                               givens={self.sym_x: x[idx * self.batch_size:(idx+1) * self.batch_size],
                                    self.sym_y: y[idx * self.batch_size:(idx+1) * self.batch_size]})

        def validate_fn(batch_id):
            return theano_validate_fn(batch_id)

        return validate_fn

    def test(self,x):
        idx = T.iscalar('idx')
        theano_test_fn = function(inputs=[idx],outputs=self.y_pred,updates=None,
                               givens={self.sym_x: x[idx :(idx+1)]})

        def test_fn(batch_id):
            return theano_test_fn(batch_id)

        return test_fn

class SDAE(object):

    def __init__(self,in_size,out_size,hid_sizes,batch_size,learning_rate,lam,act,iterations):
        self.in_size = in_size
        self.out_size = out_size
        self.layer_sizes = hid_sizes
        self.denoising = False
        self.corruption_levels = [0.05,0.05,0.05,0.05]
        self.layers = []
        self.sym_x = T.dmatrix('x')
        self.sym_y = T.ivector('y')
        self.learn_rate = learning_rate
        self.lam = lam
        self.act = act
        self.batch_size = batch_size
        self.iterations = iterations
        self.pre_costs = []
        self.fine_tune_cost = None
        self.p_y_given_x = None
        self.y_pred = None
        self.softmax = SoftMax(self.layer_sizes[-1],self.out_size,self.act)
        self.disc_cost = None
        self.neg_log = None
        self.rng = T.shared_randomstreams.RandomStreams(0)

    def process(self):

        for i in range(len(self.layer_sizes)):
            if i==0:
                self.layers.append(Layer(self.in_size,self.layer_sizes[0],self.act))
            else:
                self.layers.append(Layer(self.layer_sizes[i-1],self.layer_sizes[i],self.act))

        # pre training
        for i,layer in enumerate(self.layers):
            layer_x = self.chained_out(self.layers,self.sym_x,i,'pretrain')
            layer_out = layer.encode(layer_x,'pretrain')
            layer_x_hat = layer.decode(layer_out,'pretrain')
            if self.act == 'sigmoid':
                self.pre_costs.append(
                        T.mean(T.nnet.binary_crossentropy(layer_x_hat,layer_x))
                        + self.lam*T.sum(T.sum(layer.W**2, axis=1),axis=0))
            elif self.act == 'relu':
                self.pre_costs.append(
                        T.mean(T.sqrt(T.sum((layer_x - layer_x_hat)**2,axis=0)))
                )

        soft_in = self.chained_out(self.layers,self.sym_x,len(self.layers),'finetune')
        self.softmax.process(soft_in,self.sym_y)

        # fine-tuning
        gen_out = self.sym_x
        for i,layer in enumerate(self.layers):
            gen_out = layer.encode(gen_out,'finetune')

        gen_out_hat = gen_out
        for layer in reversed(self.layers):
            gen_out_hat = layer.decode(gen_out_hat,'finetune')

        #self.disc_cost = self.softmax.cost + (self.lam * T.mean(T.nnet.binary_crossentropy(gen_out_hat,self.sym_x)))
        weight_sums = 0.0
        for i in range(len(self.layer_sizes)):
            weight_sums += T.mean(T.mean(self.layers[i].W**2, axis=1),axis=0) \
                           + T.mean(self.layers[i].b**2) + T.mean(self.layers[i].b_prime**2)
        self.disc_cost = self.softmax.logloss + (self.lam * weight_sums)
        self.neg_log = self.softmax.neg_log  + (self.lam * weight_sums)

    #I is the index of the layer you want the out put of (index)
    def chained_out(self,layers,x,I,train_type):
        out = x
        for i in range(I):
            out = layers[i].encode(out,train_type)

        return out

    def test_output(self,x,y):
        idx = T.iscalar('idx')
        out_test_fn = function(inputs=[idx],outputs=self.layers[0].out,
                               givens={self.sym_x: x[idx * self.batch_size:(idx+1) * self.batch_size]}
                               )
        def test(batch_id):
            print(out_test_fn(batch_id))

        return test

    def test_decode(self,x,y):
        idx = T.iscalar('idx')
        out_test_fn = function(inputs=[idx],outputs=[self.layers[0].in_hat,self.layers[0].in_hat.shape],
                               givens={self.sym_x: x[idx * self.batch_size:(idx+1) * self.batch_size]}
                               )
        def test(batch_id):
            print(out_test_fn(batch_id)[0])
            print(out_test_fn(batch_id)[1])
        return test

    def test_cost(self,x,y):
        idx = T.iscalar('idx')

        cost = T.nnet.binary_crossentropy(T.nnet.sigmoid(T.dot(self.layers[0].out,self.layers[0].W.T)+self.layers[0].b_prime),self.sym_x)
        out_test_fn = function(inputs=[idx],outputs=cost,
                               givens={self.sym_x: x[idx * self.batch_size:(idx+1) * self.batch_size]}
                               )
        def test(batch_id):
            print(out_test_fn(batch_id))

        return test


    def pre_train(self,x,y):
        idx = T.iscalar('idx')
        greedy_pretrain_funcs = []
        for i,layer in enumerate(self.layers):
            print('compiling pretrain function for layer ',i)
            updates = [(param, param - self.learn_rate * grad) for param, grad in zip(layer.params, T.grad(self.pre_costs[i],wrt=layer.params))]
            greedy_pretrain_funcs.append(function(inputs=[idx],outputs=self.pre_costs[i],updates=updates,
                                               givens = {self.sym_x: x[idx * self.batch_size:(idx+1) * self.batch_size],
                                                        self.sym_y: y[idx * self.batch_size:(idx+1) * self.batch_size]
                                                        },on_unused_input='warn'))

        def train(batch_id):
            costs = []
            for i in range(len(self.layers)):
                costs.append(greedy_pretrain_funcs[i](batch_id))
            return costs

        return train

    def fine_tune(self,x,y,weights = None):
        idx = T.iscalar('idx')
        params = []

        if weights == None:
            for layer in self.layers:
                params.extend([layer.W,layer.b,layer.b_prime])
            params.extend([self.softmax.W,self.softmax.b])
            updates = [(param, param - self.learn_rate * grad) for param, grad in zip(params, T.grad(self.disc_cost,wrt=params))]
            theano_output = self.softmax.error

            theano_finetune = function(inputs=[idx],outputs=theano_output,updates=updates,
                                                   givens = {self.sym_x: x[idx * self.batch_size:(idx+1) * self.batch_size],
                                                            self.sym_y: y[idx * self.batch_size:(idx+1) * self.batch_size]
                                                            },on_unused_input='warn')
        else:
            for layer in self.layers:
                params.extend([layer.W,layer.b])
            params.extend([self.softmax.W,self.softmax.b])
            weight_batch = T.dvector('weights')

            weigh_logloss = T.sum(weight_batch*T.nnet.categorical_crossentropy(self.softmax.p_y_given_x,self.softmax.y_mat_update))

            '''
               We don't really need any of this stuff afa I understood
               This stuff comes from the mathematical explanation of the SAMME algo (it's included)
               This is just proof stuff .

            err_m = T.sum(weight_batch*T.neq(self.softmax.y_pred,self.sym_y))*1.0/T.sum(weight_batch)
            beta = ((self.out_size-1)**2/self.out_size) * (T.log((1.01-err_m)/err_m) + T.log(self.out_size -1 ))

            y_tmp = T.ones((self.sym_y.shape[0],3),dtype=config.floatX)*(-1/(1-self.out_size))
            y_symmetric = T.set_subtensor(y_tmp[T.arange(self.sym_y.shape[0]),self.sym_y], 1)

            g_tmp = T.ones((self.sym_y.shape[0],3),dtype=config.floatX)*(-1/(1-self.out_size))
            g_symmetric = T.set_subtensor(g_tmp[T.arange(self.sym_y.shape[0]),self.softmax.y_pred], 1)

            exp_loss = T.sum(weight_batch * T.exp(-(1/self.out_size)*beta*T.sum(y_symmetric*g_symmetric)))'''

            theano_output = self.softmax.error

            updates = [(param, param - self.learn_rate * grad) for param, grad in zip(params, T.grad(weigh_logloss,wrt=params))]

            theano_finetune = function(inputs=[idx],outputs=theano_output,updates=updates,
                                                   givens = {self.sym_x: x[idx * self.batch_size:(idx+1) * self.batch_size],
                                                            self.sym_y: y[idx * self.batch_size:(idx+1) * self.batch_size],
                                                             weight_batch: weights[idx * self.batch_size:(idx+1) * self.batch_size]
                                                            },on_unused_input='warn')

        def fine_tune_fn(batch_id):
            for _ in range(self.iterations):
                val = theano_finetune(batch_id)
            return val

        return fine_tune_fn

    def validate(self,x,y,ids):
        idx = T.iscalar('idx')
        entry_ids = T.ivector('entries')

        output = self.softmax.logloss
        #relu cannot handle logloss or the negative log cost
        # we used softmax.error for relu, but cost didn't change over time.
        # so we're using logloss with sigmoid for finetuning
        theano_validate_fn = function(inputs=[idx],outputs=[entry_ids,output,self.softmax.p_y_given_x,self.softmax.y_mat_update],updates=None,
                               givens={self.sym_x: x[idx * self.batch_size:(idx+1) * self.batch_size],
                                    self.sym_y: y[idx * self.batch_size:(idx+1) * self.batch_size],
                                    entry_ids: ids[idx * self.batch_size:(idx+1) * self.batch_size]})

        def validate_fn(batch_id):
            return theano_validate_fn(batch_id)

        return validate_fn

    def cross_validate(self,x,y,folds,ft_epochs,pre_epochs):
        print('\n Cross Validation ... with ', folds, ' folds \n')
        print('X: ', x.shape[0], ' Y: ',y.shape[0])

        from math import floor
        # should get all x and all y (numpy)
        x_size = x.shape[0]
        tr_size = floor(x.shape[0]*(folds-1)*1.0/folds)
        v_size = floor(x.shape[0]*1.0/folds)

        tr_batches = floor(tr_size/self.batch_size)
        v_batches = floor(v_size/self.batch_size)
        print('Train size: ',tr_size, ' Valid size: ', v_size)
        print('Train batches: ',tr_batches, ' Valid batches: ', v_batches)

        idx = T.iscalar('idx')

        min_ft_all,min_v_all = [],[]
        best_ft_epochs,best_v_epochs = [],[]
        tolerance = 10
        for i in range(folds):
            print('\n------------  Fold ',i,' ----------------\n')
            np_train_x,np_train_y = [],[]
            np_train_x.extend(x[0:i*v_size])
            np_train_x.extend(x[(i+1)*v_size:x_size])
            np_train_y.extend(y[0:i*v_size])
            np_train_y.extend(y[(i+1)*v_size:x_size])

            np_valid_x, np_valid_y = [],[]
            np_valid_x.extend(x[i*v_size:(i+1)*v_size])
            np_valid_y.extend(y[i*v_size:(i+1)*v_size])

            def get_shared_data(data_xy):
                data_x,data_y = data_xy
                shared_x = shared(value=np.asarray(data_x,dtype=config.floatX),borrow=True)
                shared_y = shared(value=np.asarray(data_y,dtype=config.floatX),borrow=True)

                return shared_x,T.cast(shared_y,'int32')

            train_x,train_y = get_shared_data((np_train_x,np_train_y))
            valid_x,valid_y = get_shared_data((np_valid_x,np_valid_y))

            min_ft_cost,min_v_cost = np.inf,np.inf
            best_ft_ep,best_v_ep = 0,0

            finetune_fn=self.fine_tune(train_x,train_y)
            validate_fn=self.validate(valid_x,valid_y)

            if i==0:
                pretrain_fn = self.pre_train(train_x,train_y)
                for pe in range(pre_epochs):
                    for b in range(n_train_batches):
                        pretrain_fn(b)

            no_increase = 0
            for e in range(ft_epochs):
                ft_cost_list = []
                for b in range(tr_batches):
                    ft_cost_list.append(finetune_fn(b))

                if np.mean(ft_cost_list) < min_ft_cost:
                    min_ft_cost = np.mean(ft_cost_list)
                    best_ft_ep = e

                v_cost_list = [] # all the v costs for all batches for given epoch and fold
                for b in range(v_batches):
                    v_cost_list.append(validate_fn(b))

                #update min value if new is minimum
                if np.mean(v_cost_list)*1.001<min_v_cost:
                    no_increase = 0
                    min_v_cost = np.mean(v_cost_list)
                    best_v_ep = e
                else:
                    no_increase += 1

                if no_increase > tolerance:
                    break

                print('For epoch ',e,' finetune cost: ',np.mean(ft_cost_list),', valid cost: ',np.mean(v_cost_list))

            min_ft_all.append(np.mean(min_ft_cost))
            min_v_all.append(np.mean(min_v_cost))
            best_ft_epochs.append(best_ft_ep)
            best_v_epochs.append(best_v_ep)

        print('\nDone Cross Validation ... \n')
        print(min_ft_all)
        print(best_ft_epochs)
        print(min_v_all)
        print(best_v_epochs)

        return np.mean(min_ft_all),np.mean(best_ft_epochs),np.mean(min_v_all),np.mean(best_v_epochs)

    def get_features(self,x,y,ids,layer_idx,isTest=False):
        idx = T.iscalar('idx')
        b_size = self.batch_size
        input_ids = T.iscalar('input_ids')

        if isTest:
            b_size = 1

        if not isTest:
            theano_get_features_fn = function(inputs=[idx],outputs=[self.layers[layer_idx].out,self.sym_y,input_ids],updates=None,
                               givens={self.sym_x: x[idx * b_size:(idx+1) * b_size],
                                       self.sym_y: y[idx * b_size:(idx+1) *b_size],
                                       input_ids: ids[idx * b_size:(idx+1) *b_size]})
        if isTest:
            theano_get_features_fn = function(inputs=[idx],outputs=[self.layers[layer_idx].out,self.sym_y,input_ids],updates=None,
                               givens={self.sym_x: x[idx * b_size:(idx+1) * b_size],
                                       self.sym_y: 0,
                                       input_ids: ids[idx * b_size:(idx+1) *b_size]})

        def get_features_fn(batch_id):
            return theano_get_features_fn(batch_id)

        return get_features_fn

    def test(self,x):
        idx = T.iscalar('idx')
        theano_test_fn = function(inputs=[idx],outputs=[self.softmax.y_pred,self.softmax.p_y_given_x],updates=None,
                               givens={self.sym_x: x[idx :(idx+1)]})

        def test_fn(batch_id):
            return theano_test_fn(batch_id)

        return test_fn


if __name__ == '__main__':

    remove_header = True

    save_features = True
    save_features_idx = 1

    crossValidate = False

    # seems pretraining helps to achieve a lower finetune error at the beginning
    isPretrained = True
    pre_epochs = 5
    finetune_epochs = 350

    batch_size = 10
    iterations = 5

    lam = 0.0
    learning_rate = 0.25
    # relu is only for pretraining
    act = 'sigmoid'

    train,valid,test_x,my_test_ids,correct_ids,train_ids,valid_ids = load_teslstra_data_v2('features_modified_2_train.csv',
                                                               'features_modified_2_test.csv',remove_header,1)
    in_size = 398
    out_size = 3
    hid_sizes = [500,250]

    print('--------------------------- Model Info ---------------------------')
    print('Batch size: ', batch_size)
    print('Layers: ',in_size,' x ',hid_sizes, ' x ',out_size)
    print('lam: ', lam)
    print('learning rate: ', learning_rate)
    print('------------------------------------------------------------------')
    print()


    from math import ceil
    n_train_batches = ceil(train[0].get_value(borrow=True).shape[0] / batch_size)
    n_valid_batches = ceil(valid[0].get_value(borrow=True).shape[0] / batch_size)
    n_test_batches = ceil(test_x.get_value(borrow=True).shape[0])

    test_out = []
    test_out_probs = []
    model = 'SDAE'
    if model == 'SDAE':
        sdae = SDAE(in_size,out_size,hid_sizes,batch_size,learning_rate,lam,act,iterations)
        sdae.process()
        test_func = sdae.test(test_x)

        if not crossValidate:
            pretrain_func = sdae.pre_train(train[0],train[1])
            finetune_func = sdae.fine_tune(train[0],train[1])
            finetune_valid_func = sdae.fine_tune(valid[0],valid[1])

            my_valid_id_tensor = shared(value=np.asarray(valid_ids,dtype=config.floatX),borrow=True)
            my_valid_id_int_tensor = T.cast(my_valid_id_tensor,'int32')
            validate_func = sdae.validate(valid[0],valid[1],my_valid_id_int_tensor)

            if save_features:

                my_train_id_tensor = shared(value=np.asarray(train_ids,dtype=config.floatX),borrow=True)
                my_train_id_int_tensor =  T.cast(my_train_id_tensor,'int32')

                my_valid_id_tensor = shared(value=np.asarray(valid_ids,dtype=config.floatX),borrow=True)
                my_valid_id_int_tensor =  T.cast(my_valid_id_tensor,'int32')

                my_test_id_tensor = shared(value=np.asarray(my_test_ids,dtype=config.floatX),borrow=True)
                my_test_id_int_tensor = T.cast(my_test_id_tensor,'int32')

                tr_feature_func = sdae.get_features(train[0],train[1],my_train_id_int_tensor,save_features_idx,False)
                v_features_func = sdae.get_features(valid[0],valid[1],my_valid_id_int_tensor,save_features_idx,False)

                ts_features_func = sdae.get_features(test_x,None,my_test_id_int_tensor,save_features_idx,True)


            if isPretrained:
                for epoch in range(pre_epochs):
                    pre_train_cost = []
                    for b in range(n_train_batches):
                        pre_train_cost.append(pretrain_func(b))
                    print('Pretrain cost ','(epoch ', epoch,'): ',np.mean(pre_train_cost))

            min_valid_err = np.inf
            for epoch in range(finetune_epochs):
                from random import shuffle
                finetune_cost = []

                b_idx =[i for i in range(0,n_train_batches)]
                shuffle(b_idx)
                for b in b_idx:
                    finetune_cost.append(finetune_func(b))
                print('Finetune cost: ','(epoch ', epoch,'): ',np.mean(finetune_cost))

                if epoch%10==0:
                    valid_cost = []
                    theano_v_ids = []
                    theano_v_pred_y = []
                    theano_v_act_y = []
                    for b in range(n_valid_batches):
                        ids,errs,pred_y,act_y = validate_func(b)
                        valid_cost.append(errs)
                        theano_v_ids.extend(ids)
                        theano_v_pred_y.extend(pred_y)
                        theano_v_act_y.extend(act_y)

                    curr_valid_err = np.mean(valid_cost)
                    print('Validation error: ',np.mean(valid_cost))
                    if curr_valid_err*0.95>min_valid_err:
                        break
                    elif  curr_valid_err<min_valid_err:
                        min_valid_err = curr_valid_err

        else:

            sym_y = T.ivector('y')
            get_train_y = function(inputs=[],outputs=sym_y, givens={sym_y: train[1]})
            get_valid_y = function(inputs=[],outputs=sym_y, givens={sym_y: valid[1]})

            all_x = train[0].get_value().tolist()
            all_x.extend(valid[0].get_value().tolist())

            all_y = get_train_y().tolist()
            all_y.extend(get_valid_y().tolist())

            cv_vals = sdae.cross_validate(np.asarray(all_x,dtype=config.floatX), np.asarray(all_y),10,finetune_epochs,pre_epochs)

            print('Mean(train): ',cv_vals[0],' , Mean(valid): ',cv_vals[2])
            print('Epocs(train): ', cv_vals[1],' , Epochs(valid): ',cv_vals[3])

        for b in range(n_test_batches):
            cls,probs = test_func(b)
            test_out.append(cls)
            test_out_probs.append(probs[0])

    '''print('\n Saving out probabilities (test)')
    with open('deepnet_out_probs.csv', 'w',newline='') as f:
        import csv
        class_dist = [0,0,0]
        writer = csv.writer(f)
        print('Ids (valid): ',len(theano_v_ids))
        print('Pred (valid): ',len(theano_v_pred_y))
        print('Act (valid): ',len(theano_v_act_y))
        for id in correct_ids:
            c_id = my_test_ids.index(id)
            probs = test_out_probs[int(c_id)]
            row= [id,probs[0], probs[1], probs[2]]
            class_dist[np.argmax(probs)] += 1
            writer.writerow(row)

    print('Predicted class distribution: ',class_dist)

    print('\n Saving out probabilities (valid)')
    with open('deepnet_valid_probs.csv', 'w',newline='') as f:
        import csv
        writer = csv.writer(f)

        row = ['id','pred_0','pred_1','pred_2','act_0','act_1','act_2']
        writer.writerow(row)
        for i,id in enumerate(theano_v_ids):
            row = [id]
            row.extend(theano_v_pred_y[i])
            row.extend(theano_v_act_y[i])
            writer.writerow(row)'''

    if save_features:
        all_tr_features = []
        all_tr_outputs = []
        all_ts_features = []
        for b in range(n_train_batches):
            features, y, ids = tr_feature_func(b)
            temp = [ids]

            temp = np.concatenate((np.reshape(ids,(ids.shape[0],1)),features),axis=1)

            all_tr_features.extend(temp.tolist())
            all_tr_outputs.extend(y)

        print('Size train features: ',len(all_tr_features),' x ',len(all_tr_features[0]))

        for b in range(n_valid_batches):
            features, y, ids = v_features_func(b)
            temp = [ids]
            temp = np.concatenate((np.reshape(ids,(ids.shape[0],1)),features),axis=1)

            all_tr_features.extend(temp.tolist())
            all_tr_outputs.extend(y)

        print('Size train+valid features: ',len(all_tr_features),' x ',len(all_tr_features[0]))

        for b in range(n_test_batches):
            features, y, id = ts_features_func(b)
            row = [id[0]]
            row.extend(features[0])
            all_ts_features.append(row)
        print('Size test features: ',len(all_ts_features),' x ',len(all_ts_features[0]))

        with open('deepnet_features_train_'+str(save_features_idx)+'.csv', 'w',newline='') as f:
            import csv
            writer = csv.writer(f)
            for feature,y in zip(all_tr_features,all_tr_outputs):
                #print('Feature: ',feature)
                row= []
                row.extend(feature)
                row.append(y)
                writer.writerow(row)

        with open('deepnet_features_test_'+str(save_features_idx)+'.csv', 'w',newline='') as f:
            import csv
            writer = csv.writer(f)
            for feature in all_ts_features:
                row= []
                row.extend(feature)
                writer.writerow(row)