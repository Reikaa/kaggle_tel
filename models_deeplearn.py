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

def relu(x):
    return T.switch(x>=0, x, 0.)


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

    def encode(self,x):
        if self.act == 'sigmoid':
            self.out = T.nnet.sigmoid(T.dot(x,self.W)+self.b)
        elif self.act == 'relu':
            self.out = relu(T.dot(x,self.W)+self.b)
        return self.out

    def decode(self,out):
        if self.act == 'sigmoid':
            self.in_hat = T.nnet.sigmoid(T.dot(out,self.W.T)+self.b_prime)
        elif self.act == 'relu':
            self.in_hat = T.log(1+T.exp(T.dot(out,self.W.T)+self.b_prime))
        return self.in_hat

class SoftMax(object):

    def __init__(self,in_size,out_size):

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


    def process(self,sym_chained_x,sym_y):
        self.p_y_given_x = T.nnet.softmax(T.dot(sym_chained_x, self.W) + self.b)
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        self.error = T.mean(T.neq(self.y_pred, sym_y))

        cost_vector = -T.log(self.p_y_given_x)[T.arange(sym_y.shape[0]), sym_y]
        self.cost = T.mean(cost_vector)

        y_mat = T.zeros((sym_y.shape[0],3),dtype=config.floatX)
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

    def __init__(self,in_size,out_size,hid_sizes,batch_size,learning_rate,lam,act):
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
        self.pre_costs = []
        self.fine_tune_cost = None
        self.p_y_given_x = None
        self.y_pred = None
        self.softmax = SoftMax(self.layer_sizes[-1],self.out_size)
        self.disc_cost = None
        self.neg_log = None
        self.rng = T.shared_randomstreams.RandomStreams(0)

    def process(self):

        for i in range(len(self.layer_sizes)):
            if i==0:
                self.layers.append(Layer(self.in_size,self.layer_sizes[0],self.act))
            else:
                self.layers.append(Layer(self.layer_sizes[i-1],self.layer_sizes[i],self.act))


        for i,layer in enumerate(self.layers):
            layer_x = self.chained_out(self.layers,self.sym_x,i)
            layer_out = layer.encode(layer_x)
            layer_x_hat = layer.decode(layer_out)
            if self.act == 'sigmoid':
                self.pre_costs.append(
                        T.mean(T.nnet.binary_crossentropy(layer_x_hat,layer_x))
                        + self.lam*T.sum(T.sum(layer.W**2, axis=1),axis=0))
            elif self.act == 'relu':
                self.pre_costs.append(
                        T.mean(T.sqrt(T.sum((layer_x - layer_x_hat)**2,axis=0)))
                )

        soft_in = self.chained_out(self.layers,self.sym_x,len(self.layers))
        self.softmax.process(soft_in,self.sym_y)

        gen_out = self.sym_x
        for i,layer in enumerate(self.layers):
            gen_out = layer.encode(gen_out)

        gen_out_hat = gen_out
        for layer in reversed(self.layers):
            gen_out_hat = layer.decode(gen_out_hat)

        #self.disc_cost = self.softmax.cost + (self.lam * T.mean(T.nnet.binary_crossentropy(gen_out_hat,self.sym_x)))
        weight_sums = 0.0
        for i in range(len(self.layer_sizes)):
            weight_sums += T.sum(T.sum(self.layers[i].W**2, axis=1),axis=0) \
                           + T.sum(self.layers[i].b**2) + T.sum(self.layers[i].b_prime**2)
        self.disc_cost = self.softmax.logloss + (self.lam * weight_sums)
        self.neg_log = self.softmax.neg_log  + (self.lam * weight_sums)

    #I is the index of the layer you want the out put of (index)
    def chained_out(self,layers,x,I):
        out = x
        for i in range(I):
            out = layers[i].encode(out)

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
            print('compiling function for layer ',i)
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

    def fine_tune(self,x,y):
        idx = T.iscalar('idx')
        params = []
        for layer in self.layers:
            params.extend([layer.W,layer.b,layer.b_prime])
        params.extend([self.softmax.W,self.softmax.b])
        if self.act == 'sigmoid':
            updates = [(param, param - self.learn_rate * grad) for param, grad in zip(params, T.grad(self.disc_cost,wrt=params))]
        elif self.act == 'relu':
            updates = [(param, param - self.learn_rate * grad) for param, grad in zip(params, T.grad(self.neg_log,wrt=params))]
        theano_finetune = function(inputs=[idx],outputs=self.softmax.error,updates=updates,
                                               givens = {self.sym_x: x[idx * self.batch_size:(idx+1) * self.batch_size],
                                                        self.sym_y: y[idx * self.batch_size:(idx+1) * self.batch_size]
                                                        },on_unused_input='warn')

        def fine_tune_fn(batch_id):
            return theano_finetune(batch_id)

        return fine_tune_fn

    def validate(self,x,y):
        idx = T.iscalar('idx')
        theano_validate_fn = function(inputs=[idx],outputs=self.softmax.logloss,updates=None,
                               givens={self.sym_x: x[idx * self.batch_size:(idx+1) * self.batch_size],
                                    self.sym_y: y[idx * self.batch_size:(idx+1) * self.batch_size]})

        def validate_fn(batch_id):
            return theano_validate_fn(batch_id)

        return validate_fn

    def get_features(self,x,y,layer_idx):
        idx = T.iscalar('idx')
        theano_get_features_fn = function(inputs=[idx],outputs=[self.layers[layer_idx].out,self.sym_y],updates=None,
                               givens={self.sym_x: x[idx * self.batch_size:(idx+1) * self.batch_size],
                                       self.sym_y: y[idx * self.batch_size:(idx+1) * self.batch_size]})

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
    save_features = False
    pre_epochs = 5
    finetune_epochs = 500
    batch_size = 10
    in_size = 254 #168 for vectorized (less), 253 for vectorized (more), 98 for non-vec
    out_size = 3
    hid_sizes = [1500]

    lam = 0.001
    learning_rate = 0.25
    act = 'relu'

    train,valid,test_x,my_ids,correct_ids = load_teslstra_data('features_modified_train.csv',
                                                               'features_modified_test.csv',remove_header,1)

    n_train_batches = int(train[0].get_value(borrow=True).shape[0] / batch_size)
    n_valid_batches = int(valid[0].get_value(borrow=True).shape[0] / batch_size)
    n_test_batches = int(test_x.get_value(borrow=True).shape[0])

    test_out = []
    test_out_probs = []
    model = 'SDAE'
    if model == 'SDAE':
        sdae = SDAE(in_size,out_size,hid_sizes,batch_size,learning_rate,lam,act)
        sdae.process()

        #sdae.test_relu()

        pretrain_func = sdae.pre_train(train[0],train[1])
        finetune_func = sdae.fine_tune(train[0],train[1])
        finetune_valid_func = sdae.fine_tune(valid[0],valid[1])
        validate_func = sdae.validate(valid[0],valid[1])
        tr_feature_func = sdae.get_features(train[0],train[1],0)
        v_features_func = sdae.get_features(valid[0],valid[1],0)
        test_func = sdae.test(test_x)

        #test_fn = sdae.test_decode(train[0],train[1])
        #test_fn(0)

        #test_cost = sdae.test_cost(train[0],train[1])
        #test_cost(0)
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

            if epoch%25==0:
                valid_cost = []
                for b in range(n_valid_batches):
                    err = validate_func(b)
                    valid_cost.append(err)

                curr_valid_err = np.mean(valid_cost)
                print('Validation error: ',np.mean(valid_cost))
                if curr_valid_err*0.95>min_valid_err:
                    break
                elif  curr_valid_err<min_valid_err:
                    min_valid_err = curr_valid_err

        for b in range(n_test_batches):
            cls,probs = test_func(b)
            test_out.append(cls)
            test_out_probs.append(probs[0])

    elif model == 'LogisticRegression':

        logreg = LogisticRegression(in_size,batch_size)
        logreg.process()
        logreg_train_func = logreg.train(train[0],train[1])
        logreg_valid_func = logreg.validate(valid[0],valid[1])
        logreg_test_func = logreg.test(test_x)

        for epoch in range(finetune_epochs):
            finetune_cost = []
            for b in range(n_train_batches):
                finetune_cost.append(logreg_train_func(b))
            print('Finetune cost: ','(epoch ', epoch,'): ',np.mean(finetune_cost))

            if epoch%10==0:
                valid_cost = []
                for b in range(n_valid_batches):
                    valid_cost.append(logreg_valid_func(b))
                print('Validation error: ',np.mean(valid_cost))


        for b in range(n_test_batches):
            test_out = logreg_test_func(b)[0]


    with open('deepnet_out_probs.csv', 'w',newline='') as f:
        import csv
        writer = csv.writer(f)
        for id in correct_ids:
            c_id = my_ids.index(id)
            probs = test_out_probs[int(c_id)]
            row= [id,probs[0], probs[1], probs[2]]
            writer.writerow(row)

    if save_features:
        all_features = []
        all_outputs = []
        for b in range(n_train_batches):
            features, y = tr_feature_func(b)
            all_features.extend(features)
            all_outputs.extend(y)

        for b in range(n_valid_batches):
            features, y = v_features_func(b)
            all_features.extend(features)
            all_outputs.extend(y)

        print('Size of features: ',len(all_features))
        print('Size of outputs: ',len(all_outputs))


        with open('deepnet_features.csv', 'w',newline='') as f:
            import csv
            writer = csv.writer(f)
            for feature,y in zip(all_features,all_outputs):
                #print('Feature: ',feature)
                row= []
                row.extend(feature)
                row.append(y)
                writer.writerow(row)