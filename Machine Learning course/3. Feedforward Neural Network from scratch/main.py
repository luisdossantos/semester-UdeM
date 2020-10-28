import pickle
import numpy as np
import time


class NN(object):
    def __init__(self,
                 hidden_dims=(512, 256),
                 datapath='cifar10.pkl',
                 n_classes=10,
                 epsilon=1e-6,
                 lr=7e-4,
                 batch_size=1000,
                 seed=None,
                 activation="relu",
                 init_method="glorot"
                 ):

        self.hidden_dims = hidden_dims
        self.n_hidden = len(hidden_dims)
        self.datapath = datapath
        self.n_classes = n_classes
        self.lr = lr
        self.batch_size = batch_size
        self.init_method = init_method
        self.seed = seed
        self.activation_str = activation
        self.epsilon = epsilon

        self.train_logs = {'train_accuracy': [], 'validation_accuracy': [], 'train_loss': [], 'validation_loss': []}

        if datapath is not None:
            u = pickle._Unpickler(open(datapath, 'rb'))
            u.encoding = 'latin1'
            self.train, self.valid, self.test = u.load()
        else:
            self.train, self.valid, self.test = None, None, None

    def initialize_weights(self, dims):
        '''
                PARAMETRE :
                dims[0] : dimension de l'input
                dims[1] : nombre de classes
        '''

        if self.seed is not None:
            np.random.seed(self.seed)

        self.weights = {}
        # self.weights is a dictionary with keys W1, b1, W2, b2, ..., Wm, Bm where m - 1 is the number of hidden layers
        all_dims = [dims[0]] + list(self.hidden_dims) + [dims[1]]
        for layer_n in range(1, self.n_hidden + 2):  # self.n_hidden + 2 = 4 ici
            value = 1 / np.sqrt(all_dims[layer_n - 1])
            self.weights[f"W{layer_n}"] = np.random.uniform(low=-value, high=value,
                                                            size=(all_dims[layer_n - 1], all_dims[layer_n]))
            self.weights[f"b{layer_n}"] = np.zeros((1, all_dims[layer_n]))



    def relu(self, x, grad=False):
        if grad:
            return (x>0)*1
        return np.maximum(x,0)


    def sigmoid(self, x, grad=False):
        expo = 1/(1+np.exp(-x))
        if grad:
            return expo*(1-expo)
        return expo

    def tanh(self, x, grad=False):
        expoP = np.exp(x)
        expoM = np.exp(-x)
        tan = (expoP-expoM)/(expoP+expoM)
        if grad:
            return 1-tan**2
        return tan

    def activation(self, x, grad=False):
        if self.activation_str == "relu":
            return self.relu(x, grad)
        elif self.activation_str == "sigmoid":
            return self.sigmoid(x, grad)
        elif self.activation_str == "tanh":
            return self.tanh(x, grad)
        else:
            raise Exception("invalid")


    def softmax(self, x):
        # Remember that softmax(x-C) = softmax(x) when C is a constant.
        is_list = False
        if x.ndim == 1:
            is_list = True
            x = np.array([x])
        c = np.max(x, axis=1)


        for i, l in enumerate(x):
            l-= c[i]

        expo = np.exp(x)

        denum = np.sum(expo, axis=1)
        for k in range(len(denum)):
            expo[k] /= denum[k]
        if is_list:
            return expo[0]
        return expo



    def forward(self, x):

        cache = {"Z0": x}
        for i in range(self.n_hidden):
            cache[f"A{i+1}"] = np.add(np.dot(cache[f"Z{i}"], self.weights[f"W{i+1}"]),self.weights[f"b{i+1}"])
            cache[f"Z{i+1}"] = self.activation(cache[f"A{i+1}"], grad=False)

        cache[f"A{self.n_hidden+1}"] = np.add(np.dot(cache[f"Z{self.n_hidden}"], self.weights[f"W{self.n_hidden+1}"]), self.weights[f"b{self.n_hidden+1}"])
        cache[f"Z{self.n_hidden+1}"] = self.softmax(cache[f"A{self.n_hidden+1}"])

        return cache






    def backward(self, cache, labels):
        grads = {}

        dl_doa = cache[f"Z{self.n_hidden + 1}"] - labels




        '''dl_dw_batch = np.zeros(self.weights[f"W{self.n_hidden+1}"].shape)
        for tour in range(cache[f"Z{self.n_hidden}"].shape[0]):
            T1 = time.time()
            dl_dw = np.zeros(self.weights[f"W{self.n_hidden+1}"].shape)
            for col in range(dl_dw.shape[0]):
                dl_dw[col] = dl_doa[tour]
            for line in range(len(dl_doa[tour])):
                dl_dw[:, line] *= cache[f"Z{self.n_hidden}"][tour]
            dl_dw_batch += dl_dw
            T2 = time.time()
            print((T2 - T1) * 1000000)
        '''

        dl_dw_b = np.dot(np.transpose(cache[f"Z{self.n_hidden}"]), dl_doa)








        grads[f"dA{self.n_hidden + 1}"] = dl_doa
        grads[f"dW{self.n_hidden + 1}"] = (1/cache[f"Z{self.n_hidden}"].shape[0])*dl_dw_b
        grads[f"db{self.n_hidden + 1}"] = (1/dl_doa.shape[0])*np.sum(dl_doa, axis=0)

        last_doa = dl_doa



        for layer_n in range(self.n_hidden, 0, -1):

            #print('icic \n')

            T4 = time.time()


            grad_dz = np.zeros(cache[f"Z{layer_n}"].shape)
            for input_n in range(grad_dz.shape[0]):
                grad_dz[input_n] = np.dot(last_doa[input_n], np.transpose(self.weights[f"W{layer_n+1}"]))



            grads[f"dZ{layer_n}"] = grad_dz

            if np.array(grads[f"dZ{layer_n}"]).ndim == 1:
                grads[f"dZ{layer_n}"] = np.array([grads[f"dZ{layer_n}"]])



            activ = self.activation(cache[f"A{layer_n}"], grad=True)
            if np.array(activ).ndim == 1:
                activ = np.array([activ])

            current_dl_doa = activ*grad_dz

            last_doa = current_dl_doa


            if np.array(cache[f"Z{layer_n - 1}"]).ndim == 1:
                cache[f"Z{layer_n - 1}"] = np.array([cache[f"Z{layer_n - 1}"]])




            '''dl_dw_batch = np.zeros(self.weights[f"W{layer_n}"].shape)
            for tour in range(cache[f"Z{layer_n-1}"].shape[0]):
                dl_dw = np.zeros(self.weights[f"W{layer_n}"].shape)
                for col in range(dl_dw.shape[0]):
                    dl_dw[col] = last_doa[tour]

                for line in range(dl_dw.shape[1]):
                    dl_dw[:, line] *= cache[f"Z{layer_n-1}"][tour]
                dl_dw_batch += dl_dw
            '''
            dl_dw_b = np.dot(np.transpose(cache[f"Z{layer_n-1}"]), last_doa)



            grads[f"dA{layer_n}"] = last_doa
            grads[f"dW{layer_n}"] = (1/cache[f"Z{self.n_hidden}"].shape[0])*dl_dw_b
            grads[f"db{layer_n}"] = (1/last_doa.shape[0])*np.sum(last_doa, axis=0)



            if grads[f"db{layer_n}"].ndim == 1:
                grads[f"db{layer_n}"] = np.array([grads[f"db{layer_n}"]])

        return grads







    def update(self, grads):
        for layer in range(1, self.n_hidden + 2):

            self.weights[f"W{layer}"] -= self.lr*grads[f"dW{layer}"]
            self.weights[f"b{layer}"] -= self.lr * grads[f"db{layer}"]



    def one_hot(self, y):
        answer = np.zeros((len(y), self.n_classes))

        for i, element in enumerate(y):
            answer[i][element] = 1

        return answer

    def loss(self, prediction, labels):
        prediction[np.where(prediction < self.epsilon)] = self.epsilon
        prediction[np.where(prediction > 1 - self.epsilon)] = 1 - self.epsilon
        total = 0
        for i, line in enumerate(prediction):
            good_answer = np.argmax(labels[i])
            total -= np.log(line[good_answer])

        return total/(len(prediction))

    def compute_loss_and_accuracy(self, X, y):
        one_y = self.one_hot(y)
        cache = self.forward(X)
        predictions = np.argmax(cache[f"Z{self.n_hidden + 1}"], axis=1)
        accuracy = np.mean(y == predictions)
        loss = self.loss(cache[f"Z{self.n_hidden + 1}"], one_y)
        return loss, accuracy, predictions

    def train_loop(self, n_epochs):
        X_train, y_train = self.train
        y_onehot = self.one_hot(y_train)
        dims = [X_train.shape[1], y_onehot.shape[1]]
        self.initialize_weights(dims)

        n_batches = int(np.ceil(X_train.shape[0] / self.batch_size))
        print(n_batches)

        for epoch in range(n_epochs):
            print(epoch)
            t1 = time.time()
            for batch in range(n_batches):


                minibatchX = X_train[self.batch_size * batch:self.batch_size * (batch + 1), :]

                minibatchY = y_onehot[self.batch_size * batch:self.batch_size * (batch + 1), :]

                forwa = self.forward(minibatchX)

                back = self.backward(forwa, minibatchY)

                self.update(back)
            t2 = time.time()


            X_train, y_train = self.train
            train_loss, train_accuracy, _ = self.compute_loss_and_accuracy(X_train, y_train)
            X_valid, y_valid = self.valid
            valid_loss, valid_accuracy, _ = self.compute_loss_and_accuracy(X_valid, y_valid)

            self.train_logs['train_accuracy'].append(train_accuracy)
            self.train_logs['validation_accuracy'].append(valid_accuracy)
            self.train_logs['train_loss'].append(train_loss)
            self.train_logs['validation_loss'].append(valid_loss)

        return self.train_logs

    def evaluate(self):
        X_test, y_test = self.test
        test_loss, test_accuracy, _ = self.compute_loss_and_accuracy(X_test, y_test)
        return test_loss, test_accuracy



