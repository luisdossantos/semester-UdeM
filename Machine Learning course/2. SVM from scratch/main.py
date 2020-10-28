import numpy as np

class SVM:
    def __init__(self,eta, C, niter, batch_size, verbose):
        self.eta = eta; self.C = C; self.niter = niter; self.batch_size = batch_size; self.verbose = verbose

    def make_one_versus_all_labels(self, y, m):
        """
        y : numpy array of shape (n,)
        m : int (in this homework, m will be 10)
        returns : numpy array of shape (n,m)
        """
        one_versus_all_labels = np.full((len(y), m), -1)
        for i in range(len(y)):
            one_versus_all_labels[i, y[i]] = 1
        return (one_versus_all_labels)

    def compute_loss(self, x, y):
        """
        x : numpy array of shape (minibatch size, 401)
        y : numpy array of shape (minibatch size, 10)
        returns : float
        """
        loss = 0
        # print(self.w)
        # Pour chacune des 20 classes
        interm2 = 0
        for j in range(y.shape[1]):
            interm = 0
            for i in range(x.shape[1]):
                interm += np.power(abs(self.w[i, j]), 2)
            interm2 += interm
        loss += (1 / 2) * interm2

        interm4 = 0
        # Pour chaque exemple du mini-batch
        for i in range(x.shape[0]):
            interm3 = 0
            for k in range(y.shape[1]):
                # 1 chelou
                if (np.argmax(y[i, :]) == k):
                    pred = 1
                else:
                    pred = -1

                interm3 += np.power(max(0, 1 - (np.dot(self.w[:, k], x[i, :])) * pred), 2)
            interm4 += interm3
        loss += (self.C / x.shape[0]) * interm4
        return (loss)



    def compute_gradient(self, x, y):
        """
        x : numpy array of shape (minibatch size, 401)
        y : numpy array of shape (minibatch size, 10)
        returns : numpy array of shape (401, 10)
        """
        #print('ici')

        premier_terme = self.w
        current_line = np.dot(x, self.w)
        tru_false = 1*(((current_line * y))<1)
        deuxieme_terme = np.zeros(self.w.shape)

        for j in range(len(self.w[0])):

            current_loss = np.zeros((x.shape[0]))
            # Pour chaque exemple du mini-batch
            for i in range(x.shape[0]):

                if (np.argmax(y[i, :]) == j):
                    pred = 1
                else:
                    pred = -1
                current_loss[i] += max(0, 1 - (np.dot(self.w[:, j], x[i, :])) * pred)

            matrix_loss = [current_loss for k in range(len(x[0]))]
            #matrix_loss = np.transpose(matrix_loss)
            currentY = [y[:, j] for k in range(len(x[0]))]
            matrix_values = -(x) * np.transpose(currentY)
            matrix_scal = [tru_false[:, j] for k in range(len(x[0]))]
            matrix_values *= np.transpose(matrix_scal)
            matrix_values *= 2#*matrix_loss
            sum = (self.C/len(x))*np.sum(matrix_values, axis=0)
            deuxieme_terme[:, j] = sum
        answer = deuxieme_terme + premier_terme
        return answer



    # Batcher function
    def minibatch(self, iterable1, iterable2, size=1):
        l = len(iterable1)
        n = size
        for ndx in range(0, l, n):
            index2 = min(ndx + n, l)
            yield iterable1[ndx: index2], iterable2[ndx: index2]

    def infer(self, x):
        """
        x : numpy array of shape (number of examples to infer, 401)
        returns : numpy array of shape (number of examples to infer, 10)
        """
        score = np.zeros(self.m)
        classe_pred = []

        # Pour chaque exemple de x
        for i in range(x.shape[0]):
            # On parcourt les droites de separations de chaque classe
            for j in range(self.w.shape[1]):
                # Si l'exemple appartient a la classe j (le point est du bon cote de la droite)
                if (np.dot(self.w[:, j], x[i, :]) > 1):
                    # On ajoute le score : la distance du point par rapport a la droite (plus la distance est grande, plus le point a de chances d'etre dans la classe)
                    score[j] += np.dot(self.w[:, j], x[i, :])
            classe_pred.append(np.argmax(score))
        return (self.make_one_versus_all_labels(classe_pred, self.m))

    def compute_accuracy(self, y_inferred, y):
        """
        y_inferred : numpy array of shape (number of examples, 10)
        y : numpy array of shape (number of examples, 10)
        returns : float
        """
        accuracy = 0
        for i in range(y_inferred.shape[0]):
            if (np.argmax(y_inferred[i, :]) == np.argmax(y[i, :])):
                accuracy += 1
        return (accuracy / y_inferred.shape[0])

    def fit(self, x_train, y_train, x_test, y_test):
        """
        x_train : numpy array of shape (number of training examples, 401)
        y_train : numpy array of shape (number of training examples, 10)
        x_test : numpy array of shape (number of training examples, 401)
        y_test : numpy array of shape (number of training examples, 10)
        returns : float, float, float, float
        """
        self.num_features = x_train.shape[1]
        self.m = y_train.max() + 1
        y_train = self.make_one_versus_all_labels(y_train, self.m)
        y_test = self.make_one_versus_all_labels(y_test, self.m)
        self.w = np.zeros([self.num_features, self.m])
        #print(self.niter)

        for iteration in range(self.niter):
            # Train one pass through the training set

            for x, y in self.minibatch(x_train, y_train, size=self.batch_size):

                grad = self.compute_gradient(x,y)
                self.w -= self.eta * grad

            # Measure loss and accuracy on training set
            train_loss = self.compute_loss(x_train,y_train)
            y_inferred = self.infer(x_train)
            train_accuracy = self.compute_accuracy(y_inferred, y_train)

            # Measure loss and accuracy on test set
            test_loss = self.compute_loss(x_test,y_test)
            y_inferred = self.infer(x_test)
            test_accuracy = self.compute_accuracy(y_inferred, y_test)

            if self.verbose:
                print("Iteration %d:" % iteration)
                print("Train accuracy: %f" % train_accuracy)
                print("Train loss: %f" % train_loss)
                print("Test accuracy: %f" % test_accuracy)
                print("Test loss: %f" % test_loss)
                print("")

        return train_loss, train_accuracy, test_loss, test_accuracy

if __name__ == "__main__":
    # Load the data files
    print("Loading data...")
    x_train = np.load("train_features.npy")
    x_test = np.load("test_features.npy")
    y_train = np.load("train_labels.npy")
    y_test = np.load("test_labels.npy")
    #gbft = np.load("grad_before_fit_truth.npy")
    #print(gbft[0])




    print("Fitting the model...")
    svm = SVM(eta=0.001, C=30, niter=1, batch_size=5000, verbose=False)

    train_loss, train_accuracy, test_loss, test_accuracy = svm.fit(x_train, y_train, x_test, y_test)

    # # to infer after training, do the following:
    # y_inferred = svm.infer(x_test)

    ## to compute the gradient or loss before training, do the following:
    #y_train_ova = svm.make_one_versus_all_labels(y_train, 10) # one-versus-all labels
    #svm.w = np.zeros([401, 10])
    #grad = svm.compute_gradient(x_train, y_train_ova)
    #print(grad[0])
    #loss = svm.compute_loss(x_train, y_train_ova)
    #print(loss)

