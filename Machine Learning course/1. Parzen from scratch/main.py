import numpy as np
from math import *



######## DO NOT MODIFY THIS FUNCTION ########
def draw_rand_label(x, label_list):
    seed = abs(np.sum(x))
    while seed < 1:
        seed = 10 * seed
    seed = int(1000000 * seed)
    np.random.seed(seed)
    return np.random.choice(label_list)
#############################################

       

class Q1:

    def feature_means(self, iris):
        return(np.mean(iris[:,0:4], axis=0))

    def covariance_matrix(self, iris):
        return(np.cov([iris[:,0], iris[:,1], iris[:,2], iris[:,3]]))

    def feature_means_class_1(self, iris):
        new_iris = iris[iris[:,4]==1]
        return(np.mean(new_iris[:,0:4], axis=0))

    def covariance_matrix_class_1(self, iris):
        new_iris = iris[iris[:,4]==1]
        return(np.cov([new_iris[:,0], new_iris[:,1], new_iris[:,2], new_iris[:,3]]))


def euclidean_distance(x, y):
    sum=0
    for i in range (len(x)):
        sum += (x[i]-y[i])**2
    return(sqrt(sum))


class HardParzen:
    def __init__(self, h):
        self.h = h

    def train(self, train_inputs, train_labels): #train_labels contient des entiers
        self.train_inputs = train_inputs
        self.train_labels = train_labels
        self.label_list = np.unique(train_labels) #returns the sorted unique elements of an array

    def compute_predictions(self, test_data):
        predicted_labels = np.zeros(test_data.shape[0])
        
        for (iterate, test_point) in enumerate(test_data) :
            classes_neighbors = np.zeros(len(self.label_list))

            #calculate the standard euclidean distances between our test_point and the train_inputs
            distances = np.zeros(self.train_inputs.shape[0])
            for i in range (len(distances)):
                distances[i] = euclidean_distance(test_point, self.train_inputs[i,:]) #distances is an array containing the euclidean distances between test_point and all the samples from train_inputs
            
            #find the neighbors
            index_neighbors = []
            for i in range (len(distances)):
                if (distances[i] < self.h):
                    index_neighbors.append(i)
            
            #if there's no training point within the radius -> random choice
            if (len(index_neighbors) == 0):
                predicted_labels[iterate] = draw_rand_label(test_point, self.label_list)
            
            #calculate the number of neighbors belonging to each class
            for index in index_neighbors :
                #-1 because classes are labeled from 1 to n
                classes_neighbors[int(self.train_labels[index])-1] += 1 #classes_neighbors is an array containing the number of appearances of each labels of the neighbors within the radius
            
            #prediction (+1 car classes_neighbors[0] correspond au label 1, puisque les labels commencent a 1 et non a 0)
            predicted_labels[iterate] = np.argmax(classes_neighbors)+1
            

        return(predicted_labels)


def gaussian_density(x, mu, sigma):
    return( (1/(sigma*sqrt(2*np.pi))) * np.exp(-((x-mu)**2)/(2*(sigma**2))) )


class SoftRBFParzen:
    def __init__(self, sigma):
        self.sigma  = sigma #standard deviation

    def train(self, train_inputs, train_labels):
        self.train_inputs = train_inputs
        self.train_labels = train_labels
        self.label_list = np.unique(train_labels) #returns the sorted unique elements of an array

    def compute_predictions(self, test_data):
        predicted_labels = np.zeros(test_data.shape[0])

        for (iterate, test_point) in enumerate(test_data) :
            classes_neighbors = np.zeros(len(self.label_list))

            #calculate the standard euclidean distances between our test_point and the train_inputs
            distances = np.zeros(self.train_inputs.shape[0])
            for i in range (len(distances)):
                distances[i] = euclidean_distance(test_point, self.train_inputs[i,:]) #distances is an array containing the euclidean distances between test_point and all the samples from train_inputs
            
            #weight of the train_inputs according to their distance from our test_point
            weights = np.zeros(self.train_inputs.shape[0])
            for i in range (len(distances)) :
                weights[i] = gaussian_density(distances[i], 0, self.sigma)

            #calculate the number of neighbors belonging to each class with their weight
            for i in range (self.train_inputs.shape[0]):
                #-1 because classes are labeled from 1 to n
                classes_neighbors[int(self.train_labels[i])-1] += weights[i] #classes_neighbors is an array containing the number of appearances of each labels with the weight of the neighbor
            
            #prediction (+1 car classes_neighbors[0] correspond au label 1, puisque les labels commencent a 1 et non a 0)
            predicted_labels[iterate] = np.argmax(classes_neighbors)+1

        return(predicted_labels)
            

def split_dataset(iris):
    training_set = []
    validation_set = []
    test_set = []

    for indices in range (iris.shape[0]) :
        if (indices % 5 == 0 or indices % 5 == 1 or indices % 5 == 2) :
            training_set.append(iris[indices,:])
        if (indices % 5 == 3):
            validation_set.append(iris[indices,:])
        if (indices % 5 == 4):
            test_set.append(iris[indices,:])

    return(np.array(training_set), np.array(validation_set), np.array(test_set))


class ErrorRate:
    def __init__(self, x_train, y_train, x_val, y_val):
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val

    def hard_parzen(self, h):
        #create the classifier
        clf_hard_parzen = HardParzen(h)

        #train the classifier
        clf_hard_parzen.train(self.x_train, self.y_train)

        #prediction
        predicted_labels = clf_hard_parzen.compute_predictions(self.x_val)

        #error rate
        wrong_classifications = 0
        for i in range (len(predicted_labels)) :
            if (predicted_labels[i] != self.y_val[i]) :
                wrong_classifications += 1

        return(wrong_classifications/len(predicted_labels))

    def soft_parzen(self, sigma):
        #create the classifier
        clf_soft_parzen = SoftRBFParzen(sigma)

        #train the classifier
        clf_soft_parzen.train(self.x_train, self.y_train)

        #prediction
        predicted_labels = clf_soft_parzen.compute_predictions(self.x_val)

        #error rate
        wrong_classifications = 0
        for i in range (predicted_labels.shape[0]) :
            if (predicted_labels[i] != self.y_val[i]) :
                wrong_classifications += 1

        return(wrong_classifications/predicted_labels.shape[0])



def get_test_errors(iris):
    #Split dataset
    training_set, validation_set, test_set = split_dataset(iris)

    x_training_set = training_set[:,0:4]
    y_training_set = training_set[:,4]
    x_validation_set = validation_set[:,0:4]
    y_validation_set = validation_set[:,4]
    x_test_set = test_set[:,0:4]
    y_test_set = test_set[:,4]

    #Creer une instance de ErrorRate
    error_rate = ErrorRate(x_training_set, y_training_set, x_validation_set, y_validation_set)

    x_axis = [0.001, 0.01, 0.1, 0.3, 1.0, 3.0, 10.0, 15.0, 20.0]
    y_axis_hard_parzen = []
    y_axis_soft_parzen = []

    for i in range (len(x_axis)) :
        y_axis_hard_parzen.append(error_rate.hard_parzen(x_axis[i]))
        y_axis_soft_parzen.append(error_rate.soft_parzen(x_axis[i]))

    h_star = x_axis[np.argmin(y_axis_hard_parzen)]
    sigma_star = x_axis[np.argmin(y_axis_soft_parzen)]

    error = []
    #FAUX : recuperer error mais sur le test set et pas val_set
    error_rate_test_set = ErrorRate(x_training_set, y_training_set, x_test_set, y_test_set)
    error.append(error_rate_test_set.hard_parzen(h_star))
    error.append(error_rate_test_set.soft_parzen(sigma_star))

    return(np.array(error))


def random_projections(X, A):
    return( (1/sqrt(2))*np.dot(X,A) )




def error_random_projections(x_train, y_train, x_val, y_val):
    values = [0.001, 0.01, 0.1, 0.3, 1.0, 3.0, 10.0, 15.0, 20.0]
    #validation_errors = np.zeros((500,9))
    validation_errors = np.zeros(9)

    A=[]
    for i in range (4) :
        A.append([np.random.normal(0,1), np.random.normal(0,1)])
    A=np.array(A)

    x_train_proj = random_projections(x_train, A)
    x_val_proj = random_projections(x_val, A)

    error_rate = ErrorRate(x_train_proj, y_train, x_val_proj, y_val)
    for i in range (len(values)) :
        err = error_rate.hard_parzen(values[i])
        validation_errors[i]

    return(validation_errors)