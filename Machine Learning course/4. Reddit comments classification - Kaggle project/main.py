# coding: utf-8

#########################################################

#   Auteurs :
#   Luis Pedro Dos Santos
#   David Kletz
#   
#   Groupe Name on Kaggle : Top 1 nous voilà

#########################################################


#########################################################

#LIBRARIES
import numpy as np
from random import randrange
from collections import Counter
import nltk
import re #re est un module python utilise pour la tokenization
import time

#IMPORT DATA
data_train = np.load('data_train.pkl', allow_pickle=True)
data_test = np.load('data_test.pkl', allow_pickle=True)

#########################################################

#USAGE DES FICHIERS

#print(data_train[0]) : print tous les commentaire ['','',...,'']
#print(data_train[1]) : print tous les subreddits ['','',...,'']

#print(data_train[0][0]) : print le 1er commentaire (string)
#print(data_train[1][0]) : print le 1er subreddit

#########################################################



''' ALGO 1 : RANDOM PREDICTION'''


def random_prediction(labels, length) :
    """
    Cree un fichier csv avec une prediction aleatoire "length" utilisant les labels (subreddits) contenus dans la list des labels

    :param labels: python str list
    :param length: int
    :return: 0
    """
    pred = np.array([ ['Id','Category'] ])
    for i in range (length) :
        pred = np.append(pred, [ [str(i),labels[randrange(20)]] ], axis=0)
    np.savetxt("foo.csv", pred, delimiter=",", fmt='%s')
    return 0


#Creer une liste contenant les differents subreddits (categories)
different_labels = []
for label in data_train[1] :
    if (label not in different_labels) :
        different_labels.append(label)

#Predictions
#random_prediction(different_labels, len(data_test))


###########################################################################################################################################################################
###########################################################################################################################################################################
###########################################################################################################################################################################



''' ALGO 2 : NAIVE BAYES'''


##################   PREPROCESSING   ##################

print('\nShuffling data and creating train and valid...')
t1 = time.clock()

#Ici, on shuffle et on cree un training set et un valid set pour avoir une estimation de nos predictions sans avoir a soumettre tout le temps sur gradescope
#Pour la submission Kaggle, taille_train a ete definie a 69999 pour entrainer notre code sur 69999 commentaires
indexes = np.arange(len(data_train[0]))
np.random.shuffle(indexes)
taille_train = 69999
indexes_train = indexes[:taille_train]
indexes_valid = indexes[taille_train:]

train_set_comments =  np.array(data_train[0])[indexes_train]
train_set_labels =  np.array(data_train[1])[indexes_train]
valid_set_comments = np.array(data_train[0])[indexes_valid]
valid_set_labels =  np.array(data_train[1])[indexes_valid]
valid_set_labels = np.array([different_labels.index(current_class) for current_class in valid_set_labels])

#Dictionnaire qui met en lien les classes avec tous les commentaires correspondant
dico_commentaires_avec_classe = {class_index : [] for class_index in range (len(different_labels))}
for i, current_class in enumerate(train_set_labels) :
    dico_commentaires_avec_classe[different_labels.index(current_class)].append(train_set_comments[i])
print('Done')


#Dictionnaire Counter de tous le corpus et du corpus de chaque classe

print('\nCreation du dictionnaire lie a tous les commentaires de train...')
count = Counter()
count_classes=[] #Contient 20 dictionnaires contenant la frequence de tous les mots d'une classe
for i in range (len(different_labels)) :
    count_intermediraire_pour_chaque_classe = Counter()
    for j in range (len(dico_commentaires_avec_classe[i])):
        one_comment = re.findall("\w+", dico_commentaires_avec_classe[i][j])
        for k in range (len(one_comment)) :
            #Tout mettre en minuscule
            one_comment[k] = one_comment[k].lower()

        count.update(one_comment)
        count_intermediraire_pour_chaque_classe.update(one_comment)
    count_classes.append(count_intermediraire_pour_chaque_classe)
print('Done')


#Stop words de NLTK

#nltk.download('stopwords')
stop_nltk = nltk.corpus.stopwords
stop_words = set(stop_nltk.words('english'))
for elem in stop_words :
    del count[elem]
    for i in range (len(different_labels)) :
        del count_classes[i][elem]

#Vocabulaire des mots uniques
cutoff = 0
vocab = [word for word in count if count[word] > cutoff]
print('\nTaille du vocabulaire : ', len(vocab),'\n')





##################   CALCUL DES PROBABILITES AVEC LE VALID SET (pour avoir une estimation de l'accuracy)   ##################

'''

probas = []
priors = [len(dico_commentaires_avec_classe[i])/taille_train for i in range (len(different_labels))]

#Pour chaque classe
for i in range (len(different_labels)) :
    print('\nCalcul des probabilites des commentaires pour la classe ',i)
    proba_exemples_une_classe = []

    #Pour chaque commentaire de valid
    for j in range (len(valid_set_comments)) :
        proba_un_exemple=0

        #Tokenize le commentaire
        words_commentaire_j_valid = re.findall("\w+", valid_set_comments[j])

        #Pour chaque mot du commentaire
        for k in range (len(words_commentaire_j_valid)) :
            
            #Tout mettre en minuscule
            words_commentaire_j_valid[k] = words_commentaire_j_valid[k].lower()

            #Si cest pas un stop words
            if (count_classes[i][words_commentaire_j_valid[k]] != 0) :
                proba_un_exemple += (count_classes[i][words_commentaire_j_valid[k]] / count[words_commentaire_j_valid[k]])
 
        proba_exemples_une_classe.append(proba_un_exemple + priors[i])
    probas.append(proba_exemples_une_classe)    

probas=np.array(probas)
'''




##################   CALCUL DES PROBABILITES AVEC LE TEST SET   ##################

probas = []
priors = [len(dico_commentaires_avec_classe[i])/taille_train for i in range (len(different_labels))]

#Pour chaque classe
for i in range (len(different_labels)) :
    print('\nCalcul des probabilites des commentaires pour la classe ',i)
    proba_exemples_une_classe = []

    #Pour chaque commentaire de test
    for j in range (len(data_test)) :
        proba_un_exemple=0

        #Tokenize le commentaire
        words_commentaire_j_test = re.findall("\w+", data_test[j])

        #Pour chaque mot du commentaire
        for k in range (len(words_commentaire_j_test)) :
            
            #Tout mettre en minuscule
            words_commentaire_j_test[k] = words_commentaire_j_test[k].lower()

            #Si cest pas un stop words
            if (count_classes[i][words_commentaire_j_test[k]] != 0) :
                proba_un_exemple += (count_classes[i][words_commentaire_j_test[k]] / count[words_commentaire_j_test[k]])
 
        proba_exemples_une_classe.append(proba_un_exemple + priors[i])
    probas.append(proba_exemples_une_classe)    

probas=np.array(probas)




##################   CLASSIFIER   ##################

#On veut la classe pour laquelle la proba est la plus grande
classes_pred = probas.argmax(axis=0)
#print("\n\nThe training accuracy is : {:.1f} %".format(100 * np.mean(classes_pred == valid_set_labels)))




##################   NOMBRE DE PREDICTIONS POUR CHAQUE CLASSE   ##################

dico_taux_pred_classes = {class_name : 0 for class_name in (different_labels)}
for current_pred_index in (classes_pred) :
    dico_taux_pred_classes[different_labels[int(current_pred_index)]] += 1
print('\n', dico_taux_pred_classes)




'''
##################   .CSV FILE   ##################

pred = np.array([ ['Id','Category'] ])
for i in range (len(classes_pred)) :
    pred = np.append(pred, [ [str(i),different_labels[int(classes_pred[i])]] ], axis=0)
np.savetxt("pred_NaiveBayes_sansGauss.csv", pred, delimiter=",", fmt='%s')
'''




##################   COMPUTATION TIME   ##################

t2 = time.clock()
print('\nTIME for algo 2 : ', round(t2 - t1, 2), 's\n\n')