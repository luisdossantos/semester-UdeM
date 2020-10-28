# coding: utf-8
#########################################################

#LIBRARIES
import numpy as np
from random import randrange
from collections import Counter
import nltk
import re, csv
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
import time
import sys
import spacy
import pandas as pd


#IMPORT DATA

# globals
blog = 'blog'
classe = 'class'

# init the spacy pipeline
# !!! desactivate components if you only need tokenization
# (as a default, you get pos tagging, parsing, etc.) !!!
print("\nSpacy load")
nlp_train = spacy.load("en_core_web_sm", disable=["parser", "tagger", "ner"])
nlp_test = spacy.load("en_core_web_sm", disable=["parser", "tagger", "ner"])
print('Done..')

# reading a csv file (from stdin) with panda is easy
# question: why from stdin ?
# memory load: initial csv x 2 (roughly)
# but possible to stream with the chunksize parameter (read the doc)
print('\nConversion en pandas data frame')
df_train = pd.read_csv('train_posts.csv', names=[blog,classe])
df_test = pd.read_csv('test_split01.csv', names=[blog,classe])
print('Done..')

# Example
#print(df_train.sample(5))

# of course, you can iterate line by line
# note that in this example, there is only tokenization done (no normalisation)
'''
for index, row in df_train.iterrows():
    sent = nlp_train(row[blog])
    words = [tok.text for tok in sent]
    print("{}\t{}\t[{}]: {}".format(index, row[classe], len(words), " ".join(words[:10])))
'''


#Creer une liste contenant les differents labels (categories)
different_labels = [0, 1, 2]



''' ALGO 2 : NAIVE BAYES'''



##################   PREPROCESSING   ##################

t1 = time.clock()


print('\nCreation du dictionnaire lie au vocabulaire...')
count = Counter()
t100 = time.clock()

#Pour chaque commentaire (on ne prend que les X 1ers commentaires)
N_comments_took_in_account = 10000
for index, row in df_train.head(n=N_comments_took_in_account).iterrows():
    if (index%1000 == 0) :
        print('\n\n\n ################################ TRAIN : CREATION DU DICTIONNAIRE %f ################################' % ((index/N_comments_took_in_account)*100))
        if (index != 0):
            t101=time.clock()
            print('Temps restant estime : ', ( int((round(t101 - t100, 2)*100)/((index/N_comments_took_in_account)*100) - round(t101 - t100, 2))), 's')

    #On recupere chaque commentaire (sans le label)
    sent = nlp_train(row[blog])
    #Tokenization
    words = [tok.text for tok in sent]
    #En minuscule
    for i in range (len(words)) :
        words[i] = words[i].lower()
    count.update(words)

print('Done')



#nltk.download('stopwords')
stop_nltk = nltk.corpus.stopwords
stop_words = set(stop_nltk.words('english'))
for elem in stop_words :
    del count[elem]

#Mots differents
cutoff = 20
vocab = [word for word in count if count[word] > cutoff]
print('\nTaille du vocabulaire : ', len(vocab))



##################   TRAIN : FEATURES SET AVEC LAPLACIAN SMOOTHING   ##################

t10 = time.clock()
features_set_train = []
d = len(vocab)
alpha = 0.008
t100 = time.clock()

for index, row in df_train.head(n=N_comments_took_in_account).iterrows():
    if (index%1000 == 0) :
        print('\n\n\n ################################ TRAIN : BAG OF WORDS %f ################################' % ((index/N_comments_took_in_account)*100))
        if (index != 0):
            t101=time.clock()
            print('Temps restant estime : ', ( int((round(t101 - t100, 2)*100)/((index/N_comments_took_in_account)*100) - round(t101 - t100, 2))), 's')

    #On recupere chaque commentaire (sans le label)
    sent = nlp_train(row[blog])
    #Tokenization
    words_train_set_comment_index = [tok.text for tok in sent]
    #En minuscule
    for i in range (len(words_train_set_comment_index)) :
        words_train_set_comment_index[i] = words_train_set_comment_index[i].lower()

    #Nb de mots pour chaque commentaire
    N = len(words_train_set_comment_index)

    #Dictionnaire pour le commentaire index
    count_vocab_train_set_comment_index = Counter(words_train_set_comment_index)

    '''
    AVEC LAPLACE
    #features_for_one_example = [0]*(len(vocab))
    features_for_one_example=[]
    for j in range (len(vocab)) :
        features_for_one_example.append((count_vocab_train_set_comment_index[vocab[j]] + alpha)/(N + alpha*d))
    '''
    features_for_one_example = [0]*(len(vocab))
    for j in range (len(vocab)) :
        if (count_vocab_train_set_comment_index[vocab[j]] != 0) :
            features_for_one_example[j] = count_vocab_train_set_comment_index[vocab[j]]

    features_set_train.append(features_for_one_example)

t11 = time.clock()
print('\n\n\nFeatures set pour train cree en : ', round(t11 - t10, 2), 's')

t12 = time.clock()
features_set_train = np.array(features_set_train)
t13 = time.clock()
print('Features set de train converti en array en : ', round(t13 - t12, 2), 's\n')




t14 = time.clock()
features_set_test = []

for index, row in df_test.iterrows():
    if (index%1000 == 0) :
        print('\n\n\n ################################ TEST : BAG OF WORDS %f ################################' % ((index/df_test.shape[0])*100))
        if (index != 0):
            t101=time.clock()
            print('Temps restant estime : ', ( int((round(t101 - t100, 2)*100)/((index/df_test.shape[0])*100) - round(t101 - t100, 2))), 's')

    #On recupere chaque commentaire (sans le label)
    sent = nlp_test(row[blog])
    #Tokenization
    words_test_set_comment_index = [tok.text for tok in sent]
    #En minuscule
    for i in range (len(words_test_set_comment_index)) :
        words_test_set_comment_index[i] = words_test_set_comment_index[i].lower()

    #Nb de mots pour chaque commentaire
    N = len(words_test_set_comment_index)

    #Dictionnaire pour le commentaire index
    count_vocab_test_set_comment_index = Counter(words_test_set_comment_index)

    features_for_one_example = [0]*(len(vocab))
    for j in range (len(vocab)) :
        if (count_vocab_test_set_comment_index[vocab[j]] != 0) :
            features_for_one_example[j] = count_vocab_test_set_comment_index[vocab[j]]

    features_set_test.append(features_for_one_example)

t15 = time.clock()
print('\n\n\nFeatures set pour test cree en : ', round(t15 - t14, 2), 's')

t16 = time.clock()
features_set_test = np.array(features_set_test)
t17 = time.clock()
print('Features set de test converti en array en : ', round(t17 - t16, 2), 's\n')






##################   MODELE (comparaison avec le Naive Bayes de Scikit)   ##################
gnb = GaussianNB()
#clf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
print('\nFit model')
gnb.fit(features_set_train, df_train.head(n=N_comments_took_in_account)['class'])
#clf.fit(features_set_train, df_train.head(n=N_comments_took_in_account)['class'])
print('Done...')
print('\nPred model')
y_pred = gnb.predict(features_set_test)
#y_pred = clf.predict(features_set_test)
print('Done...')


print('\nTRAIN : Nombre de commentaire pour chaque classe')
dico_taux_classes_train = {class_name : 0 for class_name in (different_labels)}
for current_index_label in (df_train.head(n=N_comments_took_in_account)['class']) :
    dico_taux_classes_train[current_index_label] += 1
print('\n', dico_taux_classes_train)



print('\nTEST : Nombre de commentaire pour chaque classe')
dico_taux_classes_test = {class_name : 0 for class_name in (different_labels)}
for current_index_label in (df_test['class']) :
    dico_taux_classes_test[current_index_label] += 1
print('\n', dico_taux_classes_test)



print('\nNombre de predictions pour chaque classe')
dico_taux_pred_classes = {class_name : 0 for class_name in (different_labels)}
for current_pred_index in (y_pred) :
    dico_taux_pred_classes[current_pred_index] += 1
print('\n', dico_taux_pred_classes)


print("\n\nThe training accuracy is : {:.1f} % ".format(100*np.mean(y_pred == df_test['class'])))





t2 = time.clock()
print('\nTIME for algo 2 : ', round(t2 - t1, 2), 's\n\n')
