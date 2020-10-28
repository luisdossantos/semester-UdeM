
#####################################
README Kaggle : Première étape
#####################################

Groupe : Top 1 Nous voilà
Etudiants : Luis Pedro DOS SANTOS, David KLETZ

<br />
<br />

1) gestion du texte.

Le texte est tokenisé afin de récupérer une liste de tout le vocabulaire disponible dans les commentaires d'entrainement.
Les mots du texte sont ensuite enregistrés dans une liste.
Pour chaque classe, on crée un dictionnaire associant un mot à son nombre d'occurence dans tous les commentaires de la classe. De même on crée le même dictionnaire pour le total.


2) entrainement des modèles.

Des fonctions de vraisemblance sont créées. Chacune calcule la vraisemblance qu'un commentaire soit d'une classe donnée.
Les classes sont entrainées, c'est à dire qu'elles associent à chaque mot disponible dans la classe la probabilité qu'un commentaire le contenant fasse effectivement partie de la classe donnée.
Cette probabilité est définie comme le rapport entre le nombre d'occurences de ce mot au total dans la classe et les occurences dans tous les commentaires.
La probabailité d'une phrase est calculée comme la somme des probabilités de chaque mot (probabilité de 0 si le mot n'est pas vu à l'entrainement).


3) classification.

Un Bayes Naif récupère les probabilités données par les fonctions de vraisemblance associée à chaque classe. Il associe à chaque commentaire le label correspondant à la probabilité maximale donnée par les fonctions de vraisemblance 





