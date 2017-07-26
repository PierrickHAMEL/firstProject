'''
Created on 10 avr. 2017

@author: Pierrick
'''
# data analysis and wrangling
import pandas as pd
import numpy as np
#import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
from pylab import *



import data_preview
import repair_data
import modelisation
from _tracemalloc import stop

import utilitaires

#ecriture d'un tableau dans un csv
#a = np.asarray([ [1,2,0], [4,5,6], [7,8,9] ])
#np.savetxt("foo.csv", a, delimiter=",")

#0 Parametres
#----------------
#Modele = 'LogReg'
#Modele = 'Ridge_Reg'
Modele = 'Lasso'
#Modele = 'Kernel_Ridge'
#Modele = 'Knn'
#Modele = 'STocGradient'
#Modele = 'DecisionTree'
#Modele = 'RandomForest'
modele_parameter = 10
separation_homme_femme = False
tracer_figure = False
Optimised_feature_vecteur = True
NbVectorTest = 2000

#data load
train_df = pd.read_csv('input/train.csv')
test_df = pd.read_csv('input/test.csv')
combine = [train_df, test_df]

#1 Visialisation
#----------------
#etat des donnees
data_preview.first_view(train_df,test_df)

#correlations sur les donnees
data_preview.correlation(train_df,test_df)

#tracer d'histogrammes
data_preview.dataplot(train_df,test_df,tracer_figure)

#2 Mise en ordre des donnees
#-----------------------------
#supression des ticket et des cabines
train_df,test_df,combine = repair_data.remove_data(train_df,test_df,combine)
print("After", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape)

#remplit les trous dans les donnees
train_df,test_df,combine = repair_data.fill_data(train_df,test_df,combine,np)

#transformation des noms en titres puis en valeurs numeriques
train_df,test_df,combine,pd = repair_data.merge_data(train_df,test_df,combine,pd)

    
#remplace les valeurs alpha en numerique
combine = repair_data.alpha2Num(combine,np)

data_preview.dataplot_post(train_df,test_df,tracer_figure)

print(train_df.head())

#liste des features au maximum
features = train_df.columns.values
#Nombre de features, le -1 est la car survived n'est pas une feature
Nb_features = len(features)-1
print(features)

matrice = np.zeros([2**Nb_features,Nb_features])
matrice = utilitaires.per(Nb_features)


score = np.zeros(2**Nb_features)
best_erreur = 0.
best_vecteur = np.ones(Nb_features)

if (Optimised_feature_vecteur):
    for i in range(0,NbVectorTest): #2**Nb_features):
        print (i)
        vecteur = matrice[i,0:Nb_features]
        #vecteur = np.ones(Nb_features)
        np.random.seed(i)
        vecteur = np.random.randint(2, size=Nb_features)
        #vecteur = [1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1]
        #retire certaines colonnes/features des donnees, selon le vecteur indique
        X_train,Y_train,X_test,Id_test = repair_data.select_features(train_df,test_df,vecteur,features)
        
        
        #est-ce qu'on a une meilleure precision
        erreur_loc = modelisation.kfold_erreur_estimate(X_train,Y_train,Modele,modele_parameter)
        if(best_erreur<erreur_loc):
            best_erreur = erreur_loc
            best_vecteur = vecteur
            print (best_erreur)
            print (best_vecteur)
    
    print (best_vecteur)


#3 Modelisation
#---------------------------

if (separation_homme_femme==False):
    if(Optimised_feature_vecteur):
        vecteur = best_vecteur
    else:
        #meilleur obtenu en logReg
        #score 0.78469
        #vecteur = [1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1]
        #obtenu pour Knn
        #score 0.75598
        #vecteur = [0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0]
        #obtenu pour decision tree
        #score 0.78469
        vecteur = [0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0]
        
    X_train,Y_train,X_test,Id_final = repair_data.select_features(train_df,test_df,vecteur,features)
    Y_final = modelisation.prediction(X_train,Y_train,X_test,Modele,modele_parameter)
else:
    #prediction pour les femmes
    #sex = 0
    if(Optimised_feature_vecteur):
        vecteur = best_vecteur
    else:
        #meilleur obtenu en logReg
        vecteur = [0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1]
    
    X_train,Y_train,X_test0,Id_test0 = repair_data.select_features(train_df[train_df['Sex']==0],test_df[test_df['Sex']==0],vecteur,features)
    Y_test0 = modelisation.prediction(X_train,Y_train,X_test0,Modele,modele_parameter)
    
    #sex = 1
    if(Optimised_feature_vecteur): 
        vecteur = best_vecteur
    else:
        #meilleur obtenu en logReg
         vecteur = [0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1,]

    X_train,Y_train,X_test1,Id_test1 = repair_data.select_features(train_df[train_df['Sex']==1],test_df[test_df['Sex']==1],vecteur,features)
    Y_test1 = modelisation.prediction(X_train,Y_train,X_test1,Modele,modele_parameter)
    
    #concatenation des resultats
    Y_final = np.append(Y_test0,Y_test1)
    Id_final = np.append(Id_test0,Id_test1)


#ecriture du resultat
modelisation.write_result(Id_final,Y_final,pd)


#classification des modeles
#modelisation.class_model(X_train,Y_train,X_test,pd,train_df,test_df,np)

#fin programme
print('fini')
show()