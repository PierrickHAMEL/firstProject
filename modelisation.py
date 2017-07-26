'''
Created on 11 avr. 2017

@author: Pierrick
List of model
'''
from collections import Counter

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Lasso

def select_model(Modele,modele_parameter):
    
    if(Modele == 'LogReg'):
        # Logistic Regression
        Train_Modele = LogisticRegression()
    elif(Modele == 'Ridge_Reg'):
        Train_Modele = Ridge(alpha = modele_parameter)
    elif(Modele == 'Kernel_Ridge'):
        Train_Modele = KernelRidge
    elif(Modele == 'Lasso'):
        Train_Modele = Lasso(alpha = modele_parameter)
    elif(Modele == 'Knn'):
        #Nearest Neigboor
        Train_Modele = KNeighborsClassifier(n_neighbors = 5)
    elif(Modele == 'STocGradient'):
        Train_Modele = SGDClassifier()
    elif(Modele == 'DecisionTree'):
        Train_Modele = DecisionTreeClassifier()
    elif(Modele == 'RandomForest'):
        Train_Modele = RandomForestClassifier(criterion='gini', 
                             n_estimators=700,
                             min_samples_split=10,
                             min_samples_leaf=1,
                             max_features='auto',
                             oob_score=True,
                             random_state=1,
                             n_jobs=-1)
        
    return Train_Modele

#------------------------------------------------------
#calcul pour un jeu de donnees l'erreur commise lors de la verif kfold
#le nombre de groupes pourrait etre un parametre
#------------------------------------------------------
def kfold_erreur_estimate(X_train,Y_train,Modele,modele_parameter):
    
    #nombre d'echantillons
    Nbdata = len(X_train)
    
    # Type de modele
    Train_Modele = select_model(Modele,modele_parameter)
    
    #nombre de division de l'echantillon de donnees
    kf = KFold(n_splits= int(Nbdata/10))
    #kf = KFold(n_splits= 10)
    
    #initialisation
    itest = 0
    moyenne_erreur = 0
    #Pour chaque paquet
    for train, test in kf.split(X_train):
        itest = itest + 1
        
        #input train et test
        X_train_loc = X_train.iloc[train]
        X_test_loc = X_train.iloc[test]
        
        #output train et test
        y_train_loc = Y_train.iloc[train]
        y_test_loc = Y_train.iloc[test]
        
        #adaptation du modele
        Train_Modele.fit(X_train_loc, y_train_loc)
        Y_pred_loc = Train_Modele.predict(X_test_loc)
        
        #ajout de l'erreur courante a l'erreur moyenne
        moyenne_erreur = moyenne_erreur + ((abs(y_test_loc-Y_pred_loc)).sum()/len(y_test_loc))
        
    moyenne_erreur = moyenne_erreur/itest
    return (1-moyenne_erreur)

#------------------------------------------------------
#estime le vecteur Ytest, a partir de X_test, 
#une fois le modele entraine sur X_train,Y_train
#------------------------------------------------------
def prediction(X_train,Y_train,X_test,Modele,modele_parameter):
    
    # Type de modele
    Train_Modele = select_model(Modele,modele_parameter)
    
    #entrainement du modele
    Train_Modele.fit(X_train, Y_train)
    
    #erreur estimee
    err_estimee = round(Train_Modele.score(X_train, Y_train) * 100, 2)
    print(err_estimee)
    
    #prediction
    Y_pred = Train_Modele.predict(X_test)
    
    #retour du resultat
    return Y_pred

#------------------------------------------------------
#ecriture du resultat sur un csv
#------------------------------------------------------
def write_result(Id_final,Y_final,pd):
    
    #print(liste.count(-1))
    submission = pd.DataFrame({
        "PassengerId": Id_final,
        "Survived": Y_final
    })
    
       
    submission.to_csv('output/submission_new.csv', index=False)
    
def class_model(X_train,Y_train,X_test,pd,train_df,test_df,np):
    
    Nbdata = len(X_train)
    
    # Logistic Regression
    logreg = LogisticRegression()
    logreg.fit(X_train, Y_train)
    Y_LRG = logreg.predict(X_test)
    acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
    print(acc_log)
    
    coeff_df = pd.DataFrame(train_df.columns.delete(0))
    coeff_df.columns = ['Feature']
    coeff_df["Correlation"] = pd.Series(logreg.coef_[0])
 
 
    #acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
    #print(acc_log)
    #print(coeff_df.sort_values(by='Correlation', ascending=False))

    # Support Vector Machines
    svc = SVC()
    svc.fit(X_train, Y_train)
    Y_pred = svc.predict(X_test)
    acc_svc = round(svc.score(X_train, Y_train) * 100, 2)
    #print(acc_svc)
    
    # k-Nearest Neighbors
    knn = KNeighborsClassifier(n_neighbors = 3)
    knn.fit(X_train, Y_train)
    Y_pred = knn.predict(X_test)
    acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
    #print(acc_knn)
    Y_KNC = Y_pred
    
    # Gaussian Naive Bayes
    gaussian = GaussianNB()
    gaussian.fit(X_train, Y_train)
    Y_pred = gaussian.predict(X_test)
    acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)
    #print(acc_gaussian)
    
    # Perceptron
    perceptron = Perceptron()
    perceptron.fit(X_train, Y_train)
    Y_pred = perceptron.predict(X_test)
    acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)
    #print(acc_perceptron)
    
    # Linear SVC
    linear_svc = LinearSVC()
    linear_svc.fit(X_train, Y_train)
    Y_pred = linear_svc.predict(X_test)
    acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)
    #print(acc_linear_svc)
    
    # Stochastic Gradient Descent
    sgd = SGDClassifier()
    sgd.fit(X_train, Y_train)
    Y_pred = sgd.predict(X_test)
    acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)
    #print(acc_sgd)
    
    Nbiter = 11
    Y_DTC = Y_pred * 0
    # Decision Tree
    for i in range(0,Nbiter):
        first = int(i*Nbdata/Nbiter)
        liste = range(first, min(first+int(Nbdata/Nbiter),Nbdata))
        X = X_train.drop(X_train.index[liste])
        Y = Y_train.drop(Y_train.index[liste])
        decision_tree = DecisionTreeClassifier()
        decision_tree.fit(X, Y)
        Y_pred = decision_tree.predict(X_test)
        acc_decision_tree = round(decision_tree.score(X, Y) * 100, 2)
        #print(acc_decision_tree)
        Y_DTC = Y_DTC + Y_pred 
    Y_DTC = Y_DTC/Nbiter
    #print(Y_DTC)
    
    Y_RFC = Y_pred * 0
    for i in range(0,Nbiter):
        # Random Forest
        first = int(i*Nbdata/Nbiter)
        liste = range(first, min(first+int(Nbdata/Nbiter),Nbdata))
        X = X_train.drop(X_train.index[liste])
        Y = Y_train.drop(Y_train.index[liste])
        
        random_forest = RandomForestClassifier(criterion='gini', 
                             n_estimators=700,
                             min_samples_split=10,
                             min_samples_leaf=1,
                             max_features='auto',
                             oob_score=True,
                             random_state=1,
                             n_jobs=-1)
#         random_forest = RandomForestClassifier(n_estimators=1000,oob_score=True)
        random_forest.fit(X, Y)
        Y_pred = random_forest.predict(X_test)
        random_forest.score(X, Y)
        acc_random_forest = round(random_forest.score(X, Y) * 100, 2)
        #print(acc_random_forest)
        #print("%.4f" % random_forest.oob_score_)
        Y_RFC = Y_RFC + Y_pred 
    Y_RFC = Y_RFC/Nbiter
    #print (Y_RFC)
    
    models = pd.DataFrame({
        'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
                  'Random Forest', 'Naive Bayes', 'Perceptron',
                  'Decision Tree'],
        'Score': [acc_svc, acc_knn, acc_log, 
                  acc_random_forest, acc_gaussian, acc_perceptron, 
                  acc_decision_tree]})
    #print(models.sort_values(by='Score', ascending=False))
    
    #print('Total score')
    #print(sum(models['Score']))
    

    liste = sorted(Y_KNC+ Y_DTC + Y_RFC)
    #print(type(liste))
    #print(100*(1-(liste.count(1)+liste.count(2))/418))
    #print(Y_DTC + Y_RFC)
    myList = Y_DTC + Y_RFC
    myList = [round(x) for x in myList]
    #print(myList)
    
    liste = sorted(myList)
    #print(type(liste))
    #print(100*(1-(liste.count(1))/418))
    #print(liste.count(1))
    longueur = len(liste)
    Y_final = Y_pred
#     for i in range(0,longueur):
#         if (Y_DTC[i] == Y_RFC[i]):
#             Y_final[i] = Y_RFC[i]
#         elif (np.sqrt(Y_DTC[i] + Y_RFC[i]) > 1):        
#             Y_final[i] = 1
#         elif (np.sqrt(Y_DTC[i] + Y_RFC[i]) < 1):
#             Y_final[i] = 0
#         else:
#             print(Y_DTC[i], Y_RFC[i])
#             Y_final[i] = -1
    
#     for i in range(0,longueur):
#         if (Y_RFC[i] > 0.5):        
#             Y_final[i] = 1
#         elif (Y_RFC[i] < 0.5):
#             Y_final[i] = 0
#         else:
#             print(Y_RFC[i])
#             Y_final[i] = -1
#                    
#     print(Y_final)
    
    Y_final = Y_LRG
    
    liste = sorted(Y_final)
    #print(liste.count(-1))
    submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": Y_final
    })
    submission.to_csv('output/submission.csv', index=False)