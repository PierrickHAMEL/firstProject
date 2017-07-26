'''
Created on 10 avr. 2017

@author: Pierrick
'''

import cmath

#------------------------------------------------------
#retire les tickets et les cabines
#------------------------------------------------------
def remove_data(train_df,test_df,combine):
    print("Before", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape)

    train_df = train_df.drop(['Ticket', 'Cabin'], axis=1)
    test_df = test_df.drop(['Ticket', 'Cabin'], axis=1)
    combine = [train_df, test_df]
    
    print("After", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape)
    return train_df,test_df,combine

#------------------------------------------------------
#travaille sur les noms, les transforme en titre
#ages transforme en classes
#------------------------------------------------------
def merge_data(train_df,test_df,combine,pd):
    for dataset in combine:
        dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
    
    pd.crosstab(train_df['Title'], train_df['Sex'])
    
    for dataset in combine:
        dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
         'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    
        dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
        dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
        dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    
    train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
    
    #suppression des valeurs nom et eventuellement id
    train_df = train_df.drop(['Name', 'PassengerId'], axis=1)
    test_df = test_df.drop(['Name'], axis=1)
    combine = [train_df, test_df]
    train_df.shape, test_df.shape
    
    #Regroupement par classe d'age
    train_df['AgeBand'] = train_df['Age']
    test_df['AgeBand'] = test_df['Age'] 
    #donne une valeur a chaque classe d'age
    for dataset in combine:    
        dataset.loc[ dataset['Age'] <= 16, 'AgeBand'] = 0
        dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'AgeBand'] = 1
        dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'AgeBand'] = 2
        dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'AgeBand'] = 3
        dataset.loc[ dataset['Age'] > 64, 'AgeBand'] = 4
    
    #compte le nombre de personnes dans la famille
    for dataset in combine:
        dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
    
    #train_df[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False)
    
    #indique si une personne est seule
    for dataset in combine:
        dataset['IsAlone'] = 0
        dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1
    
    #train_df[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean()
    
    
    #on range par classe les prix des billets
    #1 - Visualisation
    train_df['FareBand'] = train_df['Fare']
    test_df['FareBand'] = test_df['Fare'] 
    #2 - Application
    for dataset in combine:
        dataset.loc[ dataset['Fare'] <= 7.91, 'FareBand'] = 0
        dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'FareBand'] = 1
        dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'FareBand']   = 2
        dataset.loc[ dataset['Fare'] > 31, 'FareBand'] = 3
        dataset['FareBand'] = dataset['FareBand'].astype(int)
    
    
    
    return train_df,test_df,combine,pd

#------------------------------------------------------
#donne des valeurs aux differentes variable alpha
#------------------------------------------------------
def alpha2Num(combine,np):
    
    #donne une valeur numerique au titre
    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
    for dataset in combine:
        dataset['Title'] = dataset['Title'].map(title_mapping)
        dataset['Title'] = dataset['Title'].fillna(0)
    
    #donne une valeur au sex
    for dataset in combine:
        dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)
    
    #on rajoute juste une classe
    for dataset in combine:
        dataset['AgeBand*Class'] = dataset.AgeBand * dataset.Pclass
    
    #on rajoute juste une classe
    for dataset in combine:
        dataset['logFare'] = dataset.Fare + 1
        dataset['logFare'] = np.log(dataset.logFare) 
    
    #on rajoute juste une classe
    for dataset in combine:
        dataset['sqrtAge'] = np.sqrt(dataset.Age) 
    
    #on rajoute juste une classe
    for dataset in combine:
        dataset['sqrtFamilySize'] = np.sqrt(dataset.FamilySize) 
          
             
    #donne une valeur au port de d'origine
    for dataset in combine:
        dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 1.1} ).astype(int)

    return combine

#------------------------------------------------------
#complete les trous dans les donnees
#------------------------------------------------------
def fill_data(train_df,test_df,combine,np):
    guess_ages = np.zeros((2,3))
    sex = ['male' , 'female']
    for dataset in combine:
        for i in range(0, 2):
            for j in range(0, 3):
#                 guess_df = dataset[(dataset['Sex'] == i) & \
#                                   (dataset['Pclass'] == j+1)]['Age'].dropna()
                guess_df = dataset[(dataset['Sex'] == sex[i]) & \
                                      (dataset['Pclass'] == j+1)]['Age'].dropna()
     
                # age_mean = guess_df.mean()
                # age_std = guess_df.std()
                # age_guess = rnd.uniform(age_mean - age_std, age_mean + age_std)
    
                age_guess = guess_df.median()
                
                # Convert random age float to nearest .5 age
                guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5
                
        for i in range(0, 2):
            for j in range(0, 3):
                dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex == sex[i]) & (dataset.Pclass == j+1),\
                        'Age'] = guess_ages[i,j]
    
        dataset['Age'] = dataset['Age'].astype(int)
    
    #on cherche d'abord le port d'ou provient le plus de monde    
    freq_port = train_df.Embarked.dropna().mode()[0]
    #par defaut c'est de ce port qu e viennent les personnes pour lesquelles ce champ n'est pas renseigne
    for dataset in combine:
        dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)
        
    #donne une valeur de prix d'achat du billet pour ceux qui n'en ont pas
    test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)
    test_df.head()
    
    return train_df,test_df,combine

#------------------------------------------------------
#retire des donnees dans les differents tableaux
#------------------------------------------------------
def select_features(train_df,test_df,vecteur,feature_liste):
    
    #on parcourt le vecteur
    for i in range(1,len(vecteur)):
        
        #nom dufeature associe
        feature_name = feature_liste[i]
        #doit-on prendre ce feature
        indice_veceur = vecteur[i] 
        
        #non s'il vaut 0 dans le vecteur
        if (indice_veceur==0.):
            #print (feature_name)
            train_df = train_df.drop([feature_name], axis=1)
            test_df = test_df.drop([feature_name], axis=1)
            i = i+1
    
    X_train = train_df.drop("Survived", axis=1)
    Y_train = train_df["Survived"]
    Id_test = test_df["PassengerId"]
    X_test  = test_df.drop("PassengerId", axis=1).copy()
    
    
    return X_train,Y_train,X_test,Id_test

    
    