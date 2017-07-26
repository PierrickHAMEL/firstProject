'''
Created on 10 avr. 2017

@author: Pierrick
view some correlation on the data

'''
import seaborn as sns
import matplotlib.pyplot as plt
from pylab import *


def first_view(train_df,test_df):

    #entete
    print(train_df.columns.values)

    # preview the data
    print(train_df.head())

    #qualite des donnees
    train_df.info()
    print('_'*40)
    test_df.info()

    #valeurs chiffrees
    print('_'*40)
    print(train_df.describe())
    # Review survived rate using `percentiles=[.61, .62]` knowing our problem description mentions 38% survival rate.
    # Review Parch distribution using `percentiles=[.75, .8]`
    # SibSp distribution `[.68, .69]`
    # Age and Fare `[.1, .2, .3, .4, .5, .6, .7, .8, .9, .99]`

    #valeurs character
    print('_'*40)
    print(train_df.describe(include=['O']))

 #------------------------------------------------------
#                    correlations
#------------------------------------------------------
def correlation(train_df,test_df):

    print('_'*40)
    print(' correlations')
    #classe
    print(train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False))

    print('_'*20)
    #sex
    print(train_df[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False))

    print('_'*20)
    #famille
    print(train_df[['SibSp', 'Survived']].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False))

    print('_'*20)
    #parents
    print(train_df[['Parch', 'Survived']].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False))


#------------------------------------------------------
#                    Histogrammes
#------------------------------------------------------
def dataplot(train_df,test_df,tracer_figure):

    if(tracer_figure):
        g = sns.FacetGrid(train_df, col='Survived')
        #tracer de l'histogramme, bins = nombre de colonnes
        g.map(plt.hist, 'Age', bins=20)
        #selon la classe et l'age
        grid = sns.FacetGrid(train_df, col='Survived', row='Pclass', size=2.2, aspect=1.6)
        #alpha = transparence
        grid.map(plt.hist, 'Age', alpha=0.5 , bins=20)
        grid.add_legend();

        #sex
        grid = sns.FacetGrid(train_df, row='Embarked', size=2.2, aspect=1.6)
        grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')
        grid.add_legend()

        #tarif
        grid = sns.FacetGrid(train_df, row='Embarked', col='Survived', size=2.2, aspect=1.6)
        #grid.map(plt.hist, 'Fare', alpha=0.5 , bins=20)
        grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)
        grid.add_legend()

        grid = sns.FacetGrid(train_df, row='Pclass', col='Sex', size=2.2, aspect=1.6)
        grid.map(plt.hist, 'Age', alpha=.5, bins=20)
        grid.add_legend()
        
        
        
        #on trace les figures
        draw()


def dataplot_post(train_df,test_df,tracer_figure):

    if(tracer_figure):
        g = sns.FacetGrid(train_df, col='Survived')
        #tracer de l'histogramme, bins = nombre de colonnes
        g.map(plt.hist, 'sqrtAge', bins=20)
        #selon la classe et l'age
        grid = sns.FacetGrid(train_df, col='Survived', row='Pclass', size=2.2, aspect=1.6)
        #alpha = transparence
        grid.map(plt.hist, 'sqrtAge', alpha=0.5 , bins=20)
        grid.add_legend();

        #sex
        grid = sns.FacetGrid(train_df, row='Embarked', size=2.2, aspect=1.6)
        grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')
        grid.add_legend()

        #tarif
        grid = sns.FacetGrid(train_df, row='Embarked', col='Survived', size=2.2, aspect=1.6)
        #grid.map(plt.hist, 'Fare', alpha=0.5 , bins=20)
        grid.map(sns.barplot, 'Sex', 'logFare', alpha=.5, ci=None)
        grid.add_legend()

        grid = sns.FacetGrid(train_df, row='Pclass', col='Sex', size=2.2, aspect=1.6)
        grid.map(plt.hist, 'sqrtAge', alpha=.5, bins=20)
        grid.add_legend()

        
        grid = sns.FacetGrid(train_df, col='Embarked', row='Pclass', size=2.2, aspect=1.6)
        #alpha = transparence
        grid.map(plt.hist, 'logFare', alpha=0.5 , bins=20)
        grid.add_legend();
        
        #on trace les figures
        draw()
