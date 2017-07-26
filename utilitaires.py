'''
Created on 10 avr. 2017

@author: Pierrick
'''
# data analysis and wrangling
import numpy as np


def per(n):
    mat = np.zeros([2**n,n])
    for i in range(2**n):
        cur_vecteur = [int(x) for x in bin(i)[2:]]
        cur_longueur = len(cur_vecteur)
        mat[i,n-cur_longueur:n] = cur_vecteur
    return mat