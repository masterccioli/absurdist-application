# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 22:27:04 2020

@author: maste
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 09:04:59 2020

@author: maste
"""

# import statements
import numpy as np
import pandas as pd

# my functions
import generalized_jaccard as jac
from get_wd import loadCorpus_windowed_words_regex as load

##########################################
#load/prepare text
items_a,dict_a = load('max.txt')
items_b,dict_b = load('otto.txt')

# reduce to items that occur multiple times
counts_a = list(np.where(np.array([i.shape[0] for i in items_a]) > 1)[0])
counts_b = list(np.where(np.array([i.shape[0] for i in items_b]) > 1)[0])


with open('test_words.txt','r') as file:
    words_use = file.read().splitlines()
    
words_a = words_use.copy()
words_a.append('max')

words_b = words_use.copy()
words_b.append('otto')

items_a_reduced = [items_a[dict_a[i]] for i in words_a]
items_b_reduced = [items_b[dict_b[i]] for i in words_b]


# get second order measure for every item in a
h_a = jac.get_jaccard_matrix_simp(items = items_a_reduced,
                                  function = jac.hausdorff_euclid)
h_b = jac.get_jaccard_matrix_simp(items = items_b_reduced,
                                  function = jac.hausdorff_euclid)

#################################################
# code absurdist
# using simple example from paper
a = np.array([[0,7,9],
              [7,0,2],
              [9,2,0]])
b = np.array([[0,3,10],
              [3,0,7],
              [10,7,0]])

beta = .5
chi = .5

c_t = np.ones((a.shape[0], b.shape[0]))*.5
print(c_t)
for step in range(0,20):
    # get sim
    R = np.zeros((a.shape[0], b.shape[0]))
    for q in range(0,a.shape[0]):
#        print(q)
        for x in range(0,b.shape[0]):
            S = 0
            for r in range(0,a.shape[0]):
                if r == q:
                    continue
                for y in range(0,b.shape[0]):
                    if y == x:
                        continue
                    S += np.exp(-np.abs(a[q,r] - b[x,y]))*c_t[r,y]
            S = S / (np.min([a.shape[0],b.shape[0]]) - 1)
            R[q,x] = S
            
    # get inhibition
    I = np.zeros((a.shape[0], b.shape[0]))
    for q in range(0,a.shape[0]):
        for x in range(0,b.shape[0]):
            i = 0
            for r in range(0,a.shape[0]):
                if r == q:
                    continue
                i += c_t[r,x]
            for y in range(0,b.shape[0]):
                if y == x:
                    continue
                i += c_t[q,y]
                
            i = i / (a.shape[0] + b.shape[0] - 2)
            I[q,x] = i
    N = beta*R - chi*I
    
    less = np.where(N < 0)
    c_t[less] = c_t[less] + N[less]*(c_t[less] - 0) * 1
    greater = np.where(N >= 0)
    c_t[greater] = c_t[greater] + N[greater]*(1 - c_t[greater]) * 1
    print(c_t)
    
#####################################
# reduce operation time by removing 
# for loops in favor of matrix operations
# try using corpus
a = h_a
b = h_b
a_ = a[~np.eye(a.shape[0],dtype=bool)].reshape(a.shape[0],-1)
b_ = b[~np.eye(b.shape[0],dtype=bool)].reshape(b.shape[0],-1)

beta = .5
chi = .5
maxL2 = 1#2.0/( beta * (np.max([a.shape[0],b.shape[0]])-1) + 2 * chi)

c_t = np.ones((a.shape[0], b.shape[0]))
print(c_t)
for step in range(0,30):
    print(step)
    ### similarity
    R = np.exp(-np.abs(np.subtract.outer(a_,b_)))
    R = c_t[:,None,:,None] * R[:,:,:,:]
#    R = R.sum(axis=(1,3)) / (np.min([a.shape[0],b.shape[0]]) - 1)
    R = R.sum(axis=(1,3)) 
    R = R / R.sum(0) # keeps this between zero and one - seems to work better
    # maybe because it treats it more like a probability distribution
    
    ### inhibition
    # get inhibition
    I = np.zeros((a.shape[0], b.shape[0]))
    for q in range(0,a.shape[0]):
        for x in range(0,b.shape[0]):
            i = 0
            for r in range(0,a.shape[0]):
                if r == q:
                    continue
                i += c_t[r,x]
            for y in range(0,b.shape[0]):
                if y == x:
                    continue
                i += c_t[q,y]
                
            i = i / (a.shape[0] + b.shape[0] - 2)
            I[q,x] = i
    
    ### N
    N = beta*R - chi*I
        
    less = np.where(N < 0)
    c_t[less] = c_t[less] + N[less]*(c_t[less] - 0) * maxL2
    greater = np.where(N >= 0)
    c_t[greater] = c_t[greater] + N[greater]*(1 - c_t[greater]) * maxL2
print(c_t)
c_t = pd.DataFrame(c_t)
c_t.columns = words_b
c_t.index = words_a    
