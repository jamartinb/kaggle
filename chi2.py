import util
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import SelectKBest, chi2
from scipy import sparse
from itertools import combinations
from numpy import array, hstack

# TODO(jamartin): These would be great as class attributes.
hash_mappings = dict()
ctr = 0

def pre_process_data(X):
	util.print_debug_msg('Pre processing Data')
	x1 = array((X.ix[:]))
	x2 = group_data(x1, degree=2)
	x3 = group_data(x1, degree=3)
	#print x1.shape
	#print x2.shape
	#print x3.shape
	x_all = np.hstack((x1,x2,x3))
	enc = OneHotEncoder()
	enc.fit(x_all)
	x_transformed = enc.transform(x_all).toarray()
	return x_transformed

def my_hash(key):
	global ctr
	if key in hash_mappings:
		return hash_mappings[key]
	else:
		hash_mappings[key] = ctr
		ctr += 1
		return ctr-1

def group_data(data, degree=3, hash=my_hash):
	util.print_debug_msg('Grouping Data')
	new_data = []
	m,n = data.shape
	print m,n
	for indicies in combinations(range(n), degree):
		new_data.append([hash(tuple(v)) for v in data[:,indicies]])
	return array(new_data).T

def select_features(X, y):
	c = SelectKBest(chi2)
	c.fit_transform(X,y)
	print c.scores_
	print c.pvalues_
	
