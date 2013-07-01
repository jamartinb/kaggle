import numpy as np
from numpy import array
import util
from sklearn.preprocessing import OneHotEncoder
from sklearn import svm
from itertools import combinations

def pre_process_data(X,Y):
	'''#x1 = array((X.ix[:]))
	x1 = array(X)
	x2 = group_data(x1, degree=2)
	x3 = group_data(x1, degree=3)
	x_all = np.hstack((x1,x2,x3))
	
	#y1 = array((y.ix[:]))
	y1 = array(Y)
	y2 = group_data(y1, degree=2)
	y3 = group_data(y1, degree=3)
	y_all = np.hstack((y1,y2,y3))
	return x_all, y_all'''
	return X,Y

def group_data(data, degree=3, hash=hash):
	util.print_debug_msg('Grouping Data')
	new_data = []
	m,n = data.shape
	#print m,n
	for indicies in combinations(range(n), degree):
		new_data.append([hash(tuple(v)) for v in data[:,indicies]])
	return array(new_data).T

def train_predict(X_train, y_train, X_test):
	util.print_debug_msg('Training SVM Classifier')
	model = svm.SVR(kernel='linear', verbose=True)
	model.fit(X_train, y_train)
	util.print_debug_msg('Predicting SVM Classifier')
	return model.predict(X_test)

