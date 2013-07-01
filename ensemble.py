import numpy as np
from numpy import array
import util
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from itertools import combinations

def pre_process_data(X,Y):
	#x1 = array((X.ix[:]))
	x1 = array(X)
	x2 = group_data(x1, degree=2)
	x3 = group_data(x1, degree=3)
	x_all = np.hstack((x1,x2,x3))
	
	#y1 = array((y.ix[:]))
	y1 = array(Y)
	y2 = group_data(y1, degree=2)
	y3 = group_data(y1, degree=3)
	y_all = np.hstack((y1,y2,y3))
	'''util.print_debug_msg('Pre processing Data')
	enc = OneHotEncoder()
	enc.fit(np.vstack((X,Y)))
	X = enc.transform(X)
	Y = enc.transform(Y)'''
	return x_all, y_all

def group_data(data, degree=3, hash=hash):
	util.print_debug_msg('Grouping Data')
	new_data = []
	m,n = data.shape
	print m,n
	for indicies in combinations(range(n), degree):
		new_data.append([hash(tuple(v)) for v in data[:,indicies]])
	return array(new_data).T

def train_predict(X_train, y_train, X_test):
	util.print_debug_msg('Training RandomForestRegressor Classifier')
	model = RandomForestRegressor(compute_importances = True)
	model.fit(X_train, y_train)
	print sorted(zip(list(range(93)),model.feature_importances_),  key=lambda tup: tup[1], reverse=True)
	util.print_debug_msg('Predicting RandomForestRegressor Classifier')
	return model.predict(X_test)

