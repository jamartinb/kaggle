import util
import numpy as np
from sklearn import linear_model
from sklearn.preprocessing import OneHotEncoder
from sklearn.grid_search import GridSearchCV
from scipy import sparse
from itertools import combinations
from numpy import array, hstack
import GeneralHashFunctions as ghf

lg = linear_model.SGDClassifier(loss='log')
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
	
    # TODO(jamartin): Instead of commented code or alternative files you could
    #     have a main file where you can switch modes using FLAGS.
    # TODO(jamartin): Additionally, every module X.py should have a test module
    #     (X_test.py) with, at least, a testcase per method/function. These test
    #     modules are also a good alternative to keep track of previous attempts
    #     and configurations.
    #     See http://docs.python.org/2/library/unittest.html.
    #     See unittest.skip and unittest.skipIf may help to filter those test
    #     which are to heavy to execute all the time.

	#features = [0,8,9,10,36,37,38,41,41,43,47,53,60,61,63,64,67,69,71,75,85]
	#features = [0, 8, 9, 10, 19, 34, 36, 37, 38, 41, 42, 43, 47, 53, 55, 60, 61, 63, 64, 67, 69, 71, 75, 81, 82, 85]
	x_selected = x_all#[:,features]
	
	'''util.print_debug_msg('Selected Features. Now One Hot Encoding')
	enc = OneHotEncoder()
	#enc.fit(x_selected)
	#x_transformed = enc.transform(x_all)
	#enc = MinMaxScaler(feature_range=(0,1), copy=False)
	x_transformed = enc.fit_transform(X_selected)'''
	return x_selected

def normalize_features(X,y=None):
	util.print_debug_msg('Selected Features. Now One Hot Encoding')
	enc = OneHotEncoder()
    # TODO(jamartin): Spaces surounding '!='.
	if (y!=None):
		enc.fit(np.vstack((X,y)))
		X = enc.transform(X)
		y = enc.transform(y)
		return X,y
	return enc.fit_transform(X)
	
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

def grid_search(X_train, y_train):
	util.print_debug_msg('Starting grid search')
	parameters = {'C':[0.5,1,1.5,2,2.5,3,3.5,4],'penalty':['l1','l2']}
	clf = GridSearchCV(lg, parameters, scoring="roc_auc", n_jobs=2)
	util.print_debug_msg('Now fitting in grid search')
	clf.fit(X_train, y_train)
	print 'cv_scores: ', clf.cv_scores_
	print 'best_estimator: ', clf.best_estimator_
	print 'best_params: ', clf.best_params_
	
def train_predict(X_train, y_train, X_test):
	util.print_debug_msg('Training LG Classifier')
	lg.fit(X_train, y_train)
	util.print_debug_msg('Predicting LG Classifier')
	return lg.predict_proba(X_test)[:, 1]
	#return lg.predict(X_test)

