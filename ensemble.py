# TODO(jamartin) [SG] Imports should be sorted alphabetically to avoid
#     duplication and ease merges.
# TODO(jamartin) [SG] The preferred way to import is
#     'from X.Y.<package> import <module>'
import numpy as np
from numpy import array
import util
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from itertools import combinations

# TODO(jamartin) pre_process_data and group_data should be moved to a common
#     module called, for instance, data_handling.py. Maybe it could be part of
#     util.py but, whenever possible, modules should have descriptive names that
#     convey a quick idea of their contents.
# TODO(jamartin) All the models should be refactored into classes.
# TODO(jamartin) Chances are that these model/classes would benefit from a
#     parent "abstract" class that eases the storage and manipulation of local
#     attributes containing preprocessed data.

# TODO(jamartin) Functions and methods must be documented, this is even more
#     important in public ones.
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

# TODO(jamartin) [SG] There should be two empty lines between main-level
#     definitions such as classes and module functions.
def group_data(data, degree=3, hash=hash):
	util.print_debug_msg('Grouping Data')
	new_data = []
	m,n = data.shape
    # TODO(jamartin) Use logging.
    #     See http://docs.python.org/2/library/logging.html.
	print m,n
	for indicies in combinations(range(n), degree):
		new_data.append([hash(tuple(v)) for v in data[:,indicies]])
	return array(new_data).T

def train_predict(X_train, y_train, X_test):
    # TODO(jamartin) There is no need for this if you use logging.
	util.print_debug_msg('Training RandomForestRegressor Classifier')
    # TODO(jamartin) [style-guide (SG)] There should be no blanks between '=' in
    #     named arguments.
	model = RandomForestRegressor(compute_importances = True)
	model.fit(X_train, y_train)
    # TODO(jamartin) [SG] Single blanks after ',' and remove double spaces.
    # TODO(jmaratin) [SG] Lines should be 80 characters long (maximum).
	print sorted(zip(list(range(93)),model.feature_importances_),  key=lambda tup: tup[1], reverse=True)
	util.print_debug_msg('Predicting RandomForestRegressor Classifier')
	return model.predict(X_test)

