from numpy import array, hstack
from sklearn import metrics, cross_validation, linear_model
from scipy import sparse
from itertools import combinations
from sklearn.cross_validation import StratifiedKFold,cross_val_score

import numpy as np
import pandas as pd
import util

#SEED = 25

def group_data(data, degree=3, hash=hash):
    new_data = []
    m,n = data.shape
    print m,n
    for indicies in combinations(range(n), degree):
        new_data.append([hash(tuple(v)) for v in data[:,indicies]])
    return array(new_data).T

def OneHotEncoder(data, keymap=None):
     """
     OneHotEncoder takes data matrix with categorical columns and
     converts it to a sparse binary matrix.
     
     Returns sparse binary matrix and keymap mapping categories to indicies.
     If a keymap is supplied on input it will be used instead of creating one
     and any categories appearing in the data that are not in the keymap are
     ignored
     """
     if keymap is None:
          keymap = []
          for col in data.T:
               uniques = set(list(col))
               keymap.append(dict((key, i) for i, key in enumerate(uniques)))
     total_pts = data.shape[0]
     outdat = []
     for i, col in enumerate(data.T):
          km = keymap[i]
          num_labels = len(km)
          spmat = sparse.lil_matrix((total_pts, num_labels))
          for j, val in enumerate(col):
               if val in km:
                    spmat[j, km[val]] = 1
          outdat.append(spmat)
     outdat = sparse.hstack(outdat).tocsr()
     return outdat, keymap

def create_test_submission(filename, prediction):
    content = ['id,ACTION']
    for i, p in enumerate(prediction):
        content.append('%i,%f' %(i+1,p))
    f = open(filename, 'w')
    f.write('\n'.join(content))
    f.close()
    print 'Saved'

def cv_loop(X, y, model, N):
    mean_auc = 0.
    for i in range(N):
        X_train, X_cv, y_train, y_cv = cross_validation.train_test_split(
                                       X, y, test_size=.30)
                                       #random_state = i*SEED)
        model.fit(X_train, y_train)
        preds = model.predict_proba(X_cv)[:,1]
        #preds = model.predict(X_cv)
        auc = metrics.auc_score(y_cv, preds)
        print "AUC (fold %d/%d): %f" % (i + 1, N, auc)
        mean_auc += auc
    return mean_auc/N

def main(train='train.csv', test='test.csv', submit='logistic_pred.csv'):    
	print "Reading dataset..."
	train_data = pd.read_csv(train)
	test_data = pd.read_csv(test)
	all_data = np.vstack((train_data.ix[:,1:-1], test_data.ix[:,1:-1]))
	
	num_train = np.shape(train_data)[0]
	
	# Transform data
	print "Transforming data..."
	dp = group_data(all_data, degree=2) 
	dt = group_data(all_data, degree=3)
	df = group_data(all_data, degree=4)
	
	y = array(train_data.ACTION)
	X = all_data[:num_train]
	X_2 = dp[:num_train]
	X_3 = dt[:num_train]
	X_4 = df[:num_train]
	
	X_test = all_data[num_train:]
	X_test_2 = dp[num_train:]
	X_test_3 = dt[num_train:]
	X_test_4 = df[num_train:]
	
	X_train_all = np.hstack((X, X_2, X_3, X_4))
	X_test_all = np.hstack((X_test, X_test_2, X_test_3, X_test_4))
	num_features = X_train_all.shape[1]
	
	model = linear_model.LogisticRegression()
	
	# Xts holds one hot encodings for each individual feature in memory
    # speeding up feature selection 
	Xts = [OneHotEncoder(X_train_all[:,[i]])[0] for i in range(num_features)]
	
	''' Performing univariate feature selection '''
	scores = []
	for f in range(len(Xts)):
		feature = [f]
		Xt = sparse.hstack([Xts[j] for j in feature]).tocsr()
		score = cv_loop(Xt, y, model, 3)
		scores.append(score)
		print "Feature: %i Mean AUC: %f" % (f, score)
	print sorted(zip(list(range(len(Xts))),scores),key=lambda tup: tup[1], reverse=True)

if __name__ == "__main__":
    args = { 'train':  '../data/train.csv',
             'test':   '../data/test.csv',
             'submit': 'logistic_regression_pred.csv' }
    main(**args)
    
