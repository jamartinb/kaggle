import util
import logistic_regression as model
#import ensemble as model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import StratifiedKFold


# TODO(jamartin): Move this into main().
train = util.get_train_set()
#test = util.get_test_set()

print 'Data Loaded'

def separate_label_and_data():
	util.print_debug_msg('separating label and data')
	y = list(train['ACTION'])
	# creating a temp object of train, so as to preserve its originality.
	# i an not sure if we really need to do this.
	X = train
	del X['ACTION']
	return X,y



def validate_model():
	X,y = separate_label_and_data()
	X = model.pre_process_data(X)
	#print X.shape
	#model.select_features(X,y)
	result_sum = 0
	K = 10
	fold = 0
	skf = StratifiedKFold(y, n_folds = K)
	for train_index, test_index in skf:
		fold += 1
		X_train = []
		X_test = []
		Y_train = []
		Y_test = []
		for i in train_index:
			#X_train.append(list(X.ix[i]))
			X_train.append(X[i])
			Y_train.append(y[i])
		for i in test_index:
			#X_test.append(list(X.ix[i]))
			X_test.append(X[i])
			Y_test.append(y[i])
		
		print '========== Fold %d ==========' % (fold)
		X_train = model.normalize_features(X_train).toarray()
		X_test = model.normalize_features(X_test).toarray()
		#X_train, X_test = model.pre_process_data(X_train, X_test)
		pred = model.train_predict(X_train, Y_train, X_test)
		result = util.calculate_auc(Y_test, pred)
		result_sum += result
		print "AUC for fold %d = %0.11f" % (fold,result) 
	print "Average AUC for this classifier = %0.11f" % (result_sum/K)

def main():
	validate_model()

if __name__ == '__main__':
	main()
