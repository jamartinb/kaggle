import util
from sklearn.neighbors import KNeighborsClassifier

def pre_process_data(data):
	#del data['ROLE_CODE']
	return data

def train_predict(X_train, y_train, X_test):
	util.print_debug_msg('Training kNN Classifier')
	knn = KNeighborsClassifier(n_neighbors=10)
	knn.fit(X_train, y_train)
	util.print_debug_msg('Predicting kNN Classifier')
	return knn.predict_proba(X_test)[:, 1]

