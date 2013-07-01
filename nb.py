import util
from sklearn.naive_bayes import BernoulliNB

def pre_process_data(data):
	del data['MGR_ID']
	del data['ROLE_TITLE']
	del data['ROLE_FAMILY_DESC']
	del data['ROLE_FAMILY']
	del data['ROLE_CODE']
	return data

def train_predict(X_train, y_train, X_test):
	util.print_debug_msg('Training NB Classifier')
	nb = BernoulliNB()
	nb.fit(X_train, y_train)
	util.print_debug_msg('Predicting NB Classifier')
	return nb.predict(X_test)
