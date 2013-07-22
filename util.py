# TODO(jamartin): Files should have a general documentation header.
# TODO(jamartin): Be consistent in the use of either spaces or tabs but,
#     personally, I think spaces are a better choice.
import pandas as pd
from sklearn import metrics

def get_dataframe(file_name):
	df = pd.read_csv(file_name)
	del df['ROLE_CODE']
	return df

def get_train_set():
    # TODO(jamartin): Things like '../data/train.csv' should be defined as
    #     FLAGS. See Argparse (http://docs.python.org/2/library/argparse.html).
	return get_dataframe('../data/train.csv')

def get_test_set():
	test = get_dataframe('../data/test.csv')
	del test['id']
	return test

def calculate_auc(y, predictions):
	fpr, tpr, thresholds = metrics.roc_curve(y, predictions)
	return metrics.auc(fpr, tpr)
	
def print_debug_msg(msg, DEBUG=True):
    # @TODO(jamartin): This would be log.debug(...) if you used logging.
	if DEBUG:
		print msg

def load_pickle(file_name):
    # TODO(jamartin): These in-body imports are discouraged and should be
    #     avoided.
	import cPickle as pickle
	return pickle.load(open(file_name,'rb'))

def get_processed_train_set():
	print_debug_msg('Loading Train Set')
	return load_pickle('../data/train_processed.p')

def get_processed_test_set():
	print_debug_msg('Loading Test Set')
	return load_pickle('../data/test_processed.p')

def get_y():
	print_debug_msg('Loading y')
	return load_pickle('../data/y.p')

def get_all_data():
	X = get_processed_train_set()
	y = get_y()
	test = get_processed_test_set()
	return X, y, test
