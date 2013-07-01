import pandas as pd
from sklearn import metrics

DEBUG = True

def get_dataframe(file_name):
	df = pd.read_csv(file_name)
	del df['ROLE_CODE']
	return df

def get_train_set():
	return get_dataframe('../data/train.csv')

def get_test_set():
	return get_dataframe('../data/test.csv')

def calculate_auc(y, predictions):
	#print 'Y len:', len(y)
	#print 'P len:', len(predictions)
	#print y
	#print predictions
	fpr, tpr, thresholds = metrics.roc_curve(y, predictions)
	return metrics.auc(fpr, tpr)
	
def print_debug_msg(msg):
	if DEBUG:
		print msg

