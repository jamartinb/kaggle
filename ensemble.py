import logging
import numpy as np
import util
from sklearn import ensemble
import itertools


log = logging.getLogger(__name__)


def pre_process_data(X, Y=None):
    #x1 = np.array((X.ix[:]))
    x1 = np.array(X)
    x2 = group_data(x1, degree=2)
    x3 = group_data(x1, degree=3)
    assert x1.shape[0] == x2.shape[0], (
        'x1.shape ({!r}) != x2.shape ({!r})'.format(x1.shape, x2.shape))
    assert x2.shape[0] == x3.shape[0], (
        'x2.shape ({!r}) != x3.shape ({!r})'.format(x2.shape, x3.shape))
    x_all = np.hstack((x1, x2, x3))
    
    if (Y):
      #y1 = np.array((y.ix[:]))
      y1 = np.array(Y)
      y2 = group_data(y1, degree=2)
      y3 = group_data(y1, degree=3)
      y_all = np.hstack((y1, y2, y3))
      return x_all, y_all
    else:
      return x_all


def group_data(data, degree=3, hash_function=hash):
    log.info('Grouping data (degree={})...'.format(degree))
    new_data = []
    m, n = data.shape
    log.debug('Shape = {}x{}.'.format(m, n))
    new_data = [[hash_function(tuple(v))
                 for v in data[:,indices]]
                for indices in itertools.combinations(range(n), degree)]
    return np.array(new_data).T


def train_predict(X_train, y_train, X_test):
    model = ensemble.GradientBoostingClassifier()
    log.info('Training a {}.'
             .format(model.__class__.__name__))
    model.fit(X_train, y_train)
    log.info('Predicting...')
    return model.predict(X_test)
