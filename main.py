import logging
import util
import ensemble as model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import StratifiedKFold

log = logging.getLogger(__name__)


def load_data():
    result = util.get_train_set()
    log.info('Data loaded.')
    return result


def separate_label_and_data(train):
    log.debug('Separating label and data...')
    y = list(train['ACTION'])
    del train['ACTION']
    return train, y


def validate_model():
    train = load_data()
    X, y = separate_label_and_data(train)
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
        
        log.info('========== Fold {} =========='.format(fold))
        pred = model.train_predict(X_train, Y_train, X_test)
        result = util.calculate_auc(Y_test, pred)
        result_sum += result
        log.info('AUC for fold {} = {:0.11f}'.format(fold,result))
    log.info('Average AUC for this classifier = {:0.11f}'.format(result_sum/K))


def main():
    logging.basicConfig(level=logging.DEBUG)
    validate_model()


if __name__ == '__main__':
    main()
