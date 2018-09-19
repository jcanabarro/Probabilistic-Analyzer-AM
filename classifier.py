import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

class BestEstimator:

    def __init__(self, x_train, x_validation, x_test, y_train, y_validation, y_test):
        self.x_train = x_train
        self.y_train = np.ravel(y_train)
        self.x_test = x_test
        self.y_test = np.ravel(y_test)
        self.x_validation = x_validation
        self.y_validation = np.ravel(y_validation)

    def train_classifier(self, classifier):
        return classifier.fit(self.x_train, self.y_train)

    def set_best_param(self, classifier, params):
        model = GridSearchCV(classifier, params)
        model = model.fit(self.x_validation, self.y_validation)
        return model.best_estimator_

    def get_classifier_score(self, classifier):
        pred = classifier.predict(self.x_test)
        return accuracy_score(self.y_test, pred, normalize=True)
