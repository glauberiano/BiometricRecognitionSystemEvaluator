from abc import ABC, abstractmethod
from sklearn.svm import OneClassSVM
import pandas as pd
import numpy as np

class ClassificationAlgorithm(ABC):
 
    def __init__(self, name, parameters=dict()):
        self.name = name
        self.parameters = parameters
        super().__init__()
    
    @abstractmethod
    def train_user_model(self, data):
        pass
    
    @abstractmethod
    def test(self, data):
        pass

# Create classes for classification algorithms
        
class EuclidianClassifier(ClassificationAlgorithm):
    def train_user_model(self, user_features=None):
        try:
            user_features=user_features.drop('subject', axis=1)
        except:
            pass
        
        centroids = list()
        distance_train = list()
        
        len_features = user_features.shape[0]
        for feature in user_features:
            centroids.append(np.sum(user_features[feature] / len_features))
        for _, row in user_features.iterrows():
            distance_train.append(np.sqrt(sum((row-centroids)**2)))
        average_distance = np.sum(distance_train) / len_features
        user_model = EuclidianUserModel()
        user_model.update(model=average_distance, centroids=centroids)
        return user_model

    def test(self, genuine_user=None, test_stream=None, user_model=None, decision_threshold=0.00):
        #distance_test = list()
        #import pdb; pdb.set_trace();
        
        list_of_scores = list()
        y_genuino = list()
        y_impostor = list()
        y_true = test_stream.loc[:, 'subject']
        test_stream=test_stream.drop('subject', axis=1)

        for i, row in test_stream.iterrows():
            distance = np.sqrt(sum((row - user_model.centroids)**2))
            if (distance <= user_model.model):
                if (y_true[i] == genuine_user):
                    y_genuino.append(1)
                else:
                    y_impostor.append(1)
            else:
                if (y_true[i] == genuine_user):
                    y_genuino.append(0)
                else:
                    y_impostor.append(0)
            list_of_scores.append(distance)
        FMR, FNMR, B_acc = Metrics.report(y_genuino, y_impostor)
        return FMR, FNMR, B_acc, list_of_scores

class M2005(ClassificationAlgorithm):
    def train_user_model(self, user_features=None):
        user_model_object = M2005UserModel()
        usft = dict()
        try:
            user_features=user_features.drop('subject', axis=1)
        except:
            pass
        for feature in user_features:
            lower = min(user_features[feature].mean(), user_features[feature].median()) * (0.95 - (user_features[feature].std() / user_features[feature].mean()))
            upper = max(user_features[feature].mean(), user_features[feature].median()) * (1.05 + (user_features[feature].std() / user_features[feature].mean()))
            usft[feature] = (lower, upper)
        user_model_object.update(model=usft)
        return user_model_object

    def test(self, genuine_user=None, test_stream=None, user_model=None, decision_threshold=0.00):
        list_of_scores = list()
        y_genuino = list()
        y_impostor = list()
        y_true = test_stream.loc[:, 'subject']
        test_stream=test_stream.drop('subject', axis=1)

        for i, row in test_stream.iterrows():
            match_sum = 0
            previousDimMatched = False
            for dim in user_model.model.keys():
                if (row[dim] <= user_model.model[dim][1]) and (row[dim] >= user_model.model[dim][0]):
                    if previousDimMatched:
                        match_sum = match_sum + 1.5
                    else:
                        match_sum = match_sum + 1.0
                    previousDimMatched = True
                else:
                    previousDimMatched = False
            max_sum = 1.0 + 1.5 * (len(user_model.model.keys()) -1)
            score = match_sum/max_sum
            if (score > decision_threshold):
                if (y_true[i] == genuine_user):
                    y_genuino.append(1)
                else:
                    y_impostor.append(1)
            else:
                if (y_true[i] == genuine_user):
                    y_genuino.append(0)
                else:
                    y_impostor.append(0)
            list_of_scores.append(score)
        FMR, FNMR, B_acc = Metrics.report(y_genuino, y_impostor)
        return FMR, FNMR, B_acc, list_of_scores


# Create classes for template user models
class UserModel(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def update(self, model=None):
        pass

class EuclidianUserModel(UserModel):
    def update(self, model=None, centroids=None):
        if (model != None):
            self.model = model
        if (centroids != None):
            self.centroids = centroids

class M2005UserModel(UserModel):
    def update(self, model=None):
        self.model = model

# metrics class
class Metrics:
    def __init__(self):
        pass
        
    def report(self, y_genuino=None, y_impostor=None):
        FNMR = 1.0 - sum(y_genuino)/len(y_genuino)
        FMR = sum(y_impostor)/len(y_impostor)
        B_acc = 1.0 - (FNMR + FMR) / 2.0
        return FNMR, FMR, B_acc