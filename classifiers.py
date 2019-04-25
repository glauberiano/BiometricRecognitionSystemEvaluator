from abc import ABC, abstractmethod
from sklearn.svm import OneClassSVM
from sklearn.base import BaseEstimator
import pandas as pd
import numpy as np

class ClassificationAlgorithm(ABC):
 
    def __init__(self, name, adaptive, normalize, parameters=dict()):
        self.name = name
        self.adaptive = adaptive # [None,'GW','SW','DP']
        self.normalize = normalize
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
        
        if (self.normalize==True):
            self._normalize_params = dict()
            for col in user_features:
                self._normalize_params[col] = (np.mean(user_features[col]) , np.std(user_features[col]))
                user_features[col] = (user_features[col] - np.mean(user_features[col])) / np.std(user_features[col])

        #len_features = user_features.shape[0]
        #for feature in user_features:
        #    centroids.append(np.sum(user_features[feature] / len_features))
        #for _, row in user_features.iterrows():
        #    distance_train.append(np.sqrt(sum((row-centroids)**2)))
        #average_distance = np.mean(distance_train)
        #std_distance = np.std(distance_train)
        
        model = user_features.mean()
        
        user_model = EuclidianUserModel(features=user_features)
        user_model.init(model=model)
        return user_model

    def test(self, genuine_user=None, test_stream=None, user_model=None, decision_threshold=None, validation=False):
        list_of_scores = list()
        y_genuino = list()
        y_impostor = list()
        y_true = test_stream.loc[:, 'subject']
        test_stream=test_stream.drop('subject', axis=1)

        if self.normalize==True:
            for col in test_stream.columns:
                test_stream[col] = (test_stream[col] - self._normalize_params[col][0]) / self._normalize_params[col][1]

        for i, row in test_stream.iterrows():
            score = np.sqrt(sum((row - user_model.model)**2))
            if (score < decision_threshold[genuine_user]):
                if (y_true[i] == genuine_user):
                    y_genuino.append(1)
                    if self.adaptive != False: # ATUALIZAR O MODELO
                        AS = AdaptiveStrategy(trainFunction= EuclidianClassifier(name=self.name, adaptive=self.adaptive, normalize=self.normalize),
                        userModel= user_model, newData=row)
                        user_model = AS.run(strategy=self.adaptive)

                else:
                    y_impostor.append(1)
            else:
                if (y_true[i] == genuine_user):
                    y_genuino.append(0)
                else:
                    y_impostor.append(0)
            list_of_scores.append(score)
        return y_genuino, y_impostor, user_model

class M2005Classifier(ClassificationAlgorithm):
    def train_user_model(self, user_features=None, normalize=False):
        user_model_object = M2005UserModel(features=user_features)
        usft = dict()
        try:
            user_features=user_features.drop('subject', axis=1)
        except:
            pass
        
        if (normalize==True):
            self._normalize_params = dict()
            for col in user_features:
                self._normalize_params[col] = (np.mean(user_features[col]) , np.std(user_features[col]))
                user_features[col] = (user_features[col] - np.mean(user_features[col])) / np.std(user_features[col])
        
        for feature in user_features:
            lower = min(user_features[feature].mean(), user_features[feature].median()) * (0.95 - (user_features[feature].std() / user_features[feature].mean()))
            upper = max(user_features[feature].mean(), user_features[feature].median()) * (1.05 + (user_features[feature].std() / user_features[feature].mean()))
            usft[feature] = (lower, upper)
        user_model_object.update(model=usft)
        return user_model_object

    def test(self, genuine_user=None, test_stream=None, user_model=None, decision_threshold=0.00, normalize=False):
        list_of_scores = list()
        y_genuino = list()
        y_impostor = list()
        y_true = test_stream.loc[:, 'subject']
        test_stream=test_stream.drop('subject', axis=1)

        if (normalize==True):
            for col in test_stream.columns:
                test_stream[col] = (test_stream[col] - self._normalize_params[col][0]) / self._normalize_params[col][1]

        #import pdb;pdb.set_trace();
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
            #import pdb;pdb.set_trace();
            max_sum = 1.0 + 1.5 * (len(user_model.model.keys()) -1)
            score = match_sum/max_sum

            
            if (score > decision_threshold[genuine_user]):
                if (y_true[i] == genuine_user):
                    y_genuino.append(1)
                    if self.adaptive != False: # ATUALIZAR O MODELO
                        AS = AdaptiveStrategy(trainFunction= M2005Classifier(name=self.name, adaptive=self.adaptive, normalize=self.normalize),
                                            userModel= user_model, newData=row)
                        user_model = AS.run(strategy=self.adaptive)
                else:
                    y_impostor.append(1)
            else:
                if (y_true[i] == genuine_user):
                    y_genuino.append(0)
                else:
                    y_impostor.append(0)
            list_of_scores.append(score)
        return y_genuino, y_impostor, user_model

# Create classes for template user models
class UserModel(ABC):
    def __init__(self, features):
        self.features = features
        super().__init__()

    @abstractmethod
    def update(self, model=None):
        pass

class EuclidianUserModel(UserModel):
    def init(self, model):
        self.model = model

    def update(self):
        pass
        
class M2005UserModel(UserModel):
    def update(self, model):
        self.model = model

class AdaptiveStrategy:
    def __init__(self, trainFunction, userModel, newData):
        self.trainFunction = trainFunction
        self.userModel = userModel
        self.newData = newData
        
    def run(self, strategy):
        eval('self._' + strategy)()
        return self.new_model

    def _GrowingWindow(self):
        #import pdb; pdb.set_trace()
        self.userModel.features = self.userModel.features.append(self.newData, ignore_index=True)
        self.new_model = self.trainFunction.train_user_model(self.userModel.features)

    def _SlidingWindow(self):
        self.userModel.features = self.userModel.features.iloc[1:]
        self.userModel.features = self.userModel.features.append(self.newData, ignore_index=True)
        self.new_model = self.trainFunction.train_user_model(self.userModel.features)

    def _DoubleParallel(self):
        pass


# metrics class
class Metrics:
    @staticmethod
    def report(y_genuine, y_impostor):
        FNMR = 1.0 - sum(y_genuine)/len(y_genuine)
        FMR = sum(y_impostor)/len(y_impostor)
        B_acc = 1.0 - (FNMR + FMR) / 2.0
        return FMR, FNMR, B_acc