from abc import ABC, abstractmethod
from sklearn.svm import OneClassSVM
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import time

#####################################################################################################
#####################################################################################################

# Create classes for template user models
class UserModel(ABC):
    def __init__(self, features):
        self.features = features
        super().__init__()

    @abstractmethod
    def update(self, model=None):
        pass

class EuclidianUserModel(UserModel):
    def update(self, model):
        self.model = model

class ManhattanUserModel(UserModel):
    def update(self, model):
        self.model = model

class M2005UserModel(UserModel):
    def update(self, model):
        self.model = model

class OCSVMUserModel(UserModel):
    '''Aqui o modelo é um objeto do tipo OneClassSVM
    '''
    def __init__(self, features):
        self.features = features
        self.model = OneClassSVM()

    def update(self):
        self.model.fit(self.features)

class MahalanobisUserModel(UserModel):
    def __init__(self, features):
        self.features = features
        self.model = {'Mean': None, 'InvCov' : None}
    
    def update(self, model):
        self.model = model

class NeuralNetworkUserModel(UserModel):
    def update(self, model):
        self.model = model

#######################################################################################################
#######################################################################################################
        

class ClassificationAlgorithm(ABC):
    '''Classe "mãe" para os algoritmos de classificação utilizados.

    Parameters:
    \t name: nome do classificador.
    \t adaptive: tipo de estratégia de adaptação utilizada. Caso use alguma.
    \t normalize: se True os dados de treino e teste serão normalizados.
    \t parameters: ** Nao sei onde é utilizado **

    Methods:
    \t ** train_user_model **
    \t - Parameters:
    \t\t data: dados do usuário

    \t - Return: um objeto do tipo "userModel"

    \t ** test **
    \t - Parameters:
    \t\t data: dados do usuário

    \t - Return: (1) vetor binário de classificação das amostras de teste genuínas; 
    (2) vetor binário de classificação das amostras de teste impostoras;
    (3) objeto do tipo "userModel".

    

    '''
 
    def __init__(self, name, adaptive, normalize, parameters=dict()):
        self.name = name
        self.adaptive = adaptive # [None,'GrowingWindow','SlidingWindow','DoubleParallel']
        self.normalize = normalize
        self.parameters = parameters
        super().__init__()
    
    @abstractmethod
    def train_user_model(self, data):
        pass
    
    @abstractmethod
    def test(self, data):
        pass

#######################################################################################################

class ManhattanClassifier(ClassificationAlgorithm):
    def train_user_model(self, user_features):
        try:
            user_features=user_features.drop('subject', axis=1)
        except:
            pass
        
        if (self.normalize==True):
            self._normalize_params = dict()
            for col in user_features:
                self._normalize_params[col] = (np.mean(user_features[col]) , np.std(user_features[col]))
                user_features[col] = (user_features[col] - np.mean(user_features[col])) / np.std(user_features[col])

        model = user_features.mean()
        user_model = ManhattanUserModel(features=user_features)
        user_model.update(model=model)
        return user_model
    
    def test(self, genuine_user, test_stream, user_model, decision_threshold, validation=False):
        list_of_scores = list()
        y_genuino = list()
        y_impostor = list()
        y_true = test_stream.loc[:, 'subject']
        test_stream=test_stream.drop('subject', axis=1)

        if self.normalize==True:
            for col in test_stream.columns:
                test_stream[col] = (test_stream[col] - self._normalize_params[col][0]) / self._normalize_params[col][1]

        for i, row in test_stream.iterrows():
            score = np.sqrt(sum(abs(row - user_model.model)))
            if (score < decision_threshold[genuine_user]):
                if (y_true[i] == genuine_user):
                    y_genuino.append(1)
                    if self.adaptive != False: # ATUALIZAR O MODELO
                        ini = time.time()
                        AS = AdaptiveStrategy(trainFunction= ManhattanClassifier(name=self.name, adaptive=self.adaptive, normalize=self.normalize),
                        userModel= user_model, newData=row)
                        user_model = AS.run(strategy=self.adaptive)
                        print("adaptacao: {}".format((time.time() - ini)/60))

                else:
                    y_impostor.append(1)
            else:
                if (y_true[i] == genuine_user):
                    y_genuino.append(0)
                else:
                    y_impostor.append(0)
            list_of_scores.append(score)
        return y_genuino, y_impostor, user_model

class EuclidianClassifier(ClassificationAlgorithm):
    def train_user_model(self, user_features):
        try:
            user_features=user_features.drop('subject', axis=1)
        except:
            pass
        
        if (self.normalize==True):
            self._normalize_params = dict()
            for col in user_features:
                self._normalize_params[col] = (np.mean(user_features[col]) , np.std(user_features[col]))
                user_features[col] = (user_features[col] - np.mean(user_features[col])) / np.std(user_features[col])

        model = user_features.mean()
        user_model = EuclidianUserModel(features=user_features)
        user_model.update(model=model)
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


class MahalanobisClassifier(ClassificationAlgorithm):
    def train_user_model(self, user_features=None, normalize=False):
        try:
            user_features=user_features.drop('subject', axis=1)
        except:
            pass
        
        if (self.normalize==True):
            self._normalize_params = dict()
            for col in user_features:
                self._normalize_params[col] = (np.mean(user_features[col]) , np.std(user_features[col]))
                user_features[col] = (user_features[col] - np.mean(user_features[col])) / np.std(user_features[col])

        user_model = EuclidianUserModel(features=user_features)
        mean = user_features.mean().values
        inverse_cov_matriz = np.linalg.inv(np.cov(user_features, rowvar=False))
        model = {'Mean': mean, 'InvCov' : inverse_cov_matriz}
        user_model.update(model=model)
        return user_model

    def test(self, genuine_user=None, test_stream=None, user_model=None, decision_threshold=0.00, normalize=False):
        list_of_scores = list()
        y_genuino = list()
        y_impostor = list()
        y_true = test_stream.loc[:, 'subject']
        test_stream=test_stream.drop('subject', axis=1)

        if self.normalize==True:
            for col in test_stream.columns:
                test_stream[col] = (test_stream[col] - self._normalize_params[col][0]) / self._normalize_params[col][1]

        for i, row in test_stream.iterrows():
            score = np.sqrt(np.dot(np.dot(row-user_model.model['Mean'], user_model.model['InvCov']), row-user_model.model['Mean'])**2)
            if (score < decision_threshold[genuine_user]):
                if (y_true[i] == genuine_user):
                    y_genuino.append(1)
                    if self.adaptive != False: # ATUALIZAR O MODELO
                        AS = AdaptiveStrategy(trainFunction= MahalanobisClassifier(name=self.name, adaptive=self.adaptive, normalize=self.normalize),
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
        y_true = test_stream.loc[:, 'subject'].tolist()
        test_stream=test_stream.drop('subject', axis=1)
        max_sum = 1.0 + 1.5 * (len(user_model.model.keys()) -1)
            
        if (normalize==True):
            for col in test_stream.columns:
                test_stream[col] = (test_stream[col] - self._normalize_params[col][0]) / self._normalize_params[col][1]
        
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

class OCSVMClassifier(ClassificationAlgorithm):
    def train_user_model(self,  user_features=None, normalize=False):
        try:
            user_features=user_features.drop('subject', axis=1)
        except:
            pass
        
        if (normalize==True):
            self._normalize_params = dict()
            for col in user_features:
                self._normalize_params[col] = (np.mean(user_features[col]) , np.std(user_features[col]))
                user_features[col] = (user_features[col] - np.mean(user_features[col])) / np.std(user_features[col])

        user_model_object = OCSVMUserModel(features=user_features)
        user_model_object.update()
        return user_model_object

    def test(self, genuine_user=None, test_stream=None, user_model=None, decision_threshold=None, validation=False, normalize=False):
        list_of_scores = list()
        y_genuino = list()
        y_impostor = list()
        y_true = test_stream.loc[:, 'subject']
        test_stream=test_stream.drop('subject', axis=1)

        if normalize==True:
            for col in test_stream.columns:
                test_stream[col] = (test_stream[col] - self._normalize_params[col][0]) / self._normalize_params[col][1]

        for i, row in test_stream.iterrows():
            score = user_model.model.decision_function(row.values.reshape(1,-1))
            if (score < decision_threshold[genuine_user]):
                if (y_true[i] == genuine_user):
                    y_genuino.append(1)
                    if self.adaptive != False: # ATUALIZAR O MODELO
                        AS = AdaptiveStrategy(trainFunction= OCSVMClassifier(name=self.name, adaptive=self.adaptive, normalize=self.normalize),
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

class RandForestClassifier(ClassificationAlgorithm):
    def train_user_model(self, user_features=None, normalize=False):
        try:
            user_features=user_features.drop('subject', axis=1)
        except:
            pass

        #user_model_object = OCSVMUserModel(features=user_features)
        user_model_object = RandomForestClassifier()
        user_model_object.update()
        return user_model_object

    def test(self):
        pass

class NeuralNetworkClassifier(ClassificationAlgorithm):
    def train_user_model(self):
        pass

    def test(self):
        pass
#######################################################################################################
#######################################################################################################

class AdaptiveStrategy:
    ''' Métodos adaptativos.

    Parameters:
    \t trainFunction: uma função de treinamento.
    \t userModel: o modelo do usuário mais atual.
    \t newData: dado utilizado para atualizar o modelo do usuário.

    Return:
    \t Modelo do usuário atualizado.

    '''
    def __init__(self, trainFunction, userModel, newData):
        self.trainFunction = trainFunction
        self.userModel = userModel
        self.newData = newData
        
    def run(self, strategy):
        #eval('self._' + strategy)()
        if strategy == 'GrowingWindow':
            self._GrowingWindow()
        elif strategy == 'SlidingWindow':
            self._SlidingWindow()
        elif strategy == '_DoubleParallel':
            raise Exception("Nao esta pronto ainda")
        else:
            raise Exception("Escolha uma estratégia de adaptação válida!")
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

#######################################################################################################
#######################################################################################################


# metrics class
class Metrics:
    @staticmethod
    def report(y_genuine, y_impostor):
        FNMR = 1.0 - sum(y_genuine)/len(y_genuine)
        FMR = sum(y_impostor)/len(y_impostor)
        B_acc = 1.0 - (FNMR + FMR) / 2.0
        return FMR, FNMR, B_acc