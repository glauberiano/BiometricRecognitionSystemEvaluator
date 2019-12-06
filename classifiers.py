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

class StatisticUserModel(UserModel):
    def update(self, model):
        self.model = model

#class NeuralNetworkUserModel(UserModel):
#    def update(self, model):
#        self.model = model

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
class StatisticClassifier(ClassificationAlgorithm):
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

        #import pdb; pdb.set_trace();
        media = user_features.mean()
        desvpad = user_features.std()
        model = {'Mean': media, 'Desvpad': desvpad}
        user_model = StatisticUserModel(features=user_features)
        user_model.update(model=model)
        return user_model

    def test(self, genuine_user=None, test_stream=None, user_model=None, decision_threshold=None, normalize=False):
        y_genuine = list()
        y_impostor = list()
        model_score = list()
        y_true = test_stream.loc[:, 'subject'].tolist()
        test_stream=test_stream.drop('subject', axis=1)
            
        if (normalize==True):
            for col in test_stream.columns:
                test_stream[col] = (test_stream[col] - self._normalize_params[col][0]) / self._normalize_params[col][1]
        
        #y_pred = list()
        def help_score(features, user_model, adaptive=self.adaptive):
            return 1 - sum(np.e ** ( ( abs(features- user_model.model['Mean']) / user_model.model['Desvpad'])*-1 )) / len(features)
        
        ## Escolhendo linha de decisao
        way = 0
        if way==1:
            decision = decision_threshold[genuine_user]
        else:
            decision = np.mean([*decision_threshold.values()])

        if self.adaptive == False:
            model_score = test_stream.apply(lambda x: help_score(x, user_model), axis=1) 
            y_genuine = [1 if model_score[i] < decision else 0 for i, sample in enumerate(y_true) if sample == genuine_user]
            y_impostor = [1 if model_score[i] < decision else 0 for i, sample in enumerate(y_true) if sample != genuine_user]
        else:
            for i, features in test_stream.iterrows():
                score = 1 - sum(np.e ** ( ( abs(features- user_model.model['Mean']) / user_model.model['Desvpad'])*-1 )) / len(features)
                if score < decision:
                    AS = AdaptiveStrategy(trainFunction= StatisticClassifier(name=self.name, adaptive=self.adaptive, normalize=self.normalize),
                                          userModel= user_model, newData=features)
                    user_model = AS.run(strategy=self.adaptive)
                    if y_true[i]==genuine_user:
                        y_genuine.append(1)
                    else:
                        y_impostor.append(1)
                else:
                    if y_true[i]==genuine_user:
                        y_genuine.append(0)
                    else:
                        y_impostor.append(0)
                model_score.append(score)
        return y_genuine, y_impostor, user_model, decision, model_score    

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
        model_score = list()
        y_genuine = list()
        y_impostor = list()
        y_true = test_stream.loc[:, 'subject']
        test_stream=test_stream.drop('subject', axis=1)

        if self.normalize==True:
            for col in test_stream.columns:
                test_stream[col] = (test_stream[col] - self._normalize_params[col][0]) / self._normalize_params[col][1]

        ## Escolhendo linha de decisao
        way = 0
        if way==1:
            decision = decision_threshold[genuine_user]
        else:
            decision = np.mean([*decision_threshold.values()])

        for i, row in test_stream.iterrows():
            score = np.sqrt(sum(abs(row - user_model.model)))
            if (score < decision_threshold[genuine_user]):
                if (y_true[i] == genuine_user):
                    y_genuine.append(1)
                    if self.adaptive != False: # ATUALIZAR O MODELO
                        AS = AdaptiveStrategy(trainFunction= ManhattanClassifier(name=self.name, adaptive=self.adaptive, normalize=self.normalize),
                        userModel= user_model, newData=row)
                        user_model = AS.run(strategy=self.adaptive)
                else:
                    y_impostor.append(1)
            else:
                if (y_true[i] == genuine_user):
                    y_genuine.append(0)
                else:
                    y_impostor.append(0)
            model_score.append(score)
        return y_genuine, y_impostor, user_model, decision, model_score

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
        model_score = list()
        y_genuine = list()
        y_impostor = list()
        y_true = test_stream.loc[:, 'subject']
        test_stream=test_stream.drop('subject', axis=1)

        if self.normalize==True:
            for col in test_stream.columns:
                test_stream[col] = (test_stream[col] - self._normalize_params[col][0]) / self._normalize_params[col][1]

        # limiar de decisão
        way = 0
        if way==1:
            decision = decision_threshold[genuine_user]
        else:
            decision = np.mean([*decision_threshold.values()])

        def help_score(features, user_model, adaptive=self.adaptive):
            score = score = np.sqrt(sum((features - user_model.model)**2))
            return score

        if self.adaptive == False:
            model_score = test_stream.apply(lambda x: help_score(x, user_model), axis=1) 
            y_genuine = [1 if model_score[i] < decision else 0 for i, sample in enumerate(y_true) if sample == genuine_user]
            y_impostor = [1 if model_score[i] < decision else 0 for i, sample in enumerate(y_true) if sample != genuine_user]
        else:
            for i, row in test_stream.iterrows():
                score = np.sqrt(sum((row - user_model.model)**2))
                if (score < decision_threshold[genuine_user]):
                    if (y_true[i] == genuine_user):
                        y_genuine.append(1)
                        if self.adaptive != False: # ATUALIZAR O MODELO
                            AS = AdaptiveStrategy(trainFunction= EuclidianClassifier(name=self.name, adaptive=self.adaptive, normalize=self.normalize),
                            userModel= user_model, newData=row)
                            user_model = AS.run(strategy=self.adaptive)

                    else:
                        y_impostor.append(1)
                else:
                    if (y_true[i] == genuine_user):
                        y_genuine.append(0)
                    else:
                        y_impostor.append(0)
                model_score.append(score)
        return y_genuine, y_impostor, user_model, decision, model_score   


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
        model_score = list()
        y_genuino = list()
        y_impostor = list()
        y_true = test_stream.loc[:, 'subject']
        test_stream=test_stream.drop('subject', axis=1)

        if self.normalize==True:
            for col in test_stream.columns:
                test_stream[col] = (test_stream[col] - self._normalize_params[col][0]) / self._normalize_params[col][1]

        ## Escolhendo linha de decisao
        way = 0
        if way==1:
            decision = decision_threshold[genuine_user]
        else:
            decision = np.mean([*decision_threshold.values()])
        
        def help_score(features, user_model, adaptive=self.adaptive):
            score = np.sqrt(np.dot(np.dot(features-user_model.model['Mean'], user_model.model['InvCov']), features-user_model.model['Mean'])**2)
            return score

        if self.adaptive == False:
            model_score = test_stream.apply(lambda x: help_score(x, user_model), axis=1) 
            y_genuine = [1 if model_score[i] < decision else 0 for i, sample in enumerate(y_true) if sample == genuine_user]
            y_impostor = [1 if model_score[i] < decision else 0 for i, sample in enumerate(y_true) if sample != genuine_user]
        else:
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
                model_score.append(score)
        return y_genuine, y_impostor, user_model, decision, model_score

class M2005Classifier(ClassificationAlgorithm):
    def train_user_model(self, user_features=None, normalize=False):
        user_model_object = M2005UserModel(features=user_features)
        usft = dict()
        temp = user_features['subject']
        try:
            user_features=user_features.drop('subject', axis=1)
        except:
            pass
        
        if (normalize==True):
            self._normalize_params = dict()
            for col in user_features:
                self._normalize_params[col] = (np.mean(user_features[col]) , np.std(user_features[col]))
                user_features[col] = (user_features[col] - np.mean(user_features[col])) / np.std(user_features[col])
        
        #import pdb; pdb.set_trace()
        try:
            for feature in user_features:
                if user_features[feature].mean() == 0:
                    lower = min(user_features[feature].mean(), user_features[feature].median()) * (0.95 - (user_features[feature].std() / 0.00001))
                    upper = max(user_features[feature].mean(), user_features[feature].median()) * (1.05 + (user_features[feature].std() / 0.00001))
                else:
                    lower = min(user_features[feature].mean(), user_features[feature].median()) * (0.95 - (user_features[feature].std() / user_features[feature].mean()))
                    upper = max(user_features[feature].mean(), user_features[feature].median()) * (1.05 + (user_features[feature].std() / user_features[feature].mean()))
                usft[feature] = (lower, upper)
        except:
            pass
            #import pdb; pdb.set_trace()    
        user_model_object.update(model=usft)
        return user_model_object

    def test(self, genuine_user=None, test_stream=None, user_model=None, decision_threshold=None, normalize=False):
        y_genuine = list()
        y_impostor = list()
        model_score = list()
        y_true = test_stream.loc[:, 'subject'].tolist()
        test_stream=test_stream.drop('subject', axis=1)
        max_sum = 1.0 + 1.5 * (len(user_model.model.keys()) -1)
            
        if (normalize==True):
            for col in test_stream.columns:
                test_stream[col] = (test_stream[col] - self._normalize_params[col][0]) / self._normalize_params[col][1]
        
        #y_pred = list()
        def help_score(features, user_model, adaptive=self.adaptive):
            match_sum = 0
            previousDimMatched = False
            #import pdb;pdb.set_trace()
            for dim in user_model.model.keys():
                if (features[dim] <= user_model.model[dim][1]) and (features[dim] >= user_model.model[dim][0]):
                    if previousDimMatched:
                        match_sum = match_sum + 1.5
                    else:
                        match_sum = match_sum + 1.0
                    previousDimMatched = True
                else:
                    previousDimMatched = False
            score = match_sum/max_sum
            return score 
        
        ## Escolhendo linha de decisao
        way = 0
        if way==1:
            decision = decision_threshold[genuine_user]
        else:
            decision = np.mean([*decision_threshold.values()])

        if self.adaptive == False:
            model_score = test_stream.apply(lambda x: help_score(x, user_model), axis=1) 
            y_genuine = [1 if model_score[i] > decision else 0 for i, sample in enumerate(y_true) if sample == genuine_user]
            y_impostor = [1 if model_score[i] > decision else 0 for i, sample in enumerate(y_true) if sample != genuine_user]
        else:
            for i, features in test_stream.iterrows():
                match_sum = 0
                previousDimMatched = False
                for dim in user_model.model.keys():
                    if (features[dim] <= user_model.model[dim][1]) and (features[dim] >= user_model.model[dim][0]):
                        if previousDimMatched:
                            match_sum = match_sum + 1.5
                        else:
                            match_sum = match_sum + 1.0
                        previousDimMatched = True
                    else:
                        previousDimMatched = False
                score = match_sum/max_sum
                if score > decision:
                    AS = AdaptiveStrategy(trainFunction= M2005Classifier(name=self.name, adaptive=self.adaptive, normalize=self.normalize),
                                          userModel= user_model, newData=features)
                    user_model = AS.run(strategy=self.adaptive)
                    if y_true[i]==genuine_user:
                        y_genuine.append(1)
                    else:
                        y_impostor.append(1)
                else:
                    if y_true[i]==genuine_user:
                        y_genuine.append(0)
                    else:
                        y_impostor.append(0)
                model_score.append(score)
        return y_genuine, y_impostor, user_model, decision, model_score

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
        y_genuine = list()
        y_impostor = list()
        model_score = list()
        y_true = test_stream.loc[:, 'subject']
        test_stream=test_stream.drop('subject', axis=1)

        if normalize==True:
            for col in test_stream.columns:
                test_stream[col] = (test_stream[col] - self._normalize_params[col][0]) / self._normalize_params[col][1]
        
        def help_score(features, user_model, adaptive=self.adaptive):
            #import pdb; pdb.set_trace()
            score = user_model.model.decision_function(features.values.reshape(1,-1))
            return score.item()
            
        ## Escolhendo linha de decisao
        way = 0
        if way==1:
            decision = decision_threshold[genuine_user]
        else:
            decision = np.mean([*decision_threshold.values()])

        #import pdb; pdb.set_trace()
        if self.adaptive == False:
            model_score = test_stream.apply(lambda x: help_score(x, user_model), axis=1) 
            y_genuine = [1 if model_score[i] > decision else 0 for i, sample in enumerate(y_true) if sample == genuine_user]
            y_impostor = [1 if model_score[i] > decision else 0 for i, sample in enumerate(y_true) if sample != genuine_user]
        else:
            for i, row in test_stream.iterrows():
                score = user_model.model.decision_function(row.values.reshape(1,-1))
                if (score > decision_threshold[genuine_user]):
                    if (y_true[i] == genuine_user):
                        y_genuine.append(1)
                        if self.adaptive != False: # ATUALIZAR O MODELO
                            AS = AdaptiveStrategy(trainFunction= OCSVMClassifier(name=self.name, adaptive=self.adaptive, normalize=self.normalize),
                            userModel= user_model, newData=row)
                            user_model = AS.run(strategy=self.adaptive)

                    else:
                        y_impostor.append(1)
                else:
                    if (y_true[i] == genuine_user):
                        y_genuine.append(0)
                    else:
                        y_impostor.append(0)
                model_score.append(score)
        return y_genuine, y_impostor, user_model, decision, model_score

class RandForestClassifier(ClassificationAlgorithm):
    def train_user_model(self, user_features=None, normalize=False):
        try:
            user_features=user_features.drop('subject', axis=1)
        except:
            pass

        #user_model_object = OCSVMUserModel(features=user_features)
        user_model_object = RandomForestClassifier()
        #user_model_object.update()
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
        #import pdb;pdb.set_trace()
        FNMR = 1.0 - sum(y_genuine)/len(y_genuine)
        FMR = sum(y_impostor)/len(y_impostor)
        B_acc = 1.0 - (FNMR + FMR) / 2.0
        return FMR, FNMR, B_acc