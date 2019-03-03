#Automatic Biometric Recognition System

'''
Classe principal:
    * Em um sistema de reconhecimento padrão, o sistema cria um modelo para todos os usuários
'''

from components import Cadastramento
from components import DBCreator
from components import DataStream
from classifiers import M2005
from classifiers import EuclidianClassifier
from classifiers import Metrics
import pandas as pd
import numpy as np
import IPython.core.display as ipd
from sklearn.model_selection import KFold
from sklearn.model_selection import ParameterGrid

class RecognitionSystem():
    def __init__(self, method, adaptive=False):
        self.adaptive = adaptive
        if (method == 'M2005'):
            self.classifier = M2005(name=method)
        elif (method == 'euclidian'):
            self.classifier = EuclidianClassifier(name=method)
            
        #super().__init__()
        
    def __validation(self, method='users_kfold', n_splits=5, list_users=None):
        if method=='users_kfold':
            kfold = KFold(n_splits=5)
            return kfold.split(list_users)
        
    def __runNotAdaptive(self, dataset=None, model_size=None, impostor_rate=None, rate_external_impostor=None,
                         decision_threshold=None):
        
        list_of_scores = {'FMR' : list(),
                          'FNMR' : list(),
                          'B_acc' :  list()}
        
        
        ds = DataStream(impostor_rate=impostor_rate, rate_external_impostor=rate_external_impostor)
        userInformations = Cadastramento()
        kfold = KFold(n_splits=5)
        splits = kfold.split(self._usuarios)
        
        counter_folds = 0
        for internal_idx, external_idx in splits:
            u_reg = self._usuarios.iloc[internal_idx][self._user_column]
            u_nao_reg = self._usuarios.iloc[external_idx][self._user_column]
            usersDatabase, samples = userInformations.create(dataset=dataset, 
                                            n_amostras=model_size, users_list=u_reg)
            counter=0
            counter_folds += 1
            
            for usuario in u_reg:
                counter += 1
                ipd.clear_output(wait=True)
                print('Split ' +str(counter_folds) + ' de ' + str(kfold.n_splits) + '\n' +   'Usuário ' + str(counter) + ' de ' + str(len(u_reg)))
                #import pdb; pdb.set_trace();
                
                dataStream = ds.create(data=samples, genuine=usuario, internal=u_reg, external=u_nao_reg)
                userBiometricModel = self.classifier.train_user_model(user_features=usersDatabase[usuario])
                FMR, FNMR, B_acc, _ = self.classifier.test(genuine_user=usuario, test_stream=dataStream, 
                                                           user_model=userBiometricModel, decision_threshold=decision_threshold)
                list_of_scores['FMR'].append(FMR)
                list_of_scores['FNMR'].append(FNMR)
                list_of_scores['B_acc'].append(B_acc)
        return list_of_scores['FMR'], list_of_scores['FNMR'], list_of_scores['B_acc']
    
    def fit(self, dataset=None, user_column='subject', metric=None, param_grid=None):
        self._user_column = user_column
        self._usuarios = pd.DataFrame(dataset[self._user_column].unique(), columns=[user_column])
        self.list_params = list()
        self.list_scores = list()
        p = ParameterGrid(param_grid)
        
        for params in p:
            self.list_params.append(params)
            if (self.adaptive == False):
                FMR, FNMR, B_acc = self.__runNotAdaptive(dataset=dataset, **params)
            self.list_scores.append([FMR, FNMR, B_acc])

    def summary(self, parameters = None, metrics = None):
        #import pdb; pdb.set_trace();
        list_scores=list()
        for i in range(len(metrics)):
            scores = {'FMR_mean' : np.mean(metrics[i][0]),
                      'FMR_std' : np.std(metrics[i][0]),
                      'FNMR_mean' : np.mean(metrics[i][1]),
                      'FNMR_std' : np.std(metrics[i][1]),
                      'Bacc_mean' :  np.mean(metrics[i][2]),
                      'Bacc_std' : np.std(metrics[i][2])}
            list_scores.append(scores)
        df1 = pd.DataFrame(parameters)
        df2 = pd.DataFrame(list_scores)
        frames = [df1, df2]
        return pd.concat(frames, axis=1)
    