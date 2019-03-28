#Automatic Biometric Recognition System

'''
Classe principal:
    * Em um sistema de reconhecimento padrão, o sistema cria um modelo para todos os usuários
'''

from components import Enrollment
from components import Random, GenFirst, ImpFirst, SeriesAttack
from classifiers import M2005Classifier
from classifiers import EuclidianClassifier
from classifiers import Metrics
import pandas as pd
import numpy as np
import IPython.core.display as ipd
from sklearn.model_selection import KFold
from sklearn.model_selection import ParameterGrid

class RecognitionSystem():
    def __init__(self, system_type='notAdaptive'):
        #assert method in ['M2005', 'Euclidian'], 'method must be on of \'M2005\', \'Euclidian\''
        #self.classifier = eval(method+'Classifier')(name=method)
        assert system_type in ['Adaptive', 'notAdaptive'], 'method must be on of \'Adaptive\', \'notAdaptive\''
        self.system_type = '_'+system_type
        
    def __validation(self, method='users_kfold', n_splits=5, list_users=None):
        if method=='users_kfold':
            kfold = KFold(n_splits=5)
            return kfold.split(list_users)
    
    def _notAdaptive(self, dataset=None, method=None, model_size=None, impostor_rate=0.5, rate_external_impostor=0,
                         decision_threshold=None, sampling='Random', n_series=None, normalize=False):
        
        assert method in ['M2005', 'Euclidian'], 'method must be on of \'M2005\', \'Euclidian\''
        self.classifier = eval(method+'Classifier')(name=method)
        
        list_of_scores = {'FMR' : list(),
                          'FNMR' : list(),
                          'B_acc' :  list()}
        
        data_stream = eval(sampling)(impostor_rate=impostor_rate, rate_external_impostor=rate_external_impostor, n_series=n_series)
        #ds = DataStream(impostor_rate=impostor_rate, rate_external_impostor=rate_external_impostor)
        splitTrainTest = Enrollment()
        kfold = KFold(n_splits=5)
        splits = kfold.split(self._usuarios)
        
        counter_folds = 0
        for internal_idx, external_idx in splits:
            u_reg = self._usuarios[internal_idx]
            u_nao_reg = self._usuarios[external_idx]
            usersDatabase, samples = splitTrainTest.create(dataset=dataset, n_amostras=model_size, users_list=u_reg)
            counter=0
            counter_folds += 1
            
            for usuario in u_reg:
                counter += 1
                ipd.clear_output(wait=True)
                print('Split ' +str(counter_folds) + ' de ' + str(kfold.n_splits) + '\n' +   'Usuário ' + str(counter) + ' de ' + str(len(u_reg)))
                
                test_stream = data_stream.create(data=samples, genuine=usuario, internal=u_reg, external=u_nao_reg)
                userBiometricModel = self.classifier.train_user_model(user_features=usersDatabase[usuario], normalize=normalize)
                FMR, FNMR, B_acc, _ = self.classifier.test(genuine_user=usuario, test_stream=test_stream, normalize=normalize,
                                                           user_model=userBiometricModel, decision_threshold=decision_threshold)
                list_of_scores['FMR'].append(FMR)
                list_of_scores['FNMR'].append(FNMR)
                list_of_scores['B_acc'].append(B_acc)
        return list_of_scores['FMR'], list_of_scores['FNMR'], list_of_scores['B_acc']
    
    def _assert_params(self,parameters):
        ''' 
            Essa função deve receber a variável 'p' e checar paralelamente se todos os conjuntos de parametros enviados estao de acordo.
        '''

    def fit(self, dataset=None, user_column='subject', metric=None, param_grid=None):
        ''' Falta criar uma função 'validation' que retorna os idx dos usuarios genuinos e impostores.
            Para cada 'p', executar com cada idx de validação.
        ''' 
        
        self._user_column = user_column
        self._usuarios = dataset[self._user_column].unique()
        self.list_params = list()
        self.list_scores = list()
        p = ParameterGrid(param_grid)

        for params in p:
            self.list_params.append(params)
            FMR, FNMR, B_acc = eval('self.'+self.system_type)(dataset=dataset, **params)
            self.list_scores.append([FMR, FNMR, B_acc])

    def summary(self, parameters = None, metrics = None):
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
    