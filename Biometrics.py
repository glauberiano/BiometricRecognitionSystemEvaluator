#Automatic Biometric Recognition System

'''
Classe principal:
    * Em um sistema de reconhecimento padrão, o sistema cria um modelo para todos os usuários
'''

from components import Enrollment, Validation
from components import Random, GenFirst, ImpFirst, SeriesAttack
from classifiers import M2005Classifier
from classifiers import EuclidianClassifier
from classifiers import Metrics
import pandas as pd
import numpy as np
import pickle
import os
import time
import IPython.core.display as ipd
from sklearn.model_selection import KFold
from sklearn.model_selection import ParameterGrid

class RecognitionSystem():
    def __init__(self,  verbose=False, save_results=False, filename=None):
        if (save_results == True) and (filename==None):
            raise Exception("Se deseja salvar os resultados, defina um nome em filename")
        self.save_results = save_results
        self.filename = filename
        self.verbose = verbose
        
    def __validation(self, method='users_kfold', n_splits=5, list_users=None):
        if method=='users_kfold':
            kfold = KFold(n_splits=5)
            return kfold.split(list_users)
    
    def _run(self, dataset=None, method=None, model_size=None, impostor_rate=0.5, rate_external_impostor=0,
                         sampling='Random', len_attacks=None, normalize=False, adaptive=None):
        
        assert method in ['M2005', 'Euclidian'], 'method must be on of \'M2005\', \'Euclidian\''
        self.classifier = eval(method+'Classifier')(name=method, normalize=normalize, adaptive=adaptive)
        
        list_of_scores = {'FMR' : list(),
                          'FNMR' : list(),
                          'B_acc' :  list()}

        import pdb;pdb.set_trace();
        
        # data_stream: objeto que guarda configurações de como será construido o fluxo de dados no teste.
        data_stream = eval(sampling)(impostor_rate=impostor_rate, rate_external_impostor=rate_external_impostor, len_attacks=len_attacks)
        
        # splitTrainTest: objeto que separa fluxos de treino e de teste.
        splitTrainTest = Enrollment()
        kfold = KFold(n_splits=5)
        splits = kfold.split(self._usuarios)
        
        #counter_folds = 0
        for internal_idx, external_idx in splits:
            u_reg = self._usuarios[internal_idx]
            u_nao_reg = self._usuarios[external_idx]
            usersDatabase, samples, decision_threshold = splitTrainTest.create(dataset=dataset, n_amostras=model_size, users_list=u_reg, classifier=method, normalize=normalize)
            #counter=0
            #counter_folds += 1
            
            for usuario in u_reg:
                #counter += 1
                #ipd.clear_output(wait=True)
                #print('Split ' +str(counter_folds) + ' de ' + str(kfold.n_splits) + '\n' +   'Usuário ' + str(counter) + ' de ' + str(len(u_reg)))
                #import pdb;pdb.set_trace();
                test_stream = data_stream.create(data=samples, genuine=usuario, internal=u_reg, external=u_nao_reg)
                userBiometricModel = self.classifier.train_user_model(user_features=usersDatabase[usuario])
                y_genuine, y_impostor, userBiometricModel = self.classifier.test(genuine_user=usuario, test_stream=test_stream,
                                                           user_model=userBiometricModel, decision_threshold=decision_threshold)

                #import pdb;pdb.set_trace();
                FMR, FNMR, B_acc = Metrics.report(y_genuine=y_genuine, y_impostor=y_impostor)
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

        for i,params in enumerate(p):
            if self.verbose==True:
                print(params)
                inic = time.time()
            self.list_params.append(params)
            FMR, FNMR, B_acc = self._run(dataset=dataset, **params)
            self.list_scores.append([FMR, FNMR, B_acc])
            if self.verbose==True:
                fim = time.time()-inic
                print("Parametro {} / {}   tempo: {}".format(i, len(p), fim))

        if self.save_results == True:
            df = self.summary(parameters=self.list_params, metrics=self.list_scores)
            self.save(dataframe=df, filename=(self.filename+'.csv'))
            
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
    
    def save(self, dataframe, filename):
        if os.path.exists("resultados") == False:
            os.mkdir('resultados')
        dataframe.to_csv(os.path.join('resultados',filename))