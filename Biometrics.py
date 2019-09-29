#Automatic Biometric Recognition System

'''
Classe principal:
    * Em um sistema de reconhecimento padrão, o sistema cria um modelo para todos os usuários
'''

from components import Enrollment, FindThreshold
from components import Random, GenFirst, ImpFirst, SeriesAttack
from classifiers import M2005Classifier, EuclidianClassifier, ManhattanClassifier, OCSVMClassifier, MahalanobisClassifier
from classifiers import Metrics
import pandas as pd
import numpy as np
import pickle
import os
import time
import IPython.core.display as ipd
from sklearn.model_selection import KFold
from sklearn.model_selection import ParameterGrid

class RecognitionSystem:
    def __init__(self,  verbose=False, save_results=False, filename=None, random_state=True):
        #if (save_results == True) and (filename==None):
        #    raise Exception("Se deseja salvar os resultados, defina um nome em filename")
        self.save_results = save_results
        #self.filename = filename
        self.verbose = verbose
        self.random_state = np.random.RandomState(seed=42)

    @staticmethod
    def check_len_test_stream(y_true, impostor_rate, genuine_user):
        len_impostor = sum(y_true!=genuine_user) / len(y_true)
        if (len_impostor > impostor_rate) or (len_impostor < impostor_rate-0.2):
            raise Exception("ERROOOO! len_impostor: {}, impostor_rate {}".format*(len_impostor, impostor_rate))
        
    def __validation(self, method='users_kfold', n_splits=5, list_users=None):
        if method=='users_kfold':
            kfold = KFold(n_splits=5)
            return kfold.split(list_users)
    
    def _run(self, dataset=None, method=None, model_size=None, impostor_rate=0.5, rate_external_impostor=0,
                         sampling='Random', len_attacks=None, normalize=False, adaptive=None, base=None, index=None):
        
        '''
        Funcao: executar o sistema!

        Para o dataset selecionado; com os parametros selecionados;
        '''


        assert method in ['M2005', 'Euclidian', 'Manhattan', 'OCSVM', 'Mahalanobis'], 'method must be on of \'M2005\', \'Euclidian\', \'Manhattan\', \'OCSVM\''
        self.classifier = eval(method+'Classifier')(name=method, normalize=normalize, adaptive=adaptive)
        
        list_of_scores = {'FMR' : list(),
                          'FNMR' : list(),
                          'B_acc' :  list()}

        #list_test = list()
        # data_stream: objeto que guarda configurações de como será construido o fluxo de dados no teste.
        data_stream = eval(sampling)(impostor_rate=impostor_rate, rate_external_impostor=rate_external_impostor, len_attacks=len_attacks, random_state=self.random_state)
        
        # splitTrainTest: objeto que separa os indexes dos usuários de modo que possamos validar os resultados com amostras genuinas e impostoras
        splitTrainTest = Enrollment()
        kfold = KFold(n_splits=5)
        splits = kfold.split(self._usuarios)
        

        for internal_idx, external_idx in splits:
            u_reg = self._usuarios[internal_idx]
            u_nao_reg = self._usuarios[external_idx]
            #import pdb; pdb.set_trace();
            usersDatabase, samples, decision_threshold = splitTrainTest.create(dataset=dataset, n_amostras=model_size, users_list=u_reg, classifier=method, normalize=normalize, random_state=self.random_state)
            #import pdb; pdb.set_trace();
            for usuario in u_reg:
                test_stream = data_stream.create(data=samples, genuine=usuario, internal=u_reg, external=u_nao_reg)
                RecognitionSystem.check_len_test_stream(y_true=test_stream['subject'], impostor_rate=impostor_rate, genuine_user=usuario)
                user_biometric_model = self.classifier.train_user_model(user_features=usersDatabase[usuario])
                y_genuine, y_impostor, user_biometric_model = self.classifier.test(genuine_user=usuario, test_stream=test_stream,
                                                           user_model=user_biometric_model, decision_threshold=decision_threshold)
                import pdb; pdb.set_trace()
                fmr, fnmr, bacc = Metrics.report(y_genuine=y_genuine, y_impostor=y_impostor)
                list_of_scores['FMR'].append(fmr)
                list_of_scores['FNMR'].append(fnmr)
                list_of_scores['B_acc'].append(bacc)
                self.results_per_user[index][usuario] = self.results_per_user[index][usuario] + [bacc]
                

        return list_of_scores['FMR'], list_of_scores['FNMR'], list_of_scores['B_acc']
    
    def _assert_params(self,parameters):
        ''' 
            Essa função deve receber a variável 'p' e checar paralelamente se todos os conjuntos de parametros enviados estao de acordo.
        '''

    def fit(self, dataset=None, user_column='subject', metric=None, param_grid=None):
        
        self._user_column = user_column # Variavel necessaria para separar os usuarios
        self._usuarios = dataset[self._user_column].unique() # lista de usuarios
        self.list_params = list() # lista de parametros
        self.list_scores = list() # lista de scores
        p = ParameterGrid(param_grid)
        keys = ['param_'+str(i) for i in range(len(p))]
        self.results_per_user = dict(zip(keys, [dict(zip(self._usuarios, [list()]*len(self._usuarios)))]*len(p)))
                
        for i,params in enumerate(p): # Para cada conjunto de parametros
            if self.verbose==True:      # se verbose == True, imprimir parametros
                print(params)
                inic = time.time()
            self.list_params.append(params) # atualizando lista de parametros
            FMR, FNMR, B_acc = self._run(dataset=dataset, index=keys[i], **params) 

            self.list_scores.append([FMR, FNMR, B_acc])
            if self.verbose==True:
                fim = time.time()-inic
                print("Parametro {} / {}   tempo: {}".format(i, len(p), fim))

        if self.save_results == True:
            df = self.summary(parameters=self.list_params, metrics=self.list_scores)
            if os.path.exists("resultados") == False:
                os.mkdir('resultados')
            df.to_csv('resultados/'+params['base']+'_'+params['method']+'.csv', index=False)

    def summary(self, parameters = None, metrics = None):

        list_scores=list()
        for i in range(len(self.list_scores)):
            scores = {'FMR_mean' : np.mean(self.list_scores[i][0]),
                      'FMR_std' : np.std(self.list_scores[i][0]),
                      'FNMR_mean' : np.mean(self.list_scores[i][1]),
                      'FNMR_std' : np.std(self.list_scores[i][1]),
                      'Bacc_mean' :  np.mean(self.list_scores[i][2]),
                      'Bacc_std' : np.std(self.list_scores[i][2])}
            list_scores.append(scores)
        df1 = pd.DataFrame(self.list_params)
        df2 = pd.DataFrame(list_scores)
        frames = [df1, df2]
        return pd.concat(frames, axis=1)
