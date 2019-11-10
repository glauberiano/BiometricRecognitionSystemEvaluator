#Automatic Biometric Recognition System

'''
Classe principal:
    * Em um sistema de reconhecimento padrão, o sistema cria um modelo para todos os usuários
'''

from components import Enrollment, FindThreshold
from components import Random, GenFirst, ImpFirst, SeriesAttack
from classifiers import M2005Classifier, EuclidianClassifier, ManhattanClassifier, OCSVMClassifier, MahalanobisClassifier, StatisticClassifier
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
        self.save_results = save_results
        self.verbose = verbose
        #self.random_state = np.random.RandomState(seed=42)
        self.random_state = 42

    @staticmethod
    def check_len_test_stream(y_true, impostor_rate, genuine_user):
        len_impostor = sum(y_true!=genuine_user) / len(y_true)
        if (len_impostor > impostor_rate) or (len_impostor < impostor_rate-0.2):
            raise Exception("ERROOOO! len_impostor: {}, impostor_rate {}".format(len_impostor, impostor_rate))
        
    def __validation(self, method='users_kfold', n_splits=5, list_users=None):
        if method=='users_kfold':
            kfold = KFold(n_splits=5)
            return kfold.split(list_users)
    
    def _run(self, dataset=None, method=None, model_size=None, impostor_rate=0.5, rate_external_impostor=0,
                         sampling='Random', len_attacks=None, normalize=False, adaptive=None, base=None, index=None):
        
        '''Inicializa o sistema biometrico. Para cada usuário do conjunto de dados, o sistema retorna as métricas `fnmr`, `fmr` e `bacc` para cada configuração fluxo de dados recebido;
        '''

        assert method in ['M2005', 'Euclidian', 'Manhattan', 'OCSVM', 'Mahalanobis', 'Statistic'], 'method must be on of \'M2005\', \'Euclidian\', \'Manhattan\', \'OCSVM\', \'Statistic\''
        self.classifier = eval(method+'Classifier')(name=method, normalize=normalize, adaptive=adaptive)
        
        self.metrics = {'FMR' : list(),
                        'FNMR' : list(),
                        'B_acc' :  list()}

        # data_stream: objeto que guarda configurações de como será construido o fluxo de dados no teste.
        data_stream = eval(sampling)(impostor_rate=impostor_rate, rate_external_impostor=rate_external_impostor, len_attacks=len_attacks, random_state=self.random_state)
        
        # splitTrainTest: objeto que separa os indexes dos usuários de modo que possamos validar os resultados com amostras genuinas e impostoras. O método `create()` separa as `model_size` primeiras amostras de cada usuário para 
        splitTrainTest = Enrollment()
        kfold = KFold(n_splits=5)
        splits = kfold.split(self._usuarios)
        self.y_true = list()
        self.decision = list()
        self.model_score = list()
        for internal_idx, external_idx in splits:
            #import pdb; pdb.set_trace()
            u_reg = self._usuarios[internal_idx]
            u_nao_reg = self._usuarios[external_idx]
            usersDatabase, samples, decision_threshold = splitTrainTest.create(dataset=dataset, n_amostras=model_size, users_list=u_reg, classifier=method, normalize=normalize, random_state=self.random_state)
            #import pdb; pdb.set_trace();
            for usuario_genuino in u_reg:
                test_stream = data_stream.create(data=samples, genuine=usuario_genuino, internal=u_reg, external=u_nao_reg)
                RecognitionSystem.check_len_test_stream(y_true=test_stream['subject'], impostor_rate=impostor_rate, genuine_user=usuario_genuino)
                user_biometric_model = self.classifier.train_user_model(user_features=usersDatabase[usuario_genuino])
                y_genuine, y_impostor, user_biometric_model, decision, model_score = self.classifier.test(genuine_user=usuario_genuino, test_stream=test_stream,
                                                           user_model=user_biometric_model, decision_threshold=decision_threshold)
                #import pdb; pdb.set_trace()
                fmr, fnmr, bacc = Metrics.report(y_genuine=y_genuine, y_impostor=y_impostor)
                self.y_true.append([1 if user==usuario_genuino else 0 for user in test_stream['subject']])
                self.decision.append(decision)
                self.model_score.append(model_score)
                self.metrics['FMR'].append(fmr)
                self.metrics['FNMR'].append(fnmr)
                self.metrics['B_acc'].append(bacc)
                self.decision_threshold[index].append(decision)
                #self.results_per_user[index][usuario_genuino] = self.results_per_user[index][usuario_genuino] + [bacc]
                

        return self.metrics['FMR'], self.metrics['FNMR'], self.metrics['B_acc']
    
    def _assert_params(self,parameters):
        ''' 
            Essa função deve receber a variável 'p' e checar paralelamente se todos os conjuntos de parametros enviados estao de acordo.
        '''

    def fit(self, user_column='subject', metric=None, param_grid=None):
        #0import pdb; pdb.set_trace()
        self._user_column = user_column # Variavel necessaria para separar os usuarios
        self.list_params = list() # lista de parametros
        self.list_scores = list() # lista de scores
        p = ParameterGrid(param_grid)
        keys = ['param_'+str(i) for i in range(len(p))]
        #self.results_per_user = dict(zip(keys, [dict(zip(self._usuarios, [list()]*len(self._usuarios)))]*len(p)))
        self.decision_threshold = dict(zip(keys, [list()]*len(p)))

        for i,params in enumerate(p): # Para cada conjunto de parametros
            base = RecognitionSystem.datasets(filename=params['base'])
            self._usuarios = base[self._user_column].unique() # lista de usuarios
            
            if self.verbose==True:      # se verbose == True, imprimir parametros
                print("Parametro {} / {}".format(i+1, len(p)))
                print(params)
                inic = time.time()
            self.list_params.append(params) # atualizando lista de parametros
            #import pdb; pdb.set_trace()
            FMR, FNMR, B_acc = self._run(dataset=base, index=keys[i], **params) 

            self.list_scores.append([FMR, FNMR, B_acc])
            if self.verbose==True:
                fim = time.time()-inic
                print("Tempo: {} minutos".format(fim / 60))

        if self.save_results == True:
            df = self.summary()
            if os.path.exists("resultados") == False:
                os.mkdir('resultados')
            df.to_csv('resultados/'+params['base']+'_'+params['method']+'.csv', index=False)

    def summary(self):
        """Retorna um pd.DataFrame() com o desempenho obtido para cada métrica de avaliação.
        
        Parameters:
            \tAtualmente nao recebe nenhum parametro
            
        Returns: 
            \tpd.Dataframe
        """

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

    @staticmethod
    def datasets(filename):
        if filename == 'DSL':
            base = pd.read_csv('dados/DSL-StrongPasswordData.csv')
            base = base.drop(['sessionIndex', 'rep'], axis=1)
            return base
        elif filename == 'easy':
            base = pd.read_csv('dados/easy.csv')
            return base
        elif filename == 'strong':
            base = pd.read_csv('dados/strong.csv')
            return base
        elif filename == 'logical':
            base = pd.read_csv('dados/logical.csv')
            return base
        elif filename == 'greyc':
            base = pd.read_csv('dados/greyc_tratado.csv')
            temp = (base['subject'].value_counts() >= 60).to_frame('value').reset_index()
            users_list = temp.loc[temp['value']==True, 'index'].values.tolist()
            df = base[base['subject'].isin(users_list)].reset_index(drop=True)
            return df