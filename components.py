import pandas as pd
import numpy as np
import random
import math
from sklearn.metrics import classification_report, confusion_matrix 
from abc import ABC, abstractmethod
from classifiers import M2005Classifier, EuclidianClassifier

class Enrollment:
    def __init__(self):
        pass

    def create(self, dataset=None, n_amostras=None, users_list=None, classifier=None, normalize=None):
        '''
        parameters:
            - user_column (str): coluna do dataset que indica o usuário
            - n (int): as n primeiras amostras do usuário devem ser guardadas no banco de dados.
            - users (pandas.Series): lista de usuários que serão registrados
            
        return:
            - banco de dados (dict) com as informações de cada usuário e;
            - pandas.Dataframe das observações não utilizadas que poderão ser usadas no criação do fluxo de dados de teste
        '''
        valid = Validation(classifier=classifier, normalize=normalize) # objeto para validadar linha de decisao usada na etapa de teste
        user_information = {} #dicionário em que serão salvas as informações de cada usuário registrado
        index_used_in_db = list() #lista de indices das observações salvas no banco de dados
        for user in users_list: #para cada usuário no dataset enviado
            temp = dataset[dataset['subject'] == user] #separar as observações do usuário
            user_information[user] = temp.iloc[:n_amostras] #salvar as n primeiras observações no dicionário
            index_used_in_db = index_used_in_db + user_information[user].index.tolist() #guardar os indices utilizados
        
        users_decision_threshold = valid.run(usersData=user_information)
        data_not_used = dataset.drop(dataset.index[index_used_in_db]) #salvar as observações que não foram utilizadas
        return user_information, data_not_used, users_decision_threshold

class Validation:
    def __init__(self, classifier, normalize):
        self.train = eval(classifier+'Classifier')(name=classifier, normalize=normalize, adaptive=False)
        self.model = classifier

    def run(self, usersData):
        thresholds_dict = dict()
        first = True
        for user in usersData.keys():
            treino = usersData[user][:20]
            modelo = self.train.train_user_model(treino)
            impostor_users = np.setdiff1d([*usersData] , user)
            for iu in impostor_users:
                if first == True:
                    impostor_data = pd.DataFrame(usersData[iu][:40])
                    first = False
                else:
                    impostor_data = pd.concat([impostor_data, usersData[iu][:40]], axis=0, ignore_index=True)
            impostor_data  = impostor_data.sample(20, replace=False).reset_index(drop=True)          
            
            if self.model == 'Euclidian':
                scoreType='distance'
                genuineScores = self.euclidian_score(modelo.model, usersData[user][20:40])
                impostorScores = self.euclidian_score(modelo.model, impostor_data.loc[:,impostor_data.columns!='subject'])
            elif self.model == 'M2005':
                scoreType='similarity'
                genuineScores = self.M2005_score(modelo.model, usersData[user][20:40])
                impostorScores = self.M2005_score(modelo.model, impostor_data.loc[:,impostor_data.columns!='subject'])
            decision = self.calculate_best_threshold(userScores=genuineScores, impostorScores=impostorScores, scoreType=scoreType)
            thresholds_dict[user] = decision
        return thresholds_dict

    def euclidian_score(self, user_model, test_stream):
        try:
            test_stream=test_stream.drop('subject', axis=1)
        except:
            pass

        p = user_model.shape[0]
        if ((test_stream.shape[1]) !=p):
            raise Exception("Numero de features diferente")
        
        scores =list()
        for _, row in test_stream.iterrows():
            score = np.sqrt(sum((row - user_model)**2))
            scores.append(score)
        return scores

    def M2005_score(self, user_model, test_stream):
        try:
            test_stream=test_stream.drop('subject', axis=1)
        except:
            pass

        #p = user_model.shape[0]
        #if ((test_stream.shape[1]) !=p):
        #    raise Exception("Numero de features diferente")

        
        #import pdb;pdb.set_trace();
        scores = list()
        for _, row in test_stream.iterrows():
            match_sum = 0
            previousDimMatched = False
            for dim in user_model.keys():
                if (row[dim] <= user_model[dim][1]) and (row[dim] >= user_model[dim][0]):
                    if previousDimMatched:
                        match_sum = match_sum + 1.5
                    else:
                        match_sum = match_sum + 1.0
                    previousDimMatched = True
                else:
                    previousDimMatched = False
            max_sum = 1.0 + 1.5 * (len(user_model.keys()) -1)
            scores.append(match_sum/max_sum)
        return scores

    def reporter(self, scores, true_labels, decision, scoreType):
        y_genuine = list()
        y_impostor = list()
        if (scoreType=='distance'):
            for i, score in enumerate(scores):
                if (score < decision): #classifiquei como positivo
                    if (true_labels[i] == 1):
                        y_genuine.append(1)
                    else:
                        y_impostor.append(1)
                else:
                    if (true_labels[i] == 1):
                        y_genuine.append(0)
                    else:
                        y_impostor.append(0)
        else:
             for i, score in enumerate(scores):
                if (score > decision): #classifiquei como positivo
                    if (true_labels[i] == 1):
                        y_genuine.append(1)
                    else:
                        y_impostor.append(1)
                else:
                    if (true_labels[i] == 1):
                        y_genuine.append(0)
                    else:
                        y_impostor.append(0)
                 
        FNMR = 1- sum(y_genuine) / len(y_genuine)
        FMR = sum(y_impostor) / len(y_impostor)
        BAcc = 1- (FNMR + FMR)/2
        return FMR, FNMR, BAcc

    def calculate_best_threshold(self, userScores, impostorScores, scoreType):
        predictions = userScores + impostorScores
        labels = np.concatenate((np.ones(len(userScores)), np.zeros(len(impostorScores))))
        
        best_BAcc = -float("inf")
        for score in predictions:
            _, _, BAcc = self.reporter(scores=predictions, true_labels=labels, decision=score, scoreType=scoreType)
            if BAcc > best_BAcc:
                best_BAcc = BAcc
                decision = score
                
        return decision

class DataStream(ABC):
    def __init__(self, impostor_rate, rate_external_impostor, len_attacks=None):
        self.impostor_rate = impostor_rate # taxa de impostores
        self.rate_external_impostor = rate_external_impostor # dos impostores no fluxo, define quantos são externos
        self._len_attacks = len_attacks
        super().__init__()

    def _extract_datasets(self, data, genuine, internal, external):
        genuine_samples = data[data['subject']==genuine] # separando amostras genuínas;
        in_samples = data[data['subject'].isin(internal)] # separando amostras de usuários registrados
        int_imp_samples = in_samples[in_samples['subject'] != genuine] # com exceção do usuário genuino;
        ext_imp_samples = data[data['subject'].isin(external)] # separando amostras de usuários não registrados
        
        # separando o número de amostras que serão utilizadas no fluxo
        n_impostor = int((len(genuine_samples) * self.impostor_rate) / (1-self.impostor_rate))
        n_internal_imp = int(n_impostor * (1-self.rate_external_impostor))
        n_external_imp = n_impostor - n_internal_imp

        internal_samples=int_imp_samples.iloc[:n_internal_imp]
        external_samples=ext_imp_samples.iloc[:n_external_imp]

        impostor_samples = pd.concat([internal_samples, external_samples], axis=0, ignore_index=True)

        return genuine_samples, impostor_samples
        
    def _extrai(self, df1):
        a = df1.values.tolist()[0]
        df1.drop(df1.index[0], inplace=True)
        return a

    @abstractmethod
    def create(self):
        pass

class Random(DataStream):
    def create(self, data=None, genuine=None, internal=None, external=None):
        genuine_samples, impostor_samples = self._extract_datasets(data=data, genuine=genuine, internal=internal, external=external)
        l_ds = list()
        if type(genuine_samples) != None:
            b = np.ones(len(genuine_samples)).tolist()
            l_ds.extend(b)
            c = genuine_samples.columns
            genuine_samples = genuine_samples.reset_index(drop=True)
        if type(impostor_samples) != None:
            b = np.zeros(len(impostor_samples)).tolist()
            l_ds.extend(b)
            c = impostor_samples.columns
            impostor_samples = impostor_samples.reset_index(drop=True)

        random.Random(42).shuffle(l_ds)
        datastream = list()
        for i in l_ds:
            if i == 1:
                datastream.append(self._extrai(genuine_samples))
                genuine_samples = genuine_samples.reset_index(drop=True)
            if i == 0:
                datastream.append(self._extrai(impostor_samples))
                impostor_samples = impostor_samples.reset_index(drop=True)

        datastream = pd.DataFrame(datastream, columns=c)
        return datastream

class GenFirst(DataStream):
    def create(self, data=None, genuine=None, internal=None, external=None):
        genuine_samples, impostor_samples = self._extract_datasets(data=data, genuine=genuine, internal=internal, external=external)
        frames = [genuine_samples, impostor_samples]
        return pd.concat(frames, ignore_index=True)

class ImpFirst(DataStream):
    def create(self, data=None, genuine=None, internal=None, external=None):
        genuine_samples, impostor_samples = self._extract_datasets(data=data, genuine=genuine, internal=internal, external=external)
        frames = [impostor_samples, genuine_samples]
        return pd.concat(frames, ignore_index=True)

class SeriesAttack(DataStream):
    '''falta ajustar para impostores externos
    '''
    def create(self, data=None, genuine=None, internal=None, external=None):
        #import pdb;pdb.set_trace();
        genuine_samples, impostor_samples = self._extract_datasets(data=data, genuine=genuine, internal=internal, external=external)
        n_series = math.ceil(len(impostor_samples) / self._len_attacks)
        lenG = math.ceil(len(genuine_samples)/n_series)
        ds = list()
        for i in range(n_series):
            i_idx = i*self._len_attacks
            g_idx = i*lenG
            try:
                ds.append(impostor_samples[i_idx:i_idx+self._len_attacks])
                ds.append(genuine_samples[g_idx:g_idx+lenG])
            except:
                ds.append(impostor_samples[i_idx:])
                ds.append(genuine_samples[g_idx:])
        return pd.concat(ds, ignore_index=True)