import pandas as pd
import numpy as np
import random
import math
from sklearn.metrics import classification_report, confusion_matrix 
from abc import ABC, abstractmethod
from classifiers import M2005Classifier, EuclidianClassifier, ManhattanClassifier, OCSVMClassifier, MahalanobisClassifier, StatisticClassifier
import time

class Enrollment:
    def __init__(self):
        pass

    def create(self, dataset=None, n_amostras=None, users_list=None, classifier=None, normalize=None, random_state=None):
        '''
        parameters:
            - user_column (str): coluna do dataset que indica o usuário
            - n (int): as n primeiras amostras do usuário devem ser guardadas no banco de dados.
            - users (pandas.Series): lista de usuários que serão registrados
            
        return:
            - banco de dados (dict) com as informações de cada usuário e;
            - pandas.Dataframe das observações não utilizadas que poderão ser usadas no criação do fluxo de dados de teste
        '''
        best_threshold = FindThreshold(classifier=classifier, normalize=normalize, random_state=random_state) # objeto para validadar linha de decisao usada na etapa de teste
        user_information = {} #dicionário em que serão salvas as informações de cada usuário registrado
        index_used_in_db = list() #lista de indices das observações salvas no banco de dados
        for user in users_list: #para cada usuário no dataset enviado
            temp = dataset[dataset['subject'] == user].reset_index(drop=True) #separar as observações de cada usuário
            user_information[user] = temp.iloc[:n_amostras] #salvar as n primeiras observações para encontrar o melhor threshold
            index_used_in_db = index_used_in_db + user_information[user].index.tolist() #guardar os indices utilizados
        
        #import pdb;pdb.set_trace();
        users_decision_threshold = best_threshold.run(usersData=user_information, model_size=n_amostras)
        data_not_used = dataset.drop(dataset.index[index_used_in_db]) #salvar as observações que não foram utilizadas, para utilizar na etapa de teste
        return user_information, data_not_used, users_decision_threshold

class FindThreshold:
    ''' Essa classe é responsável por encontrar o melhor limiar de decisão que separa amostras impostoras e genuínas para cada um dos
    classificadores utilizados.

    '''

    def __init__(self, classifier, normalize, random_state):
        self.train = eval(classifier+'Classifier')(name=classifier, normalize=normalize, adaptive=False)
        self.model = classifier
        self.random_state = random_state

    def run(self, usersData, model_size):
        '''Parameters:
        usersData - dicionário em que .key() é o usuário e .values() as amostras desse usuario que devem ser utilizadas no cadastramento


        Return:
        threshold_dict - dicionario em que .key() é o usuário e .values() a melhor linha de decisão os dados de treinamento vistos.
        '''

        thresholds_dict = dict()
        first = True
        #import pdb; pdb.set_trace()
        #list_impostor_samples = list()
        
        # Um usuario é definido como genuino, enquanto os outros são impostores
        for user in usersData.keys(): #Para cada usuario no banco
            treino = usersData[user][:model_size//2] 
            user_model = self.train.train_user_model(treino) #treinar um modelo 
            impostor_users = np.setdiff1d([*usersData] , user)
            #import pdb;pdb.set_trace()
            for iu in impostor_users:
                if first == True:
                    impostor_df = pd.DataFrame(usersData[iu][:model_size])
                    first = False
                else:
                    impostor_df = pd.concat([impostor_df, usersData[iu][:model_size]], axis=0, ignore_index=True)
            
            #import pdb; pdb.set_trace()
            impostor_df = impostor_df.reset_index(drop=True)
            impostor_data  = impostor_df.sample(model_size//2, replace=False, random_state=self.random_state) #sorteio de 20 amostras aleatorias dentre todas as possíveis          
        
            # Gerando scores para fluxo genuino e impostor
            #import pdb; pdb.set_trace()
            if self.model == 'Euclidian':
                scoreType = 'distance'
                genuineScores = self.euclidian_score(user_model.model, usersData[user][model_size//2:model_size])
                impostorScores = self.euclidian_score(user_model.model, impostor_data.loc[:,impostor_data.columns!='subject'])
                #impostorScores = self.euclidian_score(user_model.model, list_impostor_samples)
            elif self.model == 'M2005':
                scoreType = 'similarity'
                genuineScores = self.M2005_score(user_model.model, usersData[user][model_size//2:model_size])
                impostorScores = self.M2005_score(user_model.model, impostor_data.loc[:,impostor_data.columns!='subject'])
            elif self.model == 'Manhattan':
                scoreType = 'distance'
                genuineScores = self.manhattan_score(user_model.model, usersData[user][model_size//2:model_size])
                impostorScores = self.manhattan_score(user_model.model, impostor_data.loc[:,impostor_data.columns!='subject'])
            elif self.model == 'OCSVM':
                scoreType = 'similarity'
                genuineScores = self.ocsvm_score(user_model, usersData[user][model_size//2:model_size])
                impostorScores = self.ocsvm_score(user_model, impostor_data.loc[:,impostor_data.columns!='subject'])
            elif self.model == 'Mahalanobis':
                scoreType = 'distance'
                genuineScores = self.mahalanobis_score(user_model, usersData[user][model_size//2:model_size])
                impostorScores = self.mahalanobis_score(user_model, impostor_data.loc[:,impostor_data.columns!='subject'])
            elif self.model == 'Statistic':
                scoreType = 'distance'
                genuineScores = self.statistic_score(user_model, usersData[user][model_size//2:model_size])
                impostorScores = self.statistic_score(user_model, impostor_data.loc[:,impostor_data.columns!='subject'])
            #import pdb;pdb.set_trace()
            decision_threshold = self.calculate_best_threshold(user_scores=genuineScores, impostor_scores=impostorScores, scoreType=scoreType)
            thresholds_dict[user] = decision_threshold
        return thresholds_dict
    
    def calculate_best_threshold(self, user_scores, impostor_scores, scoreType):
        predictions = user_scores + impostor_scores
        labels = np.concatenate((np.ones(len(user_scores)), np.zeros(len(impostor_scores))))
        best_bacc = -float("inf")
        #import pdb; pdb.set_trace()
        for score in predictions:
            _, _, bacc = self.reporter(scores=predictions, true_labels=labels, decision=score, scoreType=scoreType)
            if bacc > best_bacc:
                best_bacc = bacc
                decision = score
        #import pdb; pdb.set_trace()
        return decision

    
    def reporter(self, scores, true_labels, decision, scoreType):
        y_genuine = list()
        y_impostor = list()
        if scoreType=='distance':
            y_genuine = [1 if scores[i] < decision else 0 for i, sample in enumerate(true_labels) if sample == 1]
            y_impostor = [1 if scores[i] < decision else 0 for i, sample in enumerate(true_labels) if sample == 0]

        elif scoreType=='similarity':
            y_genuine = [1 if scores[i] > decision else 0 for i, sample in enumerate(true_labels) if sample == 1]
            y_impostor = [1 if scores[i] > decision else 0 for i, sample in enumerate(true_labels) if sample == 0]
        
        fnmr= 1- sum(y_genuine) / len(y_genuine)
        fmr = sum(y_impostor) / len(y_impostor)
        bacc = 1- (fnmr + fmr)/2
        return fmr, fnmr, bacc

    def statistic_score(self, user_model, test_stream):
        try:
            test_stream=test_stream.drop('subject', axis=1)
        except:
            pass

        scores =list()
        for _, row in test_stream.iterrows():
            score = 1 - sum(np.e ** ( ( abs(row- user_model.model['Mean']) / user_model.model['Desvpad'])*-1 )) / len(row)
            scores.append(score)
        return scores

    def ocsvm_score(self, user_model, test_stream):
        try:
            test_stream=test_stream.drop('subject', axis=1)
        except:
            pass

        scores =list()
        for _, row in test_stream.iterrows():
            score = user_model.model.decision_function(row.values.reshape(1,-1))
            scores.append(score)
        return scores

    def mahalanobis_score(self, user_model, test_stream):
        try:
            test_stream=test_stream.drop('subject', axis=1)
        except:
            pass
        
        scores =list()
        for _, row in test_stream.iterrows():
            score = np.sqrt(np.dot(np.dot(row-user_model.model['Mean'], user_model.model['InvCov']), row-user_model.model['Mean'])**2)
            scores.append(score)
        return scores

    def euclidian_score(self, user_model, test_stream):
        try:
            test_stream=test_stream.drop('subject', axis=1)
        except:
            pass
        
        scores = list()
        for _, row in test_stream.iterrows():
            score = np.sqrt(sum((row - user_model)**2))
            scores.append(score)
            #list_scores.append(scores)
        return scores

    def manhattan_score(self, user_model, test_stream):
        try:
            test_stream=test_stream.drop('subject', axis=1)
        except:
            pass

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

        scores = list()
        for _, row in test_stream.iterrows():
            match_sum = 0
            previousDimMatched = False
            for dim in user_model.keys():
                if (row[dim] >= user_model[dim][0]) and (row[dim] <= user_model[dim][1]):
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

#####################################################################################################
#####################################################################################################

class DataStream(ABC):
    def __init__(self, impostor_rate, rate_external_impostor, len_attacks=None, random_state=None):
        self.impostor_rate = impostor_rate # taxa de impostores
        self.rate_external_impostor = rate_external_impostor # dos impostores no fluxo, define quantos são externos
        self._len_attacks = len_attacks
        self.random_state = random_state
        super().__init__()

    def _extract_datasets(self, data, genuine, internal, external):
        genuine_samples = data[data['subject']==genuine].reset_index(drop=True) # fluxo de dados genuinos;
        in_samples = data[data['subject'].isin(internal)] # fluxo de dados de usuarios cadastrados
        int_imp_samples = in_samples[in_samples['subject'] != genuine].reset_index(drop=True) # com exceção do usuário genuino;
        ext_imp_samples = data[data['subject'].isin(external)].reset_index(drop=True) # fluxo de dados de usuarios externos
        
        # separando o número de amostras que serão utilizadas no fluxo
        n_impostor = int((len(genuine_samples) * self.impostor_rate) / (1-self.impostor_rate))
        n_internal_imp = int(n_impostor * (1-self.rate_external_impostor))
        n_external_imp = n_impostor - n_internal_imp

        internal_samples=int_imp_samples.iloc[:n_internal_imp]
        external_samples=ext_imp_samples.iloc[:n_external_imp]
        impostor_samples = pd.concat([internal_samples, external_samples], axis=0, ignore_index=True)

        # retorna um fluxo de dados genuinos e um fluxo de dados impostores
        return genuine_samples, impostor_samples
        
    def _extrai(self, df1):
        a = df1.values.tolist()[0]
        df1.drop(df1.index[0], inplace=True)
        return a

    @abstractmethod
    def create(self):
        pass

class Random(DataStream):
    def create(self, data=None, genuine=None, internal=None, external=None, random_state=None):
        genuine_samples, impostor_samples = self._extract_datasets(data=data, genuine=genuine, internal=internal, external=external)
        l_ds = list() # lista binária. 1 == amostra genuina; 0 == amostra impostora; 
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

        #random.Random(42).shuffle(l_ds) #Embaralha a a lista binária
        random.seed(random_state)
        random.shuffle(l_ds)
        datastream = list()
        #import pdb;pdb.set_trace();
        
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
    def create(self, data=None, genuine=None, internal=None, external=None):
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

#############################################################################################
#############################################################################################
