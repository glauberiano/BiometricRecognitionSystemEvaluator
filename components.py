import pandas as pd
import numpy as np
import random
from sklearn.metrics import classification_report, confusion_matrix 
from abc import ABC, abstractmethod
from classifiers import M2005Classifier

class Enrollment:
    def __init__(self):
        pass

    def create(self, dataset=None, n_amostras=None, users_list=None):
        '''
        parameters:
            - user_column (str): coluna do dataset que indica o usuário
            - n (int): as n primeiras amostras do usuário devem ser guardadas no banco de dados.
            - users (pandas.Series): lista de usuários que serão registrados
            
        return:
            - banco de dados (dict) com as informações de cada usuário e;
            - pandas.Dataframe das observações não utilizadas que poderão ser usadas no criação do fluxo de dados de teste
        '''
        user_information = {} #dicionário em que serão salvas as informações de cada usuário registrado
        index_used_in_db = list() #lista de indices das observações salvas no banco de dados
        for user in users_list: #para cada usuário no dataset enviado
            temp = dataset[dataset['subject'] == user] #separar as observações do usuário
            user_information[user] = temp.iloc[:n_amostras] #salvar as n primeiras observações no dicionário
            index_used_in_db = index_used_in_db + user_information[user].index.tolist() #guardar os indices utilizados
        
        data_not_used = dataset.drop(dataset.index[index_used_in_db]) #salvar as observações que não foram utilizadas
        return user_information, data_not_used
    
class DataStream(ABC):
    def __init__(self, impostor_rate, rate_external_impostor, n_series=None):
        self.impostor_rate = impostor_rate # taxa de impostores
        self.rate_external_impostor = rate_external_impostor # dos impostores no fluxo, define quantos são externos
        self._n_series = n_series
        super().__init__()

    def _extract_datasets(self, data, genuine, internal, external):
        genuine_samples = data[data['subject']==genuine] # separando amostras genuínas;
        in_samples = data[data['subject'].isin(internal)] # separando amostras de usuários registrados
        int_imp_samples = in_samples[in_samples['subject'] != genuine] # com exceção do usuário genuino;
        ext_imp_samples = data[data['subject'].isin(external)] # separando amostras de usuários não registrados
        
        # separando o número de amostras que serão utilizadas no fluxo
        n_impostor = int((len(genuine_samples) * (1-self.impostor_rate)) / self.impostor_rate)
        n_internal_imp = int(n_impostor * (1-self.rate_external_impostor))
        n_external_imp = n_impostor - n_internal_imp

        genuine_samples=genuine_samples
        internal_samples=int_imp_samples.iloc[:n_internal_imp]
        external_samples=ext_imp_samples.iloc[:n_external_imp]

        return genuine_samples, internal_samples, external_samples
        
    def _extrai(self, df1):
        a = df1.values.tolist()[0]
        df1.drop(df1.index[0], inplace=True)
        return a

    @abstractmethod
    def create(self):
        pass

class Random(DataStream):
    def create(self, data=None, genuine=None, internal=None, external=None):
        genuine_samples, internal_samples, external_samples = self._extract_datasets(data=data, genuine=genuine, internal=internal, external=external)
        l_ds = list()
        if type(genuine_samples) != None:
            b = np.zeros(len(genuine_samples)).tolist()
            l_ds.extend(b)
            c = genuine_samples.columns
            genuine_samples = genuine_samples.reset_index(drop=True)
        if type(internal_samples) != None:
            b = np.ones(len(internal_samples)).tolist()
            l_ds.extend(b)
            c = internal_samples.columns
            internal_samples = internal_samples.reset_index(drop=True)
        if type(external_samples) != None:
            b = np.ones(len(external_samples))*2
            l_ds.extend(b.tolist())
            c = external_samples.columns
            external_samples = external_samples.reset_index(drop=True)

        random.Random(42).shuffle(l_ds)
        datastream = list()
        for i in l_ds:
            if i == 0:
                datastream.append(self._extrai(genuine_samples))
                genuine_samples = genuine_samples.reset_index(drop=True)
            if i == 1:
                datastream.append(self._extrai(internal_samples))
                internal_samples = internal_samples.reset_index(drop=True)
            if i == 2:
                datastream.append(self._extrai(external_samples))
                external_samples = external_samples.reset_index(drop=True)

        datastream = pd.DataFrame(datastream, columns=c)
        return datastream

class GenFirst(DataStream):
    def create(self, data=None, genuine=None, internal=None, external=None):
        genuine_samples, internal_samples, external_samples = self._extract_datasets(data=data, genuine=genuine, internal=internal, external=external)
        frames = [genuine_samples, internal_samples, external_samples]
        return pd.concat(frames)

class ImpFirst(DataStream):
    def create(self, data=None, genuine=None, internal=None, external=None):
        genuine_samples, internal_samples, external_samples = self._extract_datasets(data=data, genuine=genuine, internal=internal, external=external)
        frames = [internal_samples, external_samples, genuine_samples]
        return pd.concat(frames)

class SeriesAttack(DataStream):
    '''falta ajustar para impostores externos
    '''
    def create(self, data=None, genuine=None, internal=None, external=None):
        genuine_samples, internal_impostors, _ = self._extract_datasets(data=data, genuine=genuine, internal=internal, external=external)
        m = int(len(genuine_samples) / self._n_series)
        g_idx = list(range(0,len(genuine_samples), m))
        n = int(len(internal_impostors) / self._n_series)
        i_idx = list(range(0,len(internal_impostors), n))
        g_idx.append(len(genuine_samples))
        i_idx.append(len(internal_impostors))
        ds = list()
        for i in range(self._n_series):
            ds.append(internal_impostors[i_idx[i]:i_idx[i+1]])
            ds.append(genuine_samples[g_idx[i]:g_idx[i+1]])
        return pd.concat(ds)