import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix 

class DBCreator():
    '''
    Classe para separar os usuários, dados e variaveis que serão salvas em um banco de dados para utilização do sistema de reconhecimento.
    '''
    def __init__(self, dataset=None):
        '''parameters:
            - dataset: pandas.DataFrame
        '''
        self._dataset = dataset
    
    def create_db(self, user_column=None, n=None, users=None):
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
        for us in users: #para cada usuário no dataset enviado
            temp = self._dataset[self._dataset[user_column] == us] #separar as observações do usuário
            user_information[us] = temp.iloc[:n] #salvar as n primeiras observações no dicionário
            index_used_in_db = index_used_in_db + user_information[us].index.tolist() #guardar os indices utilizados
        
        data_not_used = self._dataset.drop(self._dataset.index[index_used_in_db]) #salvar as observações que não foram utilizadas
        return user_information, data_not_used
        
class FeatureExtractor():
    '''
    Classe que extrai características (traços biométricos) de um conjunto de variáveis.
    '''
    def __init__(self):
        pass
    
    def create_feat(self, dataset=None, method='M2005', ignored_columns=None):
        '''
        parameters:
            - dataset (pandas.Dataframe): banco de dados com informações de usuários
            - method (str): método de extração
            - ignored_columns (list): colunas do dataset que devem ser ignoradas
        
        return:
            - banco de dados (dict) de traços biométricos de cada usuário registrado 
        '''
        biometricFeatures = dict() 
        if method=='M2005':
            biometricFeatures = classification.M2005_FE(dataset=dataset, ignored_columns=ignored_columns)
            return biometricFeatures
        
class DataStream():
    '''
    Class que cria o fluxo de dados utilizado para testes.
    '''
    def __init__(self, impostor_rate=0.5, rate_external_impostor=0.5):
        self.__impostor_rate = impostor_rate # taxa de impostores
        self.__rate_external_impostor = rate_external_impostor # dos impostores no fluxo, define quantos são externos
    
    def create(self, data=None, genuine=None, internal=None, external=None):
        '''
        parameters:
            - data (pandas.Dataframe): amostras para serem usadas no fluxo de dados biométricos
            - genuine (str): id do usuário genuino
            - internal (list): id dos usuários registrados
            - external (list): id dos usuários nao registrados
        
        return:
            - dataStream (pandas.Dataframe): fluxo de dados biométricos
        '''
        genuine_samples = data[data['subject']==genuine] # separando amostras genuínas;
        in_samples = data[data['subject'].isin(internal)] # separando amostras de usuários registrados
        int_imp_samples = in_samples[in_samples['subject'] != genuine] # com exceção do usuário genuino;
        ext_imp_samples = data[data['subject'].isin(external)] # separando amostras de usuários não registrados
        
        # separando o número de amostras que serão utilizadas no fluxo
        #import pdb; pdb.set_trace();
        n_impostor = int((len(genuine_samples) * (1-self.__impostor_rate)) / self.__impostor_rate)
        n_internal_imp = int(n_impostor * (1-self.__rate_external_impostor))
        n_external_imp = n_impostor - n_internal_imp
        
        # enviando amostras aleatórias de cada usuário para a criação do fluxo
        dataStream = sampling.random(genuine_samples=genuine_samples,
                                    internal_samples=int_imp_samples.iloc[:n_internal_imp],
                                    external_samples=ext_imp_samples.iloc[:n_external_imp])
        return dataStream

class sampling():
    '''
    Class que reproduz a forma de amostragem do fluxo de dados
    '''
    def random(genuine_samples=None, internal_samples=None, external_samples=None):
        '''
        parameters:
            - Recebe amostras genuínas e impostoras
        return:
            - dataStream
        '''
        d = [genuine_samples, internal_samples, external_samples]
        dataStream = pd.concat(d) #concatenando os dados
        dataStream = dataStream.sample(frac=1).reset_index(drop=True) #reordenando aleatóriamente
        return dataStream
    
class classification():
    def M2005_FE(dataset=None, ignored_columns=None):
        features = dict()
        for user in dataset.keys():
            usft = dict()
            data = dataset[user]
            try:
                data=data.drop(ignored_columns, axis=1)
            except:
                pass
            for feature in data:
                lower = min(data[feature].mean(), data[feature].median()) * (0.95 - (data[feature].std() / data[feature].mean()))
                upper = max(data[feature].mean(), data[feature].median()) * (1.05 - (data[feature].std() / data[feature].mean()))
                usft[feature] = (lower, upper)
            features[user] = usft
        return features
    
    def M2005(genuineUser=None, dataStream=None, biometricsDatabase=None, decision_threshold=0.00):
        y_true = list()
        y_pred = list()
        list_of_scores = list()
        
        for index, row in dataStream.iterrows():
            if (row['subject'] == genuineUser):
                y_true.append('genuine')
            else:
                y_true.append('impostor')
            match_sum = 0
            previousDimMatched = False
            for dim in biometricsDatabase[genuineUser].keys():
                if (row[dim] < biometricsDatabase[genuineUser][dim][1]) & (row[dim] > biometricsDatabase[genuineUser][dim][0]):
                    if previousDimMatched:
                        match_sum = match_sum + 1.5
                    else:
                        match_sim = match_sum + 1.0
                    previousDimMatched = True
                else:
                    previousDimMatched = False
            max_sum = 1.0 + 1.5 * (len(biometricsDatabase[genuineUser].keys()) -1)
            score = match_sum/max_sum
            if score > decision_threshold:
                y_pred.append('genuine')
            else:
                y_pred.append('impostor')
            list_of_scores.append(score)
        FNMR, FMR, B_acc = classification.report_metrics(y_true=y_true, y_pred=y_pred)
        return FNMR, FMR, B_acc, list_of_scores
    
    def report_metrics(y_true=None, y_pred=None):
        #import pdb; pdb.set_trace();
        cm = confusion_matrix(y_true, y_pred, labels=['genuine','impostor'])
        FNMR = cm[1,0] / (cm[0,0] + cm[1,0])
        FMR = cm[0,1] / (cm[0,1] + cm[1,1])
        B_acc = 1 - ((FNMR + FMR)/2)
        metrics = {'FNMR' : FNMR, 'FMR' : FMR, 'B_acc' : B_acc}
        return FNMR, FMR, B_acc
