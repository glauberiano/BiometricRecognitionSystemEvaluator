#Automatic Biometric Recognition System

'''
Classe principal
'''
from components import DBCreator
from components import FeatureExtractor
from components import DataStream
from components import classification
import pandas as pd
import IPython.core.display as ipd
from sklearn.model_selection import KFold

class RecognitionSystem():
    def __init__(self, dataset=None, method='M2005', adaptative=False):
        #self.dataset = dataset
        self.method = method
        self.adaptative = adaptative
        
        self.dict_params = {'method' : method,
                           'adaptativa' : adaptative}
        
    def fit(self, dataset=None, user_column=None, n_samples_features=None, impostor_rate=None, rate_external_impostor=None, decision_threshold=None):
        #import pdb;pdb.set_trace();
        CMU = DBCreator(dataset=dataset)
        FE = FeatureExtractor()
        usuarios = pd.DataFrame(dataset[user_column].unique(), columns=[user_column])
        kfold = KFold(n_splits=5)
        splits = kfold.split(usuarios)
        self.metrics = list()
        counter_folds = 0

        for internal_idx, external_idx in splits:
            u_reg = usuarios.iloc[internal_idx][user_column]
            u_nao_reg = usuarios.iloc[external_idx][user_column]
            usersDatabase, samples = CMU.create_db(user_column=user_column, n=n_samples_features, users=u_reg)
            usersBiometrics = FE.create_feat(dataset=usersDatabase, method=self.method,
                                             ignored_columns=user_column)
            counter=0
            counter_folds += 1
            for usuario in u_reg:
                counter += 1
                ipd.clear_output(wait=True)
                print('Split ' +str(counter_folds) + ' de ' + str(kfold.n_splits) + '\n' +   'Usuário ' + str(counter) + ' de ' + str(len(u_reg)))
                ds = DataStream(impostor_rate=impostor_rate, rate_external_impostor = rate_external_impostor)
                dataStream = ds.create(data=samples, genuine=usuario, internal=u_reg, external=u_nao_reg)
                FMR, FMNR, B_acc, _ = classification.M2005(genuineUser=usuario, dataStream=dataStream, biometricsDatabase=usersBiometrics)
                self.metrics.append((FMR, FMNR, B_acc))
        
    def summary(self):
        pass
    