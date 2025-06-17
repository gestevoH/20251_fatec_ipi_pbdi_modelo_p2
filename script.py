#REQ 1
# faça os imports que julgar necessários
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

#REQ 2
#essa função deve devolver a base de dados
def ler_base():
  dataset = pd.read_csv(r'dados.csv')
  return dataset

#REQ 3
#essa função recebe a base lida anteriormente
#ela deve devolver uma tupla contendo as features e a classe
def dividir_em_features_e_classe(dataset):
  features = dataset.iloc[:,:-1].values
  classe = dataset.iloc[:,-1].values
  return features, classe

#REQ 4
#essa função recebe as features
#ela deve devolver as features da seguinte forma
#Valores faltantes da coluna "Gastos com pesquisa e desenvolvimento": substituir pela média
#Valores faltantes da coluna "Gastos com administracao": substituir pela mediana
#Valores faltantes da coluna "Gastos com marketing": Substituir por zero
#Valores faltantes da coluna "Estado": Substituir pela moda
def lidar_com_valores_faltantes(features):
  imputer_mean = SimpleImputer(missing_values = np.nan, strategy = "mean")
  features[:, 0:1] = imputer_mean.fit_transform(features[:, 0:1])
  imputer_median = SimpleImputer(missing_values = np.nan, strategy = "median")
  features[:, 1:2] = imputer_median.fit_transform(features[:, 1:2])
  imputer_zero = SimpleImputer(missing_values = np.nan, strategy = "zero")
  features[:, 2:3] = imputer_zero.fit_transform(features[:, 2:3])
  imputer_mode = SimpleImputer(missing_values = np.nan, strategy = "mode")
  features[:, 3:4] = imputer_mode.fit_transform(features[:, 3:4])
  return features

#REQ 5
#essa função recebe as features
#ela deve devolver as features da seguinte forma
#Variável "Estado": Codificar com OneHotEncoding
def codificar_categoricas(features):
  columnTransformer = ColumnTransformer(
    transformers= [('encoder', OneHotEncoder(), [3])],
    remainder= 'passthrough')
  features = np.array(columnTransformer.fit_transform(features))
  return features

#REQ 6
#essa função recebe as features e a classe
#ela deve devolver uma tupla com 4 itens
# features de treinamento, features de teste, classe de treinamento, classe de teste
# a base de treinamento deve ter 75% das instâncias
def obter_bases_de_treinamento_e_teste(features, classe):
  features_treinamento, features_teste, classe_treinamento, classe_teste = train_test_split(
    features, classe, test_size = 0.25, random_state = 1)
  return features_treinamento, features_teste, classe_treinamento, classe_teste

#REQ 7
#essa função recebe as features de treinamento e de teste
#ela deve devolver uma tupla com 2 itens, da seguinte forma
#todas as variáveis normalizadas com o método MinMax
def normalizar(features_treinamento, features_teste):
  min_max_scaler = MinMaxScaler()
  features_treinamento_normalizadas = min_max_scaler.fit_transform(features_treinamento)
  features_teste_normalizadas = min_max_scaler.transform(features_teste)
  return features_treinamento_normalizadas, features_teste_normalizadas

#REQ 8
def vai():
  #chame as suas funções aqui
  #exiba as quatro bases aqui
     pass

vai()