# Trabalho 1 - Diego d'Orsi Duarte
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_halving_search_cv 
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score 
from sklearn.model_selection import KFold
from sklearn.model_selection import LeaveOneOut
from scipy import stats


dados = pd.read_csv('conjunto_de_treinamento.csv', delimiter=',', decimal='.')
dados_teste = pd.read_csv('conjunto_de_teste.csv', delimiter=',', decimal='.')

dados['meses_na_residencia'] = dados['meses_na_residencia'].fillna(dados['meses_na_residencia'].median())
dados['profissao_companheiro'] = dados['profissao_companheiro'].fillna(dados['profissao_companheiro'].median())
dados['tipo_residencia'] = dados['tipo_residencia'].fillna(dados['tipo_residencia'].median())

# dados['ocupacao'] = dados['ocupacao'].fillna('XX')
# dados.loc[dados['renda_mensal_regular'] > 1000 ,'valor_patrimonio_pessoal'] = 1
# dados.loc[dados['renda_mensal_regular'] < 1000 ,'valor_patrimonio_pessoal'] = 0
# dados.loc[dados['valor_patrimonio_pessoal']  ,'valor_patrimonio_pessoal'] = dados.loc[dados['valor_patrimonio_pessoal']  ,'valor_patrimonio_pessoal']*8
# dados.loc[dados['valor_patrimonio_pessoal'] < 50000 ,'valor_patrimonio_pessoal'] = 0

dados_teste['meses_na_residencia'] = dados_teste['meses_na_residencia'].fillna(dados_teste['meses_na_residencia'].median())
dados_teste['profissao_companheiro'] = dados_teste['profissao_companheiro'].fillna(dados_teste['profissao_companheiro'].median())
dados_teste['tipo_residencia'] = dados_teste['tipo_residencia'].fillna(dados_teste['tipo_residencia'].median())

# dados_teste['ocupacao'] = dados_teste['ocupacao'].fillna('XX')
# dados.loc[dados['renda_mensal_regular'] > 1e4 ,'renda_mensal_regular'] = 1e4
# dados.loc[dados['renda_extra'] > 1e4 ,'renda_extra'] = 5000
# dados.loc[dados['valor_patrimonio_pessoal'] > 1e4 ,'valor_patrimonio_pessoal'] = 5000
# dados_teste.loc[dados_teste['renda_mensal_regular'] > 1e4 ,'renda_mensal_regular'] = 1e4
# dados_teste.loc[dados_teste['renda_extra'] > 1e4 ,'renda_extra'] = 5000
# dados_teste.loc[dados_teste['valor_patrimonio_pessoal'] > 1e4 ,'valor_patrimonio_pessoal'] = 5000
# dados.loc[dados['renda_extra'] > 0 ,'renda_extra'] = 1
# dados_teste.loc[dados_teste['renda_extra'] >0 ,'renda_extra'] = 1
# dados.loc[dados['qtde_contas_bancarias'] > 0 ,'qtde_contas_bancarias'] = 1
# dados_teste.loc[dados_teste['qtde_contas_bancarias'] >0 ,'qtde_contas_bancarias'] = 1
# dados_teste.loc[dados_teste['renda_mensal_regular'] > 1000 ,'renda_mensal_regular'] = 1
# dados_teste.loc[dados_teste['renda_mensal_regular'] < 1000 ,'renda_mensal_regular'] = 0
# dados_teste.loc[dados_teste['valor_patrimonio_pessoal']  ,'valor_patrimonio_pessoal'] = dados_teste.loc[dados_teste['valor_patrimonio_pessoal']  ,'valor_patrimonio_pessoal']*8
# dados_teste.loc[dados_teste['valor_patrimonio_pessoal'] < 50000 ,'valor_patrimonio_pessoal'] = 0

#regioes = {'RJ':'altoidh',
#         'ES':'medioidh',
#         'MG':'medioidh',
#         'SP':'altoidh',
#         'MT':'medioidh',
#         'MS':'medioidh',
#         'DF':'altoidh',
#         'GO':'medioidh',
#         'RS':'medioidh',
#         'PR':'altoidh',
#         'SC':'altoidh',
#         'SE':'baixoidh',
#         'AL':'baixoidh',
#         'BA':'baixoidh',
#         'PE':'baixoidh',
#         'RN':'baixoidh',
#         'PB':'baixoidh',
#         'MA':'baixoidh',
#         'PI':'baixoidh',
#         'CE':'baixoidh',
#         'AP':'baixoidh',
#         'RR':'baixoidh',
#         'PA':'baixoidh',
#         'RO':'baixoidh',
#         'TO':'baixoidh',
#         'AC':'baixoidh',
#         'AM':'baixoidh',}

#dados = dados.replace(regioes)
#dados_teste = dados_teste.replace(regioes)

#regioes = {'RJ':'49.57',
#         'ES':'41.49',
#         'MG':'38.31',
#         'SP':'43.45',
#         'MT':'47.68',
#         'MS':'45.27',
#         'DF':'49.16',
#         'GO':'40.36',
#         'RS':'36.26',
#         'PR':'39.71',
#         'SC':'34.76',
#         'SE':'41.03',
#         'AL':'36.84',
#         'BA':'37.2',
#         'PE':'41.52',
#         'RN':'39.93',
#         'PB':'37.91',
#         'MA':'51.75',
#         'PI':'33.49',
#         'CE':'40.32',
#         'AP':'49.3',
#         'RR':'47.46',
#         'PA':'37.12',
#         'RO':'42.34',
#         'TO':'42.2',
#         'AC':'44.13',
#         'AM':'51.75',
#         ' ': '40',}

#dados = dados.replace(regioes)
#dados_teste = dados_teste.replace(regioes)

regioes = {'RJ':'sudeste',
         'ES':'sudeste',
         'MG':'sudeste',
         'SP':'sudeste',
         'MT':'centroOeste',
         'MS':'centroOeste',
         'DF':'centroOeste',
         'GO':'centroOeste',
         'RS':'sul',
         'PR':'sul',
         'SC':'sul',
         'SE':'nordeste',
         'AL':'nordeste',
         'BA':'nordeste',
         'PE':'nordeste',
         'RN':'nordeste',
         'PB':'nordeste',
         'MA':'nordeste',
         'PI':'nordeste',
         'CE':'nordeste',
         'AP':'norte',
         'RR':'norte',
         'PA':'norte',
         'RO':'norte',
         'TO':'norte',
         'AC':'norte',
         'AM':'norte',}

dados = dados.replace(regioes)
dados_teste = dados_teste.replace(regioes)

dados = dados.fillna(0)
dados_teste  = dados_teste.fillna(0)

variaveis_categoricas = [ x for x in dados.columns if dados[x].dtype == 'object']

for v in variaveis_categoricas:
  print('\n%15s: '%v, "%4d categorias" % len(dados[v].unique()))
  print(dados[v].unique(),'\n')
  
# Descartando dados de alta cardinalidade
dados = dados.drop (['codigo_area_telefone_residencial', 'codigo_area_telefone_trabalho', 'possui_telefone_celular',
                     'id_solicitante','estado_onde_trabalha','qtde_contas_bancarias_especiais','estado_onde_nasceu','grau_instrucao_companheiro','local_onde_trabalha', 'grau_instrucao'], axis = 1)


# dados = dados.drop(['meses_no_trabalho'],axis=1)
# dados = dados.drop(['forma_envio_solicitacao'],axis=1)
dados_teste = dados_teste.drop (['codigo_area_telefone_residencial', 'codigo_area_telefone_trabalho', 'possui_telefone_celular',
                     'id_solicitante','estado_onde_trabalha','estado_onde_nasceu','qtde_contas_bancarias_especiais','grau_instrucao_companheiro','local_onde_trabalha','grau_instrucao'], axis = 1)


# dados_teste = dados_teste.drop(['meses_no_trabalho'],axis=1)
# dados_teste = dados_teste.drop(['estado_onde_reside'],axis=1)
# dados_teste = dados_teste.drop(['forma_envio_solicitacao'],axis=1)


dados['sexo'] = dados['sexo'].replace({' ': 'N'})  # Substituir espaço por "N" (ou outro valor relevante)
dados_teste['sexo'] = dados_teste['sexo'].replace({' ': 'N'})

# colocando 0 onde há espaço em branco
dados = pd.get_dummies(dados,columns = ['forma_envio_solicitacao'])
dados = pd.get_dummies(dados,columns = ['estado_onde_reside'])
dados = pd.get_dummies(dados,columns = ['sexo'])
# dados = pd.get_dummies(dados,columns = ['produto_solicitado'])
# dados = pd.get_dummies(dados,columns = ['ocupacao'])
# dados = pd.get_dummies(dados,columns = ['tipo_residencia'])
# dados = pd.get_dummies(dados,columns = ['local_onde_reside'])

dados_teste = pd.get_dummies(dados_teste,columns = ['forma_envio_solicitacao'])
dados_teste = pd.get_dummies(dados_teste,columns = ['estado_onde_reside'])
dados_teste = pd.get_dummies(dados_teste,columns = ['sexo'])
# dados_teste = pd.get_dummies(dados_teste,columns = ['produto_solicitado'])
# dados_teste = pd.get_dummies(dados_teste,columns = ['ocupacao'])
# dados_teste = pd.get_dummies(dados_teste,columns = ['tipo_residencia'])
# dados_teste = pd.get_dummies(dados_teste,columns = ['local_onde_reside'])

# Adicionar colunas ausentes no conjunto de teste
for col in dados.columns:
    if col not in dados_teste.columns:
        dados_teste[col] = 0

# Garantir que a ordem das colunas seja a mesma
dados_teste = dados_teste[dados.columns]

# Binarizando variáveis
binarizador = LabelBinarizer()
for x in ['vinculo_formal_com_empresa', 'possui_telefone_trabalho','possui_telefone_residencial']:
  dados[x] = binarizador.fit_transform(dados[x])
  dados_teste[x] = binarizador.fit_transform(dados_teste[x])

dados.columns.tolist()

atributosSelecionados = [
 'id_solicitante',
 'produto_solicitado',
 'dia_vencimento',
 'forma_envio_solicitacao',
 'tipo_endereco',
 'sexo',
 'idade',
 'estado_civil',
 'qtde_dependentes',
 'grau_instrucao',
 'nacionalidade',
 'estado_onde_nasceu',
 'estado_onde_reside',
 'possui_telefone_residencial',
 'codigo_area_telefone_residencial',
 'tipo_residencia',
 'meses_na_residencia',
 'possui_telefone_celular',
 'possui_email',
 'renda_mensal_regular',
 'renda_extra',
 'possui_cartao_visa',
 'possui_cartao_mastercard',
 'possui_cartao_diners',
 'possui_cartao_amex',
 'possui_outros_cartoes',
 'qtde_contas_bancarias',
 'qtde_contas_bancarias_especiais',
 'valor_patrimonio_pessoal',
 'possui_carro',
 'vinculo_formal_com_empresa',
 'estado_onde_trabalha',
 'possui_telefone_trabalho',
 'codigo_area_telefone_trabalho',
 'meses_no_trabalho',
 'profissao',
 'ocupacao',
 'profissao_companheiro',
 'local_onde_reside',
 'local_onde_trabalha',
 ]

alvo = 'inadimplente'

dados_embaralhados = dados.sample(frac=1,random_state=12345) 
dados_teste_embaralhados =dados_teste.sample(frac=1,random_state=12)

x = dados_embaralhados.loc[:,dados_embaralhados.columns != 'inadimplente'].values
y = dados_embaralhados.loc[:,dados_embaralhados.columns == 'inadimplente'].values

x_testagem = dados_teste.loc[:,dados_teste.columns != 'inadimplente'].values
y_testagem = dados_teste.loc[:,dados_teste.columns == 'inadimplente'].values

#treino
x_treino = x[:17000,:-1]
y_treino= y[:17000,-1].ravel()

#teste
x_teste = x[17000:,:-1]
y_teste = y[17000:,-1].ravel()

scaler = MinMaxScaler()
scaler.fit_transform(x_treino)
x_treino = scaler.transform(x_treino)
x_teste = scaler.transform(x_teste)

#treino
x_treino_final = x
y_treino_final = y.ravel()

#teste
x_teste_final = x_testagem

for k in range(1, 4):
  classificador = KNeighborsClassifier(n_neighbors=k)
  classificador = classificador.fit(x_treino, y_treino)

  y_resposta_treino = classificador.predict(x_treino)
  y_resposta_teste = classificador.predict(x_teste)

  acuracia_treino = np.sum(y_resposta_treino == y_treino) / len(y_treino)
  acuracia_teste = np.sum(y_resposta_teste == y_teste) / len(y_teste)

  print(
      "%3d" % k,
      "%6.1f" % (100 * acuracia_treino),
      "%6.1f" % (100 * acuracia_teste)
  )

classificador = KNeighborsClassifier(n_neighbors=32)
classificador = classificador.fit(x_treino_final,y_treino_final)
y_resposta_treino = classificador.predict(x_treino_final)
y_resposta_testeKNN = classificador.predict(x_teste_final)   

for k in range(-1,10):
  c = pow(10,k)
  classificador = LogisticRegression(penalty = 'l2', C = c, max_iter=1000)
  classificador = classificador.fit(x_treino,y_treino)

  y_resposta_treino = classificador.predict(x_treino)
  y_resposta_teste  = classificador.predict(x_teste)

  acuracia_treino = sum(y_resposta_treino==y_treino)/len(y_treino)
  acuracia_teste  = sum(y_resposta_teste ==y_teste) /len(y_teste)

  print(
    "%14.6f"%c,
    "%6.1f" % (100*acuracia_treino),
    "%6.1f" % (100*acuracia_teste)
    )
  
c = 1
classificador = LogisticRegression(penalty = 'l2', C = c)
classificador = classificador.fit(x_treino_final,y_treino_final)
y_resposta_treino = classificador.predict(x_treino_final)
y_resposta_testeRegressaoLogistica = classificador.predict(x_teste_final)

x = dados_embaralhados.loc[:,dados_embaralhados.columns != 'inadimplente'].values # redefinindo x e y
y = dados_embaralhados.loc[:,dados_embaralhados.columns == 'inadimplente'].values

x_treino = x
y_treino = y.ravel()

param_grid = { 
  'n_estimators': [100, 200],
  'criterion': ['entropy','gini'],
  'max_depth': [2,5,6,8,10,12],
  'min_samples_split': [0.001, 0.01,0.1],
  'min_samples_leaf': [0.01, 1,3],
  'max_features': [None,7, "log2", 'sqrt']
}

parametrosTeste = {
  'n_estimators': [100, 200],
  'criterion': ['entropy','gini'],
  'max_depth': [2,5,6,8,10,12],
  'min_samples_split': [0.001, 0.01,0.1],
  'min_samples_leaf': [0.01, 1,3],
  'max_features': [None,7, "log2", 'sqrt']
}

def searchGrid(parametrosTeste):
  base_estimator = RandomForestClassifier(random_state = 0, oob_score = True)

  with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    sh = HalvingGridSearchCV(base_estimator, parametrosTeste,cv=2,n_jobs=4).fit(x_treino, y_treino)
    melhoresParametros = sh.best_params_

  return melhoresParametros


print("Iniciando validação cruzada")
for n in range(1,7):
  model = RandomForestClassifier(criterion='gini', max_depth=15, max_features=10, min_samples_leaf=2, min_samples_split=0.01, n_estimators= n)
  model.fit(x_treino,y_treino)

  kfold  = KFold(n_splits=5, shuffle=True) 
  resultado = cross_val_score(model, x_treino, y_treino, cv = kfold, scoring='accuracy')
  print("n = %i"%n)
  print("K-Fold Scores: {0}".format(resultado))
  print("Media da acuracia por validação cruzada K-Fold: {0}".format(resultado.mean()))

  print(f"Conclusão da validação cruzada {n}")


print(f"Diretório atual: {os.getcwd()}")

  # Código para busca de melhores hiperparâmetros utilizando HalvingGridSearchCV
parametrosTeste = {
  'n_estimators': [100],
  'criterion': ['entropy'],
  'max_depth': [5],
  'min_samples_split': [0.01],
  'min_samples_leaf': [1],
  'max_features': [None]
}


def searchGrid(parametrosTeste):
    print("Iniciando o HalvingGridSearchCV...")
    base_estimator = RandomForestClassifier(random_state=0, oob_score=True)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sh = HalvingGridSearchCV(base_estimator, parametrosTeste, cv=2, n_jobs=4)
        print("A busca começou...")
        sh.fit(x_treino, y_treino)
        print("A busca foi concluída.")
        melhoresParametros = sh.best_params_

    return melhoresParametros

# Chamar a função para buscar os melhores parâmetros
melhoresParametros = searchGrid(parametrosTeste)
print("Melhores Parâmetros do Grid Search:", melhoresParametros)
  
model = RandomForestClassifier(criterion='gini', max_depth=15, max_features=10, min_samples_leaf=5, min_samples_split=0.001, n_estimators= 200)
model.fit(x_treino,y_treino)
kfold  = KFold(n_splits=5, shuffle=True) 
resultado = cross_val_score(model, x_treino, y_treino, cv = kfold, scoring='accuracy')
print("K-Fold Scores: {0}".format(resultado))
print("Media de acuracia por validação cruzada K-Fold: {0}".format(resultado.mean()))

model = RandomForestClassifier(criterion='gini', max_depth=15, max_features=10, min_samples_leaf=5, min_samples_split=0.001, n_estimators= 200)
model.fit(x_treino_final,y_treino_final)

y_random_forest = model.predict(x_teste_final)

print(y_random_forest)


print(f"O Diretório atual: {os.getcwd()}")
aux = pd.read_csv('conjunto_de_teste.csv')
minharesposta_random_forestClassificador21 = pd.DataFrame({'id_solicitante':aux.pop('id_solicitante'), 'inadimplente':np.squeeze(np.int16(y_random_forest))})
print("Gerando o arquivo de respostas...")
minharesposta_random_forestClassificador21.to_csv("resposta_random_forestClassificador.csv", index=False)
print("Arquivo gerado com sucesso!")

# Trabalho 1 - Diego d'Orsi Duarte
