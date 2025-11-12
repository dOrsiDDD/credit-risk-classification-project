Modelo de Machine Learning para Classificação de Risco de Crédito

Este projeto apresenta o desenvolvimento de um modelo de machine learning para classificação de risco de crédito. O objetivo é construir um classificador capaz de identificar clientes com alto risco de inadimplência, auxiliando na tomada de decisão para aprovação de produtos de crédito (como empréstimos ou cartões de crédito).

Metodologia

O projeto seguiu um pipeline estruturado de Machine Learning:

1. Carga e Limpeza de Dados

    Os dados foram carregados a partir de arquivos conjunto_de_treinamento.csv e conjunto_de_teste.csv.

    Foi realizado o tratamento de valores ausentes (NaN) em colunas como meses_na_residencia, profissao_companheiro e tipo_residencia. A estratégia de imputação utilizada foi a mediana, uma abordagem robusta para dados com distribuições assimétricas.

2. Pré-processamento e Engenharia de Features

    Normalização: As features numéricas foram normalizadas utilizando o MinMaxScaler do Scikit-learn. Isso coloca todas as features na mesma escala (entre 0 e 1), o que é crucial para o desempenho de algoritmos como KNN e Regressão Logística.

    Engenharia de Features: (Ex: Agrupamento de estados por região, se aplicável, ou outras transformações que você fez).

3. Modelagem e Comparação

Foram treinados e avaliados três algoritmos de classificação distintos para comparar suas performances:

    K-Nearest Neighbors (KNN): Um modelo baseado em instância, sensível à escala dos dados.

    Regressão Logística: Um modelo linear clássico para classificação.

    Random Forest Classifier: Um modelo de ensemble (baseado em árvores de decisão) robusto e que lida bem com relações não-lineares.

4. Validação e Otimização

    Validação Cruzada (K-Fold): Foi utilizada a validação cruzada KFold (com 5 splits) para avaliar a performance dos modelos de forma robusta, garantindo que o resultado não fosse dependente de uma única divisão de treino/teste.

    Otimização de Hiperparâmetros: Para o modelo Random Forest, que apresentou o melhor potencial, foi utilizada a técnica HalvingGridSearchCV. Esta é uma abordagem moderna e eficiente (mais rápida que o GridSearchCV tradicional) para encontrar a melhor combinação de hiperparâmetros (como n_estimators, max_depth, min_samples_leaf, etc.).

Resultados

O modelo Random Forest Classifier apresentou o desempenho mais estável e com maior acurácia média após o processo de validação cruzada e otimização de hiperparâmetros.

O modelo final foi treinado com os parâmetros ótimos encontrados pelo HalvingGridSearchCV, demonstrando ser a solução mais eficaz para este problema de classificação de risco.

Tecnologias Utilizadas

    Python

    Pandas: Para manipulação e limpeza dos dados.

    NumPy: Para operações numéricas.

    Scikit-learn: Para pré-processamento (MinMaxScaler), modelagem (KNeighborsClassifier, LogisticRegression, RandomForestClassifier) e validação (KFold, cross_val_score, HalvingGridSearchCV).

    Matplotlib: Para visualização de dados (utilizado na análise exploratória).

Relatório Técnico Completo

Para uma análise detalhada da metodologia, exploração dos dados (EDA) e justificativa das escolhas técnicas, consulte o relatório técnico completo:
  
