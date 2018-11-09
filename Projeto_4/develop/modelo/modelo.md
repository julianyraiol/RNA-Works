
## Projeto Prático 4

**Universidade do Estado do Amazonas**  
**Escola Superior de Tecnologia**  
**Professora:** Elloá B. Guedes  
**Alunos:** Juliany Raiol, Raí Soledade, Richardson Souza  
**Disciplina:** Redes Neurais Artificiais

## Aprendizado de Máquina com tarefa de classificação aplicado no dataset  de variedades de trigo

### Introdução

Três variedades de trigo (Kama, Rosa e Canadian) possuem sementes muito parecidas,
entretanto diferentes. Um grupo de pesquisadores poloneses coletou 70 amostras de cada
tipo e, usando uma técnica particular de raio-X, coletou medidas geométricas destas
sementes, a citar: área, perímetro, compactude, comprimento, largura, coeficiente de
assimetria e comprimento do sulco da semente.



```python
# Módulos utilizados no projeto

import pandas as pd
import numpy as np
import random
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
% matplotlib inline
sns.set()
import warnings
warnings.filterwarnings('ignore')
```

Leitura do dataset


```python
names = ["Area", "Perimeter", "Compactness", "Length", "Width", "Asymmetry", "Groove", "Seed"]

df = pd.read_csv('../../data/seeds_dataset.txt', delim_whitespace=True, names = names)
```

### Treinamento

X = atributos preditores, y = atributo alvo


```python
X = df.drop('Seed', axis=1)
y = df['Seed']
```

Definição dos parâmetros de taxa de aprendizado, neurônios na camada de entrada e saída, funções de ativação e o alfa da regra da pirâmide geométrica utilizada para calcular a quantidade de neurônios nas camadas ocultas


```python
rate  = [0.01, 0.05]
alpha = [0.5, 2, 3]

neuron_out = 2
neuron_ini = 7
activation_functions = ['identity', 'logistic', 'tanh', 'relu']
```

Cálculo da quantidade de neurônios nas camadas ocultas utilizando a regra da pirâmide geométrica.

\begin{align}
\dot{N_{h}} & = \alpha.\sqrt{\dot{N_{i} . \dot{N_{o}}}}
\end{align}

<strong> Nh </strong> é o número de neurônios ocultos (a serem distribuídos em uma ou duas camadas
ocultas)

<strong>Ni</strong> é o número de neurônios na camada de entrada

<strong>No</strong> é o número de neurônios
na camada de saída.


```python
n = []
for a in alpha:
    n.append(int( a * np.sqrt((neuron_ini*neuron_out))))
print("Quantidade de neurônios nas camadas ocultas a serem testadas respectivamente: ", n)
```

    Quantidade de neurônios nas camadas ocultas a serem testadas respectivamente:  [1, 7, 11]


Parâmetro que define uma série de combinações de neurônios distribuídos em 1 ou 2 camadas, de acordo com a quantidade de neurônios calculada anteriormente


```python
hidden_layer = [(1,), (7,),(1,6),(2,5),(3,4), (11,),(1,10),(2,9),(3,8),(4,7),(5,6)]
```

Definição dos parâmetros para inicialização dos modelos


```python
parameters = dict([
                ('hidden_layer_sizes', hidden_layer),
                ('learning_rate_init', rate),
                ('activation', activation_functions)
            ])
```

No treinamento das redes neurais, o solver escolhido foi o LBFGS pois ele é o que se comporta melhor com datasets com poucos dados. 


```python
clf = GridSearchCV(MLPClassifier(solver='lbfgs'), parameters, iid=True, cv = 3, return_train_score=True)
clf.fit(X, y)
```




    GridSearchCV(cv=3, error_score='raise',
           estimator=MLPClassifier(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,
           beta_2=0.999, early_stopping=False, epsilon=1e-08,
           hidden_layer_sizes=(100,), learning_rate='constant',
           learning_rate_init=0.001, max_iter=200, momentum=0.9,
           nesterovs_momentum=True, power_t=0.5, random_state=None,
           shuffle=True, solver='lbfgs', tol=0.0001, validation_fraction=0.1,
           verbose=False, warm_start=False),
           fit_params=None, iid=True, n_jobs=1,
           param_grid={'hidden_layer_sizes': [(1,), (7,), (1, 6), (2, 5), (3, 4), (11,), (1, 10), (2, 9), (3, 8), (4, 7), (5, 6)], 'learning_rate_init': [0.01, 0.05], 'activation': ['identity', 'logistic', 'tanh', 'relu']},
           pre_dispatch='2*n_jobs', refit=True, return_train_score=True,
           scoring=None, verbose=0)



Listagem de todas as redes neurais geradas pelo GridSearchCV, com k-fold de tamanho 3.


```python
results = pd.DataFrame.from_dict(clf.cv_results_)
results
```




<div>

<table class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mean_fit_time</th>
      <th>std_fit_time</th>
      <th>mean_score_time</th>
      <th>std_score_time</th>
      <th>param_activation</th>
      <th>param_hidden_layer_sizes</th>
      <th>param_learning_rate_init</th>
      <th>params</th>
      <th>split0_test_score</th>
      <th>split1_test_score</th>
      <th>split2_test_score</th>
      <th>mean_test_score</th>
      <th>std_test_score</th>
      <th>rank_test_score</th>
      <th>split0_train_score</th>
      <th>split1_train_score</th>
      <th>split2_train_score</th>
      <th>mean_train_score</th>
      <th>std_train_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.135941</td>
      <td>0.141183</td>
      <td>0.000357</td>
      <td>5.463353e-05</td>
      <td>identity</td>
      <td>(1,)</td>
      <td>0.01</td>
      <td>{'activation': 'identity', 'hidden_layer_sizes...</td>
      <td>0.875000</td>
      <td>0.913043</td>
      <td>0.753623</td>
      <td>0.847619</td>
      <td>0.067576</td>
      <td>33</td>
      <td>0.847826</td>
      <td>0.851064</td>
      <td>0.914894</td>
      <td>0.871261</td>
      <td>0.030881</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.036299</td>
      <td>0.005218</td>
      <td>0.000309</td>
      <td>9.381164e-06</td>
      <td>identity</td>
      <td>(1,)</td>
      <td>0.05</td>
      <td>{'activation': 'identity', 'hidden_layer_sizes...</td>
      <td>0.861111</td>
      <td>0.913043</td>
      <td>0.797101</td>
      <td>0.857143</td>
      <td>0.047081</td>
      <td>29</td>
      <td>0.891304</td>
      <td>0.851064</td>
      <td>0.921986</td>
      <td>0.888118</td>
      <td>0.029041</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.047110</td>
      <td>0.022264</td>
      <td>0.000334</td>
      <td>1.857014e-06</td>
      <td>identity</td>
      <td>(7,)</td>
      <td>0.01</td>
      <td>{'activation': 'identity', 'hidden_layer_sizes...</td>
      <td>0.972222</td>
      <td>0.956522</td>
      <td>0.826087</td>
      <td>0.919048</td>
      <td>0.065347</td>
      <td>12</td>
      <td>0.978261</td>
      <td>0.971631</td>
      <td>1.000000</td>
      <td>0.983297</td>
      <td>0.012117</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.043567</td>
      <td>0.003085</td>
      <td>0.000365</td>
      <td>7.439279e-05</td>
      <td>identity</td>
      <td>(7,)</td>
      <td>0.05</td>
      <td>{'activation': 'identity', 'hidden_layer_sizes...</td>
      <td>0.972222</td>
      <td>0.985507</td>
      <td>0.840580</td>
      <td>0.933333</td>
      <td>0.065113</td>
      <td>1</td>
      <td>0.985507</td>
      <td>0.964539</td>
      <td>1.000000</td>
      <td>0.983349</td>
      <td>0.014557</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.054199</td>
      <td>0.002430</td>
      <td>0.000350</td>
      <td>1.644954e-05</td>
      <td>identity</td>
      <td>(1, 6)</td>
      <td>0.01</td>
      <td>{'activation': 'identity', 'hidden_layer_sizes...</td>
      <td>0.875000</td>
      <td>0.913043</td>
      <td>0.753623</td>
      <td>0.847619</td>
      <td>0.067576</td>
      <td>33</td>
      <td>0.891304</td>
      <td>0.851064</td>
      <td>0.914894</td>
      <td>0.885754</td>
      <td>0.026352</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.054986</td>
      <td>0.000695</td>
      <td>0.000330</td>
      <td>1.191351e-05</td>
      <td>identity</td>
      <td>(1, 6)</td>
      <td>0.05</td>
      <td>{'activation': 'identity', 'hidden_layer_sizes...</td>
      <td>0.875000</td>
      <td>0.942029</td>
      <td>0.753623</td>
      <td>0.857143</td>
      <td>0.077447</td>
      <td>29</td>
      <td>0.855072</td>
      <td>0.851064</td>
      <td>0.914894</td>
      <td>0.873677</td>
      <td>0.029191</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.055007</td>
      <td>0.000889</td>
      <td>0.000353</td>
      <td>2.355965e-05</td>
      <td>identity</td>
      <td>(2, 5)</td>
      <td>0.01</td>
      <td>{'activation': 'identity', 'hidden_layer_sizes...</td>
      <td>0.958333</td>
      <td>0.956522</td>
      <td>0.710145</td>
      <td>0.876190</td>
      <td>0.116159</td>
      <td>25</td>
      <td>0.985507</td>
      <td>0.914894</td>
      <td>0.957447</td>
      <td>0.952616</td>
      <td>0.029030</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.050661</td>
      <td>0.002991</td>
      <td>0.000312</td>
      <td>5.388940e-06</td>
      <td>identity</td>
      <td>(2, 5)</td>
      <td>0.05</td>
      <td>{'activation': 'identity', 'hidden_layer_sizes...</td>
      <td>0.972222</td>
      <td>0.971014</td>
      <td>0.826087</td>
      <td>0.923810</td>
      <td>0.068363</td>
      <td>5</td>
      <td>0.978261</td>
      <td>0.971631</td>
      <td>1.000000</td>
      <td>0.983297</td>
      <td>0.012117</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.052154</td>
      <td>0.000413</td>
      <td>0.000348</td>
      <td>4.086497e-05</td>
      <td>identity</td>
      <td>(3, 4)</td>
      <td>0.01</td>
      <td>{'activation': 'identity', 'hidden_layer_sizes...</td>
      <td>0.958333</td>
      <td>0.971014</td>
      <td>0.840580</td>
      <td>0.923810</td>
      <td>0.058454</td>
      <td>5</td>
      <td>0.978261</td>
      <td>0.957447</td>
      <td>1.000000</td>
      <td>0.978569</td>
      <td>0.017374</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.052812</td>
      <td>0.000774</td>
      <td>0.000324</td>
      <td>2.219329e-05</td>
      <td>identity</td>
      <td>(3, 4)</td>
      <td>0.05</td>
      <td>{'activation': 'identity', 'hidden_layer_sizes...</td>
      <td>0.972222</td>
      <td>0.985507</td>
      <td>0.739130</td>
      <td>0.900000</td>
      <td>0.112667</td>
      <td>18</td>
      <td>0.978261</td>
      <td>0.957447</td>
      <td>0.992908</td>
      <td>0.976205</td>
      <td>0.014550</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.043776</td>
      <td>0.003893</td>
      <td>0.000318</td>
      <td>1.590212e-05</td>
      <td>identity</td>
      <td>(11,)</td>
      <td>0.01</td>
      <td>{'activation': 'identity', 'hidden_layer_sizes...</td>
      <td>0.972222</td>
      <td>0.971014</td>
      <td>0.840580</td>
      <td>0.928571</td>
      <td>0.061556</td>
      <td>3</td>
      <td>0.992754</td>
      <td>0.964539</td>
      <td>1.000000</td>
      <td>0.985764</td>
      <td>0.015297</td>
    </tr>
    <tr>
      <th>11</th>
      <td>0.037431</td>
      <td>0.012556</td>
      <td>0.000316</td>
      <td>1.455764e-05</td>
      <td>identity</td>
      <td>(11,)</td>
      <td>0.05</td>
      <td>{'activation': 'identity', 'hidden_layer_sizes...</td>
      <td>0.972222</td>
      <td>0.985507</td>
      <td>0.840580</td>
      <td>0.933333</td>
      <td>0.065113</td>
      <td>1</td>
      <td>0.978261</td>
      <td>0.964539</td>
      <td>1.000000</td>
      <td>0.980933</td>
      <td>0.014600</td>
    </tr>
    <tr>
      <th>12</th>
      <td>0.054175</td>
      <td>0.004356</td>
      <td>0.000337</td>
      <td>1.094362e-05</td>
      <td>identity</td>
      <td>(1, 10)</td>
      <td>0.01</td>
      <td>{'activation': 'identity', 'hidden_layer_sizes...</td>
      <td>0.861111</td>
      <td>0.927536</td>
      <td>0.768116</td>
      <td>0.852381</td>
      <td>0.064923</td>
      <td>31</td>
      <td>0.898551</td>
      <td>0.843972</td>
      <td>0.914894</td>
      <td>0.885805</td>
      <td>0.030324</td>
    </tr>
    <tr>
      <th>13</th>
      <td>0.050294</td>
      <td>0.009735</td>
      <td>0.000322</td>
      <td>4.536223e-06</td>
      <td>identity</td>
      <td>(1, 10)</td>
      <td>0.05</td>
      <td>{'activation': 'identity', 'hidden_layer_sizes...</td>
      <td>0.861111</td>
      <td>0.913043</td>
      <td>0.811594</td>
      <td>0.861905</td>
      <td>0.041124</td>
      <td>27</td>
      <td>0.891304</td>
      <td>0.851064</td>
      <td>0.936170</td>
      <td>0.892846</td>
      <td>0.034762</td>
    </tr>
    <tr>
      <th>14</th>
      <td>0.055419</td>
      <td>0.000129</td>
      <td>0.000325</td>
      <td>9.246899e-06</td>
      <td>identity</td>
      <td>(2, 9)</td>
      <td>0.01</td>
      <td>{'activation': 'identity', 'hidden_layer_sizes...</td>
      <td>0.972222</td>
      <td>0.971014</td>
      <td>0.739130</td>
      <td>0.895238</td>
      <td>0.109205</td>
      <td>19</td>
      <td>0.934783</td>
      <td>0.964539</td>
      <td>0.992908</td>
      <td>0.964076</td>
      <td>0.023732</td>
    </tr>
    <tr>
      <th>15</th>
      <td>0.053975</td>
      <td>0.000550</td>
      <td>0.000331</td>
      <td>1.678026e-05</td>
      <td>identity</td>
      <td>(2, 9)</td>
      <td>0.05</td>
      <td>{'activation': 'identity', 'hidden_layer_sizes...</td>
      <td>0.944444</td>
      <td>0.985507</td>
      <td>0.840580</td>
      <td>0.923810</td>
      <td>0.060604</td>
      <td>5</td>
      <td>0.978261</td>
      <td>0.964539</td>
      <td>1.000000</td>
      <td>0.980933</td>
      <td>0.014600</td>
    </tr>
    <tr>
      <th>16</th>
      <td>0.053100</td>
      <td>0.000383</td>
      <td>0.000315</td>
      <td>5.745179e-06</td>
      <td>identity</td>
      <td>(3, 8)</td>
      <td>0.01</td>
      <td>{'activation': 'identity', 'hidden_layer_sizes...</td>
      <td>0.972222</td>
      <td>0.956522</td>
      <td>0.840580</td>
      <td>0.923810</td>
      <td>0.058577</td>
      <td>5</td>
      <td>0.971014</td>
      <td>0.964539</td>
      <td>1.000000</td>
      <td>0.978518</td>
      <td>0.015419</td>
    </tr>
    <tr>
      <th>17</th>
      <td>0.053272</td>
      <td>0.000345</td>
      <td>0.000356</td>
      <td>5.345127e-05</td>
      <td>identity</td>
      <td>(3, 8)</td>
      <td>0.05</td>
      <td>{'activation': 'identity', 'hidden_layer_sizes...</td>
      <td>0.972222</td>
      <td>0.971014</td>
      <td>0.826087</td>
      <td>0.923810</td>
      <td>0.068363</td>
      <td>5</td>
      <td>0.985507</td>
      <td>0.971631</td>
      <td>1.000000</td>
      <td>0.985713</td>
      <td>0.011582</td>
    </tr>
    <tr>
      <th>18</th>
      <td>0.046115</td>
      <td>0.012787</td>
      <td>0.000328</td>
      <td>1.537918e-05</td>
      <td>identity</td>
      <td>(4, 7)</td>
      <td>0.01</td>
      <td>{'activation': 'identity', 'hidden_layer_sizes...</td>
      <td>0.972222</td>
      <td>0.971014</td>
      <td>0.826087</td>
      <td>0.923810</td>
      <td>0.068363</td>
      <td>5</td>
      <td>0.992754</td>
      <td>0.964539</td>
      <td>1.000000</td>
      <td>0.985764</td>
      <td>0.015297</td>
    </tr>
    <tr>
      <th>19</th>
      <td>0.054600</td>
      <td>0.000273</td>
      <td>0.000327</td>
      <td>1.389116e-05</td>
      <td>identity</td>
      <td>(4, 7)</td>
      <td>0.05</td>
      <td>{'activation': 'identity', 'hidden_layer_sizes...</td>
      <td>0.972222</td>
      <td>0.985507</td>
      <td>0.753623</td>
      <td>0.904762</td>
      <td>0.105868</td>
      <td>15</td>
      <td>0.992754</td>
      <td>0.964539</td>
      <td>1.000000</td>
      <td>0.985764</td>
      <td>0.015297</td>
    </tr>
    <tr>
      <th>20</th>
      <td>0.054023</td>
      <td>0.000832</td>
      <td>0.000325</td>
      <td>2.444175e-05</td>
      <td>identity</td>
      <td>(5, 6)</td>
      <td>0.01</td>
      <td>{'activation': 'identity', 'hidden_layer_sizes...</td>
      <td>0.958333</td>
      <td>0.985507</td>
      <td>0.840580</td>
      <td>0.928571</td>
      <td>0.062552</td>
      <td>3</td>
      <td>0.985507</td>
      <td>0.964539</td>
      <td>1.000000</td>
      <td>0.983349</td>
      <td>0.014557</td>
    </tr>
    <tr>
      <th>21</th>
      <td>0.043481</td>
      <td>0.014068</td>
      <td>0.000312</td>
      <td>4.052337e-07</td>
      <td>identity</td>
      <td>(5, 6)</td>
      <td>0.05</td>
      <td>{'activation': 'identity', 'hidden_layer_sizes...</td>
      <td>0.972222</td>
      <td>0.971014</td>
      <td>0.826087</td>
      <td>0.923810</td>
      <td>0.068363</td>
      <td>5</td>
      <td>0.942029</td>
      <td>0.964539</td>
      <td>1.000000</td>
      <td>0.968856</td>
      <td>0.023863</td>
    </tr>
    <tr>
      <th>22</th>
      <td>0.002886</td>
      <td>0.000473</td>
      <td>0.000300</td>
      <td>1.146394e-05</td>
      <td>logistic</td>
      <td>(1,)</td>
      <td>0.01</td>
      <td>{'activation': 'logistic', 'hidden_layer_sizes...</td>
      <td>0.333333</td>
      <td>0.333333</td>
      <td>0.333333</td>
      <td>0.333333</td>
      <td>0.000000</td>
      <td>76</td>
      <td>0.333333</td>
      <td>0.333333</td>
      <td>0.333333</td>
      <td>0.333333</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>23</th>
      <td>0.003865</td>
      <td>0.000251</td>
      <td>0.000297</td>
      <td>6.880716e-06</td>
      <td>logistic</td>
      <td>(1,)</td>
      <td>0.05</td>
      <td>{'activation': 'logistic', 'hidden_layer_sizes...</td>
      <td>0.333333</td>
      <td>0.333333</td>
      <td>0.333333</td>
      <td>0.333333</td>
      <td>0.000000</td>
      <td>76</td>
      <td>0.333333</td>
      <td>0.333333</td>
      <td>0.333333</td>
      <td>0.333333</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>24</th>
      <td>0.052061</td>
      <td>0.000233</td>
      <td>0.000348</td>
      <td>1.946809e-05</td>
      <td>logistic</td>
      <td>(7,)</td>
      <td>0.01</td>
      <td>{'activation': 'logistic', 'hidden_layer_sizes...</td>
      <td>0.875000</td>
      <td>0.942029</td>
      <td>0.855072</td>
      <td>0.890476</td>
      <td>0.036976</td>
      <td>22</td>
      <td>0.891304</td>
      <td>0.964539</td>
      <td>0.964539</td>
      <td>0.940127</td>
      <td>0.034523</td>
    </tr>
    <tr>
      <th>25</th>
      <td>0.051037</td>
      <td>0.000551</td>
      <td>0.000336</td>
      <td>1.351366e-05</td>
      <td>logistic</td>
      <td>(7,)</td>
      <td>0.05</td>
      <td>{'activation': 'logistic', 'hidden_layer_sizes...</td>
      <td>0.986111</td>
      <td>0.884058</td>
      <td>0.811594</td>
      <td>0.895238</td>
      <td>0.071911</td>
      <td>19</td>
      <td>0.942029</td>
      <td>0.978723</td>
      <td>0.943262</td>
      <td>0.954672</td>
      <td>0.017015</td>
    </tr>
    <tr>
      <th>26</th>
      <td>0.063004</td>
      <td>0.000587</td>
      <td>0.000350</td>
      <td>1.343209e-05</td>
      <td>logistic</td>
      <td>(1, 6)</td>
      <td>0.01</td>
      <td>{'activation': 'logistic', 'hidden_layer_sizes...</td>
      <td>0.861111</td>
      <td>0.913043</td>
      <td>0.594203</td>
      <td>0.790476</td>
      <td>0.138940</td>
      <td>45</td>
      <td>0.913043</td>
      <td>0.865248</td>
      <td>0.687943</td>
      <td>0.822078</td>
      <td>0.096834</td>
    </tr>
    <tr>
      <th>27</th>
      <td>0.063142</td>
      <td>0.000136</td>
      <td>0.000355</td>
      <td>2.278890e-05</td>
      <td>logistic</td>
      <td>(1, 6)</td>
      <td>0.05</td>
      <td>{'activation': 'logistic', 'hidden_layer_sizes...</td>
      <td>0.875000</td>
      <td>0.898551</td>
      <td>0.753623</td>
      <td>0.842857</td>
      <td>0.063164</td>
      <td>37</td>
      <td>0.913043</td>
      <td>0.879433</td>
      <td>0.936170</td>
      <td>0.909549</td>
      <td>0.023294</td>
    </tr>
    <tr>
      <th>28</th>
      <td>0.042257</td>
      <td>0.026653</td>
      <td>0.000334</td>
      <td>1.266739e-05</td>
      <td>logistic</td>
      <td>(2, 5)</td>
      <td>0.01</td>
      <td>{'activation': 'logistic', 'hidden_layer_sizes...</td>
      <td>0.861111</td>
      <td>0.217391</td>
      <td>0.623188</td>
      <td>0.571429</td>
      <td>0.266149</td>
      <td>61</td>
      <td>0.905797</td>
      <td>0.248227</td>
      <td>0.645390</td>
      <td>0.599805</td>
      <td>0.270380</td>
    </tr>
    <tr>
      <th>29</th>
      <td>0.041851</td>
      <td>0.026198</td>
      <td>0.000321</td>
      <td>1.149915e-05</td>
      <td>logistic</td>
      <td>(2, 5)</td>
      <td>0.05</td>
      <td>{'activation': 'logistic', 'hidden_layer_sizes...</td>
      <td>0.847222</td>
      <td>0.101449</td>
      <td>0.710145</td>
      <td>0.557143</td>
      <td>0.323685</td>
      <td>62</td>
      <td>0.898551</td>
      <td>0.163121</td>
      <td>0.943262</td>
      <td>0.668311</td>
      <td>0.357690</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>58</th>
      <td>0.003189</td>
      <td>0.000264</td>
      <td>0.000324</td>
      <td>5.619580e-07</td>
      <td>tanh</td>
      <td>(2, 9)</td>
      <td>0.01</td>
      <td>{'activation': 'tanh', 'hidden_layer_sizes': (...</td>
      <td>0.333333</td>
      <td>0.333333</td>
      <td>0.333333</td>
      <td>0.333333</td>
      <td>0.000000</td>
      <td>76</td>
      <td>0.333333</td>
      <td>0.333333</td>
      <td>0.333333</td>
      <td>0.333333</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>59</th>
      <td>0.047735</td>
      <td>0.029690</td>
      <td>0.000403</td>
      <td>4.787024e-05</td>
      <td>tanh</td>
      <td>(2, 9)</td>
      <td>0.05</td>
      <td>{'activation': 'tanh', 'hidden_layer_sizes': (...</td>
      <td>0.861111</td>
      <td>0.768116</td>
      <td>0.405797</td>
      <td>0.680952</td>
      <td>0.196216</td>
      <td>53</td>
      <td>0.898551</td>
      <td>0.872340</td>
      <td>0.432624</td>
      <td>0.734505</td>
      <td>0.213730</td>
    </tr>
    <tr>
      <th>60</th>
      <td>0.029221</td>
      <td>0.031550</td>
      <td>0.000383</td>
      <td>5.281939e-05</td>
      <td>tanh</td>
      <td>(3, 8)</td>
      <td>0.01</td>
      <td>{'activation': 'tanh', 'hidden_layer_sizes': (...</td>
      <td>0.861111</td>
      <td>0.449275</td>
      <td>0.333333</td>
      <td>0.552381</td>
      <td>0.227898</td>
      <td>63</td>
      <td>0.898551</td>
      <td>0.404255</td>
      <td>0.333333</td>
      <td>0.545380</td>
      <td>0.251402</td>
    </tr>
    <tr>
      <th>61</th>
      <td>0.071679</td>
      <td>0.045737</td>
      <td>0.000640</td>
      <td>7.652019e-05</td>
      <td>tanh</td>
      <td>(3, 8)</td>
      <td>0.05</td>
      <td>{'activation': 'tanh', 'hidden_layer_sizes': (...</td>
      <td>0.958333</td>
      <td>0.637681</td>
      <td>0.420290</td>
      <td>0.676190</td>
      <td>0.222029</td>
      <td>55</td>
      <td>0.949275</td>
      <td>0.687943</td>
      <td>0.340426</td>
      <td>0.659215</td>
      <td>0.249391</td>
    </tr>
    <tr>
      <th>62</th>
      <td>0.099149</td>
      <td>0.012122</td>
      <td>0.000436</td>
      <td>7.837652e-06</td>
      <td>tanh</td>
      <td>(4, 7)</td>
      <td>0.01</td>
      <td>{'activation': 'tanh', 'hidden_layer_sizes': (...</td>
      <td>0.944444</td>
      <td>0.913043</td>
      <td>0.782609</td>
      <td>0.880952</td>
      <td>0.069988</td>
      <td>23</td>
      <td>0.992754</td>
      <td>0.858156</td>
      <td>0.978723</td>
      <td>0.943211</td>
      <td>0.060415</td>
    </tr>
    <tr>
      <th>63</th>
      <td>0.095193</td>
      <td>0.009208</td>
      <td>0.000607</td>
      <td>9.744526e-05</td>
      <td>tanh</td>
      <td>(4, 7)</td>
      <td>0.05</td>
      <td>{'activation': 'tanh', 'hidden_layer_sizes': (...</td>
      <td>0.819444</td>
      <td>0.913043</td>
      <td>0.811594</td>
      <td>0.847619</td>
      <td>0.045880</td>
      <td>33</td>
      <td>0.884058</td>
      <td>0.879433</td>
      <td>0.950355</td>
      <td>0.904615</td>
      <td>0.032398</td>
    </tr>
    <tr>
      <th>64</th>
      <td>0.103916</td>
      <td>0.005361</td>
      <td>0.000666</td>
      <td>1.601452e-04</td>
      <td>tanh</td>
      <td>(5, 6)</td>
      <td>0.01</td>
      <td>{'activation': 'tanh', 'hidden_layer_sizes': (...</td>
      <td>0.986111</td>
      <td>0.913043</td>
      <td>0.811594</td>
      <td>0.904762</td>
      <td>0.071718</td>
      <td>15</td>
      <td>0.978261</td>
      <td>0.971631</td>
      <td>0.964539</td>
      <td>0.971477</td>
      <td>0.005603</td>
    </tr>
    <tr>
      <th>65</th>
      <td>0.085413</td>
      <td>0.005117</td>
      <td>0.000515</td>
      <td>1.450531e-04</td>
      <td>tanh</td>
      <td>(5, 6)</td>
      <td>0.05</td>
      <td>{'activation': 'tanh', 'hidden_layer_sizes': (...</td>
      <td>0.833333</td>
      <td>0.913043</td>
      <td>0.695652</td>
      <td>0.814286</td>
      <td>0.089181</td>
      <td>42</td>
      <td>0.898551</td>
      <td>0.886525</td>
      <td>0.929078</td>
      <td>0.904718</td>
      <td>0.017911</td>
    </tr>
    <tr>
      <th>66</th>
      <td>0.007049</td>
      <td>0.001915</td>
      <td>0.000749</td>
      <td>1.252861e-04</td>
      <td>relu</td>
      <td>(1,)</td>
      <td>0.01</td>
      <td>{'activation': 'relu', 'hidden_layer_sizes': (...</td>
      <td>0.333333</td>
      <td>0.333333</td>
      <td>0.333333</td>
      <td>0.333333</td>
      <td>0.000000</td>
      <td>76</td>
      <td>0.333333</td>
      <td>0.333333</td>
      <td>0.333333</td>
      <td>0.333333</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>67</th>
      <td>0.031012</td>
      <td>0.036171</td>
      <td>0.000572</td>
      <td>6.779813e-05</td>
      <td>relu</td>
      <td>(1,)</td>
      <td>0.05</td>
      <td>{'activation': 'relu', 'hidden_layer_sizes': (...</td>
      <td>0.333333</td>
      <td>0.333333</td>
      <td>0.811594</td>
      <td>0.490476</td>
      <td>0.224636</td>
      <td>72</td>
      <td>0.333333</td>
      <td>0.333333</td>
      <td>0.943262</td>
      <td>0.536643</td>
      <td>0.287523</td>
    </tr>
    <tr>
      <th>68</th>
      <td>0.074731</td>
      <td>0.016695</td>
      <td>0.000474</td>
      <td>4.666471e-05</td>
      <td>relu</td>
      <td>(7,)</td>
      <td>0.01</td>
      <td>{'activation': 'relu', 'hidden_layer_sizes': (...</td>
      <td>0.944444</td>
      <td>0.956522</td>
      <td>0.811594</td>
      <td>0.904762</td>
      <td>0.065362</td>
      <td>15</td>
      <td>0.985507</td>
      <td>0.978723</td>
      <td>1.000000</td>
      <td>0.988077</td>
      <td>0.008874</td>
    </tr>
    <tr>
      <th>69</th>
      <td>0.074705</td>
      <td>0.007908</td>
      <td>0.000535</td>
      <td>9.163300e-05</td>
      <td>relu</td>
      <td>(7,)</td>
      <td>0.05</td>
      <td>{'activation': 'relu', 'hidden_layer_sizes': (...</td>
      <td>0.986111</td>
      <td>0.971014</td>
      <td>0.797101</td>
      <td>0.919048</td>
      <td>0.085531</td>
      <td>12</td>
      <td>0.956522</td>
      <td>0.978723</td>
      <td>0.929078</td>
      <td>0.954774</td>
      <td>0.020305</td>
    </tr>
    <tr>
      <th>70</th>
      <td>0.049354</td>
      <td>0.061788</td>
      <td>0.000663</td>
      <td>5.085729e-05</td>
      <td>relu</td>
      <td>(1, 6)</td>
      <td>0.01</td>
      <td>{'activation': 'relu', 'hidden_layer_sizes': (...</td>
      <td>0.847222</td>
      <td>0.333333</td>
      <td>0.333333</td>
      <td>0.509524</td>
      <td>0.243925</td>
      <td>69</td>
      <td>0.876812</td>
      <td>0.333333</td>
      <td>0.333333</td>
      <td>0.514493</td>
      <td>0.256198</td>
    </tr>
    <tr>
      <th>71</th>
      <td>0.061354</td>
      <td>0.046126</td>
      <td>0.000553</td>
      <td>9.150823e-05</td>
      <td>relu</td>
      <td>(1, 6)</td>
      <td>0.05</td>
      <td>{'activation': 'relu', 'hidden_layer_sizes': (...</td>
      <td>0.333333</td>
      <td>0.942029</td>
      <td>0.753623</td>
      <td>0.671429</td>
      <td>0.255873</td>
      <td>56</td>
      <td>0.333333</td>
      <td>0.851064</td>
      <td>0.929078</td>
      <td>0.704492</td>
      <td>0.264374</td>
    </tr>
    <tr>
      <th>72</th>
      <td>0.003965</td>
      <td>0.000549</td>
      <td>0.000437</td>
      <td>7.229139e-05</td>
      <td>relu</td>
      <td>(2, 5)</td>
      <td>0.01</td>
      <td>{'activation': 'relu', 'hidden_layer_sizes': (...</td>
      <td>0.333333</td>
      <td>0.333333</td>
      <td>0.333333</td>
      <td>0.333333</td>
      <td>0.000000</td>
      <td>76</td>
      <td>0.333333</td>
      <td>0.333333</td>
      <td>0.333333</td>
      <td>0.333333</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>73</th>
      <td>0.003714</td>
      <td>0.000921</td>
      <td>0.000375</td>
      <td>5.384824e-05</td>
      <td>relu</td>
      <td>(2, 5)</td>
      <td>0.05</td>
      <td>{'activation': 'relu', 'hidden_layer_sizes': (...</td>
      <td>0.333333</td>
      <td>0.333333</td>
      <td>0.333333</td>
      <td>0.333333</td>
      <td>0.000000</td>
      <td>76</td>
      <td>0.333333</td>
      <td>0.333333</td>
      <td>0.333333</td>
      <td>0.333333</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>74</th>
      <td>0.026111</td>
      <td>0.032034</td>
      <td>0.000350</td>
      <td>3.287373e-05</td>
      <td>relu</td>
      <td>(3, 4)</td>
      <td>0.01</td>
      <td>{'activation': 'relu', 'hidden_layer_sizes': (...</td>
      <td>0.333333</td>
      <td>0.898551</td>
      <td>0.333333</td>
      <td>0.519048</td>
      <td>0.265479</td>
      <td>68</td>
      <td>0.333333</td>
      <td>0.950355</td>
      <td>0.333333</td>
      <td>0.539007</td>
      <td>0.290867</td>
    </tr>
    <tr>
      <th>75</th>
      <td>0.003819</td>
      <td>0.000825</td>
      <td>0.000370</td>
      <td>3.635933e-05</td>
      <td>relu</td>
      <td>(3, 4)</td>
      <td>0.05</td>
      <td>{'activation': 'relu', 'hidden_layer_sizes': (...</td>
      <td>0.333333</td>
      <td>0.333333</td>
      <td>0.333333</td>
      <td>0.333333</td>
      <td>0.000000</td>
      <td>76</td>
      <td>0.333333</td>
      <td>0.333333</td>
      <td>0.333333</td>
      <td>0.333333</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>76</th>
      <td>0.057809</td>
      <td>0.000933</td>
      <td>0.000489</td>
      <td>1.406642e-04</td>
      <td>relu</td>
      <td>(11,)</td>
      <td>0.01</td>
      <td>{'activation': 'relu', 'hidden_layer_sizes': (...</td>
      <td>0.986111</td>
      <td>0.913043</td>
      <td>0.782609</td>
      <td>0.895238</td>
      <td>0.084282</td>
      <td>19</td>
      <td>0.978261</td>
      <td>0.978723</td>
      <td>0.900709</td>
      <td>0.952564</td>
      <td>0.036668</td>
    </tr>
    <tr>
      <th>77</th>
      <td>0.039110</td>
      <td>0.024852</td>
      <td>0.000356</td>
      <td>2.999171e-05</td>
      <td>relu</td>
      <td>(11,)</td>
      <td>0.05</td>
      <td>{'activation': 'relu', 'hidden_layer_sizes': (...</td>
      <td>0.861111</td>
      <td>0.898551</td>
      <td>0.333333</td>
      <td>0.700000</td>
      <td>0.256957</td>
      <td>51</td>
      <td>0.905797</td>
      <td>0.985816</td>
      <td>0.333333</td>
      <td>0.741649</td>
      <td>0.290565</td>
    </tr>
    <tr>
      <th>78</th>
      <td>0.026958</td>
      <td>0.032028</td>
      <td>0.000552</td>
      <td>5.363659e-05</td>
      <td>relu</td>
      <td>(1, 10)</td>
      <td>0.01</td>
      <td>{'activation': 'relu', 'hidden_layer_sizes': (...</td>
      <td>0.888889</td>
      <td>0.333333</td>
      <td>0.333333</td>
      <td>0.523810</td>
      <td>0.263702</td>
      <td>65</td>
      <td>0.862319</td>
      <td>0.333333</td>
      <td>0.333333</td>
      <td>0.509662</td>
      <td>0.249366</td>
    </tr>
    <tr>
      <th>79</th>
      <td>0.005890</td>
      <td>0.000464</td>
      <td>0.000530</td>
      <td>8.410564e-05</td>
      <td>relu</td>
      <td>(1, 10)</td>
      <td>0.05</td>
      <td>{'activation': 'relu', 'hidden_layer_sizes': (...</td>
      <td>0.333333</td>
      <td>0.333333</td>
      <td>0.333333</td>
      <td>0.333333</td>
      <td>0.000000</td>
      <td>76</td>
      <td>0.333333</td>
      <td>0.333333</td>
      <td>0.333333</td>
      <td>0.333333</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>80</th>
      <td>0.053087</td>
      <td>0.033624</td>
      <td>0.000480</td>
      <td>1.160410e-04</td>
      <td>relu</td>
      <td>(2, 9)</td>
      <td>0.01</td>
      <td>{'activation': 'relu', 'hidden_layer_sizes': (...</td>
      <td>0.875000</td>
      <td>0.333333</td>
      <td>0.826087</td>
      <td>0.680952</td>
      <td>0.243999</td>
      <td>53</td>
      <td>0.862319</td>
      <td>0.333333</td>
      <td>0.978723</td>
      <td>0.724792</td>
      <td>0.280853</td>
    </tr>
    <tr>
      <th>81</th>
      <td>0.031613</td>
      <td>0.037837</td>
      <td>0.000571</td>
      <td>7.561775e-05</td>
      <td>relu</td>
      <td>(2, 9)</td>
      <td>0.05</td>
      <td>{'activation': 'relu', 'hidden_layer_sizes': (...</td>
      <td>0.833333</td>
      <td>0.333333</td>
      <td>0.333333</td>
      <td>0.504762</td>
      <td>0.237332</td>
      <td>71</td>
      <td>0.905797</td>
      <td>0.333333</td>
      <td>0.333333</td>
      <td>0.524155</td>
      <td>0.269862</td>
    </tr>
    <tr>
      <th>82</th>
      <td>0.005323</td>
      <td>0.000982</td>
      <td>0.000425</td>
      <td>4.400529e-06</td>
      <td>relu</td>
      <td>(3, 8)</td>
      <td>0.01</td>
      <td>{'activation': 'relu', 'hidden_layer_sizes': (...</td>
      <td>0.333333</td>
      <td>0.333333</td>
      <td>0.333333</td>
      <td>0.333333</td>
      <td>0.000000</td>
      <td>76</td>
      <td>0.333333</td>
      <td>0.333333</td>
      <td>0.333333</td>
      <td>0.333333</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>83</th>
      <td>0.030598</td>
      <td>0.035581</td>
      <td>0.000507</td>
      <td>7.015406e-05</td>
      <td>relu</td>
      <td>(3, 8)</td>
      <td>0.05</td>
      <td>{'activation': 'relu', 'hidden_layer_sizes': (...</td>
      <td>0.333333</td>
      <td>0.942029</td>
      <td>0.333333</td>
      <td>0.533333</td>
      <td>0.285901</td>
      <td>64</td>
      <td>0.333333</td>
      <td>0.978723</td>
      <td>0.333333</td>
      <td>0.548463</td>
      <td>0.304240</td>
    </tr>
    <tr>
      <th>84</th>
      <td>0.068686</td>
      <td>0.046201</td>
      <td>0.000557</td>
      <td>9.535836e-05</td>
      <td>relu</td>
      <td>(4, 7)</td>
      <td>0.01</td>
      <td>{'activation': 'relu', 'hidden_layer_sizes': (...</td>
      <td>0.847222</td>
      <td>0.333333</td>
      <td>0.710145</td>
      <td>0.633333</td>
      <td>0.217245</td>
      <td>58</td>
      <td>0.891304</td>
      <td>0.333333</td>
      <td>0.957447</td>
      <td>0.727361</td>
      <td>0.279925</td>
    </tr>
    <tr>
      <th>85</th>
      <td>0.047182</td>
      <td>0.043665</td>
      <td>0.000739</td>
      <td>2.082295e-04</td>
      <td>relu</td>
      <td>(4, 7)</td>
      <td>0.05</td>
      <td>{'activation': 'relu', 'hidden_layer_sizes': (...</td>
      <td>0.916667</td>
      <td>0.333333</td>
      <td>0.608696</td>
      <td>0.623810</td>
      <td>0.239174</td>
      <td>59</td>
      <td>0.884058</td>
      <td>0.333333</td>
      <td>0.680851</td>
      <td>0.632747</td>
      <td>0.227391</td>
    </tr>
    <tr>
      <th>86</th>
      <td>0.059921</td>
      <td>0.041102</td>
      <td>0.000520</td>
      <td>6.886652e-05</td>
      <td>relu</td>
      <td>(5, 6)</td>
      <td>0.01</td>
      <td>{'activation': 'relu', 'hidden_layer_sizes': (...</td>
      <td>0.972222</td>
      <td>0.913043</td>
      <td>0.333333</td>
      <td>0.742857</td>
      <td>0.287504</td>
      <td>49</td>
      <td>0.956522</td>
      <td>0.943262</td>
      <td>0.333333</td>
      <td>0.744372</td>
      <td>0.290699</td>
    </tr>
    <tr>
      <th>87</th>
      <td>0.068476</td>
      <td>0.005581</td>
      <td>0.000397</td>
      <td>1.476699e-05</td>
      <td>relu</td>
      <td>(5, 6)</td>
      <td>0.05</td>
      <td>{'activation': 'relu', 'hidden_layer_sizes': (...</td>
      <td>0.875000</td>
      <td>0.855072</td>
      <td>0.826087</td>
      <td>0.852381</td>
      <td>0.020124</td>
      <td>31</td>
      <td>0.876812</td>
      <td>0.943262</td>
      <td>1.000000</td>
      <td>0.940025</td>
      <td>0.050344</td>
    </tr>
  </tbody>
</table>
<p>88 rows × 19 columns</p>
</div>



Métricas do modelo que o obteve a melhor média de acurácia entre os folds


```python
print("Melhor média de acurácia entre os folds = " + str(max(results['mean_train_score'])))
```

    Melhor média de acurácia entre os folds = 0.9880768835440437



```python
results.loc[results['mean_train_score']==max(results['mean_train_score'])]
```




<div>

<table class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mean_fit_time</th>
      <th>std_fit_time</th>
      <th>mean_score_time</th>
      <th>std_score_time</th>
      <th>param_activation</th>
      <th>param_hidden_layer_sizes</th>
      <th>param_learning_rate_init</th>
      <th>params</th>
      <th>split0_test_score</th>
      <th>split1_test_score</th>
      <th>split2_test_score</th>
      <th>mean_test_score</th>
      <th>std_test_score</th>
      <th>rank_test_score</th>
      <th>split0_train_score</th>
      <th>split1_train_score</th>
      <th>split2_train_score</th>
      <th>mean_train_score</th>
      <th>std_train_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>68</th>
      <td>0.074731</td>
      <td>0.016695</td>
      <td>0.000474</td>
      <td>0.000047</td>
      <td>relu</td>
      <td>(7,)</td>
      <td>0.01</td>
      <td>{'activation': 'relu', 'hidden_layer_sizes': (...</td>
      <td>0.944444</td>
      <td>0.956522</td>
      <td>0.811594</td>
      <td>0.904762</td>
      <td>0.065362</td>
      <td>15</td>
      <td>0.985507</td>
      <td>0.978723</td>
      <td>1.0</td>
      <td>0.988077</td>
      <td>0.008874</td>
    </tr>
  </tbody>
</table>
</div>



Características do melhor modelo que endereça a tarefa


```python
clf.best_estimator_
```




    MLPClassifier(activation='identity', alpha=0.0001, batch_size='auto',
           beta_1=0.9, beta_2=0.999, early_stopping=False, epsilon=1e-08,
           hidden_layer_sizes=(7,), learning_rate='constant',
           learning_rate_init=0.05, max_iter=200, momentum=0.9,
           nesterovs_momentum=True, power_t=0.5, random_state=None,
           shuffle=True, solver='lbfgs', tol=0.0001, validation_fraction=0.1,
           verbose=False, warm_start=False)



Como estamos trabalhando com pouco dados, escolhemos o solver 'LBFGS', que converge mais rápido e trabalha com pouca memória.
