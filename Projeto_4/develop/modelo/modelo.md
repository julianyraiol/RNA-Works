
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

    UsageError: Line magic function `%` not found.


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

Cálculo da quantidade de neurônios nas camadas ocultas utilizando a regra da pirâmide geométrica 


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

Treinamento das redes neurais. O solver escolhido foi o LBFGS pois ele é um solver que se comporta melhor com datasets com poucos dados.


```python
clf = GridSearchCV(MLPClassifier(solver='lbfgs'), parameters, iid=True, cv = 3, return_train_score=True)
clf.fit(X, y)
```




    GridSearchCV(cv=3, error_score='raise-deprecating',
           estimator=MLPClassifier(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,
           beta_2=0.999, early_stopping=False, epsilon=1e-08,
           hidden_layer_sizes=(100,), learning_rate='constant',
           learning_rate_init=0.001, max_iter=200, momentum=0.9,
           n_iter_no_change=10, nesterovs_momentum=True, power_t=0.5,
           random_state=None, shuffle=True, solver='lbfgs', tol=0.0001,
           validation_fraction=0.1, verbose=False, warm_start=False),
           fit_params=None, iid=True, n_jobs=None,
           param_grid={'hidden_layer_sizes': [(1,), (7,), (1, 6), (2, 5), (3, 4), (11,), (1, 10), (2, 9), (3, 8), (4, 7), (5, 6)], 'learning_rate_init': [0.01, 0.05], 'activation': ['identity', 'logistic', 'tanh', 'relu']},
           pre_dispatch='2*n_jobs', refit=True, return_train_score=True,
           scoring=None, verbose=0)



Listagem de todas as redes neurais geradas pelo GridSearchCV


```python
results = pd.DataFrame.from_dict(clf.cv_results_)
results
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
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
      <td>0.067127</td>
      <td>0.023680</td>
      <td>0.000867</td>
      <td>0.000066</td>
      <td>identity</td>
      <td>(1,)</td>
      <td>0.01</td>
      <td>{'activation': 'identity', 'hidden_layer_sizes...</td>
      <td>0.888889</td>
      <td>0.913043</td>
      <td>0.753623</td>
      <td>0.852381</td>
      <td>0.069790</td>
      <td>30</td>
      <td>0.840580</td>
      <td>0.858156</td>
      <td>0.914894</td>
      <td>0.871210</td>
      <td>0.031712</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.058989</td>
      <td>0.000247</td>
      <td>0.000831</td>
      <td>0.000008</td>
      <td>identity</td>
      <td>(1,)</td>
      <td>0.05</td>
      <td>{'activation': 'identity', 'hidden_layer_sizes...</td>
      <td>0.888889</td>
      <td>0.913043</td>
      <td>0.753623</td>
      <td>0.852381</td>
      <td>0.069790</td>
      <td>30</td>
      <td>0.891304</td>
      <td>0.851064</td>
      <td>0.914894</td>
      <td>0.885754</td>
      <td>0.026352</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.051481</td>
      <td>0.017373</td>
      <td>0.000855</td>
      <td>0.000068</td>
      <td>identity</td>
      <td>(7,)</td>
      <td>0.01</td>
      <td>{'activation': 'identity', 'hidden_layer_sizes...</td>
      <td>0.972222</td>
      <td>0.971014</td>
      <td>0.797101</td>
      <td>0.914286</td>
      <td>0.081977</td>
      <td>9</td>
      <td>0.985507</td>
      <td>0.964539</td>
      <td>1.000000</td>
      <td>0.983349</td>
      <td>0.014557</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.048233</td>
      <td>0.016283</td>
      <td>0.000800</td>
      <td>0.000015</td>
      <td>identity</td>
      <td>(7,)</td>
      <td>0.05</td>
      <td>{'activation': 'identity', 'hidden_layer_sizes...</td>
      <td>0.972222</td>
      <td>0.956522</td>
      <td>0.840580</td>
      <td>0.923810</td>
      <td>0.058577</td>
      <td>3</td>
      <td>0.978261</td>
      <td>0.964539</td>
      <td>1.000000</td>
      <td>0.980933</td>
      <td>0.014600</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.057886</td>
      <td>0.014541</td>
      <td>0.000831</td>
      <td>0.000010</td>
      <td>identity</td>
      <td>(1, 6)</td>
      <td>0.01</td>
      <td>{'activation': 'identity', 'hidden_layer_sizes...</td>
      <td>0.875000</td>
      <td>0.913043</td>
      <td>0.753623</td>
      <td>0.847619</td>
      <td>0.067576</td>
      <td>34</td>
      <td>0.869565</td>
      <td>0.851064</td>
      <td>0.914894</td>
      <td>0.878508</td>
      <td>0.026815</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.067743</td>
      <td>0.005989</td>
      <td>0.000844</td>
      <td>0.000017</td>
      <td>identity</td>
      <td>(1, 6)</td>
      <td>0.05</td>
      <td>{'activation': 'identity', 'hidden_layer_sizes...</td>
      <td>0.888889</td>
      <td>0.942029</td>
      <td>0.797101</td>
      <td>0.876190</td>
      <td>0.059454</td>
      <td>26</td>
      <td>0.862319</td>
      <td>0.858156</td>
      <td>0.929078</td>
      <td>0.883184</td>
      <td>0.032496</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.080654</td>
      <td>0.009721</td>
      <td>0.000882</td>
      <td>0.000041</td>
      <td>identity</td>
      <td>(2, 5)</td>
      <td>0.01</td>
      <td>{'activation': 'identity', 'hidden_layer_sizes...</td>
      <td>0.930556</td>
      <td>0.942029</td>
      <td>0.811594</td>
      <td>0.895238</td>
      <td>0.058701</td>
      <td>18</td>
      <td>0.927536</td>
      <td>0.879433</td>
      <td>1.000000</td>
      <td>0.935656</td>
      <td>0.049555</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.073303</td>
      <td>0.003854</td>
      <td>0.001056</td>
      <td>0.000276</td>
      <td>identity</td>
      <td>(2, 5)</td>
      <td>0.05</td>
      <td>{'activation': 'identity', 'hidden_layer_sizes...</td>
      <td>0.916667</td>
      <td>0.956522</td>
      <td>0.782609</td>
      <td>0.885714</td>
      <td>0.073951</td>
      <td>23</td>
      <td>0.927536</td>
      <td>0.964539</td>
      <td>1.000000</td>
      <td>0.964025</td>
      <td>0.029585</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.072371</td>
      <td>0.003193</td>
      <td>0.000842</td>
      <td>0.000005</td>
      <td>identity</td>
      <td>(3, 4)</td>
      <td>0.01</td>
      <td>{'activation': 'identity', 'hidden_layer_sizes...</td>
      <td>0.986111</td>
      <td>0.956522</td>
      <td>0.797101</td>
      <td>0.914286</td>
      <td>0.082867</td>
      <td>9</td>
      <td>0.985507</td>
      <td>0.964539</td>
      <td>0.992908</td>
      <td>0.980985</td>
      <td>0.012015</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.061050</td>
      <td>0.012812</td>
      <td>0.000860</td>
      <td>0.000029</td>
      <td>identity</td>
      <td>(3, 4)</td>
      <td>0.05</td>
      <td>{'activation': 'identity', 'hidden_layer_sizes...</td>
      <td>0.972222</td>
      <td>0.971014</td>
      <td>0.826087</td>
      <td>0.923810</td>
      <td>0.068363</td>
      <td>3</td>
      <td>0.985507</td>
      <td>0.971631</td>
      <td>1.000000</td>
      <td>0.985713</td>
      <td>0.011582</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.061105</td>
      <td>0.001411</td>
      <td>0.000806</td>
      <td>0.000015</td>
      <td>identity</td>
      <td>(11,)</td>
      <td>0.01</td>
      <td>{'activation': 'identity', 'hidden_layer_sizes...</td>
      <td>0.972222</td>
      <td>0.985507</td>
      <td>0.797101</td>
      <td>0.919048</td>
      <td>0.085480</td>
      <td>6</td>
      <td>0.978261</td>
      <td>0.964539</td>
      <td>1.000000</td>
      <td>0.980933</td>
      <td>0.014600</td>
    </tr>
    <tr>
      <th>11</th>
      <td>0.054143</td>
      <td>0.010299</td>
      <td>0.000800</td>
      <td>0.000013</td>
      <td>identity</td>
      <td>(11,)</td>
      <td>0.05</td>
      <td>{'activation': 'identity', 'hidden_layer_sizes...</td>
      <td>0.986111</td>
      <td>0.942029</td>
      <td>0.840580</td>
      <td>0.923810</td>
      <td>0.060959</td>
      <td>3</td>
      <td>0.985507</td>
      <td>0.971631</td>
      <td>1.000000</td>
      <td>0.985713</td>
      <td>0.011582</td>
    </tr>
    <tr>
      <th>12</th>
      <td>0.076107</td>
      <td>0.002274</td>
      <td>0.000987</td>
      <td>0.000187</td>
      <td>identity</td>
      <td>(1, 10)</td>
      <td>0.01</td>
      <td>{'activation': 'identity', 'hidden_layer_sizes...</td>
      <td>0.861111</td>
      <td>0.942029</td>
      <td>0.753623</td>
      <td>0.852381</td>
      <td>0.076625</td>
      <td>30</td>
      <td>0.891304</td>
      <td>0.851064</td>
      <td>0.914894</td>
      <td>0.885754</td>
      <td>0.026352</td>
    </tr>
    <tr>
      <th>13</th>
      <td>0.073388</td>
      <td>0.000108</td>
      <td>0.000808</td>
      <td>0.000011</td>
      <td>identity</td>
      <td>(1, 10)</td>
      <td>0.05</td>
      <td>{'activation': 'identity', 'hidden_layer_sizes...</td>
      <td>0.861111</td>
      <td>0.942029</td>
      <td>0.753623</td>
      <td>0.852381</td>
      <td>0.076625</td>
      <td>30</td>
      <td>0.884058</td>
      <td>0.858156</td>
      <td>0.914894</td>
      <td>0.885703</td>
      <td>0.023192</td>
    </tr>
    <tr>
      <th>14</th>
      <td>0.071367</td>
      <td>0.000450</td>
      <td>0.000851</td>
      <td>0.000028</td>
      <td>identity</td>
      <td>(2, 9)</td>
      <td>0.01</td>
      <td>{'activation': 'identity', 'hidden_layer_sizes...</td>
      <td>0.944444</td>
      <td>0.971014</td>
      <td>0.782609</td>
      <td>0.900000</td>
      <td>0.082838</td>
      <td>17</td>
      <td>0.985507</td>
      <td>0.971631</td>
      <td>1.000000</td>
      <td>0.985713</td>
      <td>0.011582</td>
    </tr>
    <tr>
      <th>15</th>
      <td>0.071353</td>
      <td>0.000087</td>
      <td>0.000802</td>
      <td>0.000010</td>
      <td>identity</td>
      <td>(2, 9)</td>
      <td>0.05</td>
      <td>{'activation': 'identity', 'hidden_layer_sizes...</td>
      <td>0.986111</td>
      <td>0.913043</td>
      <td>0.739130</td>
      <td>0.880952</td>
      <td>0.103627</td>
      <td>25</td>
      <td>0.971014</td>
      <td>0.971631</td>
      <td>0.992908</td>
      <td>0.978518</td>
      <td>0.010178</td>
    </tr>
    <tr>
      <th>16</th>
      <td>0.070807</td>
      <td>0.001054</td>
      <td>0.000812</td>
      <td>0.000020</td>
      <td>identity</td>
      <td>(3, 8)</td>
      <td>0.01</td>
      <td>{'activation': 'identity', 'hidden_layer_sizes...</td>
      <td>0.958333</td>
      <td>0.956522</td>
      <td>0.811594</td>
      <td>0.909524</td>
      <td>0.068510</td>
      <td>11</td>
      <td>0.971014</td>
      <td>0.964539</td>
      <td>1.000000</td>
      <td>0.978518</td>
      <td>0.015419</td>
    </tr>
    <tr>
      <th>17</th>
      <td>0.070942</td>
      <td>0.000266</td>
      <td>0.000825</td>
      <td>0.000030</td>
      <td>identity</td>
      <td>(3, 8)</td>
      <td>0.05</td>
      <td>{'activation': 'identity', 'hidden_layer_sizes...</td>
      <td>0.972222</td>
      <td>0.956522</td>
      <td>0.782609</td>
      <td>0.904762</td>
      <td>0.085693</td>
      <td>14</td>
      <td>0.971014</td>
      <td>0.957447</td>
      <td>0.992908</td>
      <td>0.973790</td>
      <td>0.014609</td>
    </tr>
    <tr>
      <th>18</th>
      <td>0.068196</td>
      <td>0.004319</td>
      <td>0.000816</td>
      <td>0.000033</td>
      <td>identity</td>
      <td>(4, 7)</td>
      <td>0.01</td>
      <td>{'activation': 'identity', 'hidden_layer_sizes...</td>
      <td>0.972222</td>
      <td>0.985507</td>
      <td>0.753623</td>
      <td>0.904762</td>
      <td>0.105868</td>
      <td>14</td>
      <td>0.971014</td>
      <td>0.957447</td>
      <td>1.000000</td>
      <td>0.976154</td>
      <td>0.017748</td>
    </tr>
    <tr>
      <th>19</th>
      <td>0.070639</td>
      <td>0.000360</td>
      <td>0.000818</td>
      <td>0.000005</td>
      <td>identity</td>
      <td>(4, 7)</td>
      <td>0.05</td>
      <td>{'activation': 'identity', 'hidden_layer_sizes...</td>
      <td>0.972222</td>
      <td>0.971014</td>
      <td>0.782609</td>
      <td>0.909524</td>
      <td>0.088784</td>
      <td>11</td>
      <td>0.978261</td>
      <td>0.964539</td>
      <td>1.000000</td>
      <td>0.980933</td>
      <td>0.014600</td>
    </tr>
    <tr>
      <th>20</th>
      <td>0.071261</td>
      <td>0.000238</td>
      <td>0.000815</td>
      <td>0.000023</td>
      <td>identity</td>
      <td>(5, 6)</td>
      <td>0.01</td>
      <td>{'activation': 'identity', 'hidden_layer_sizes...</td>
      <td>0.972222</td>
      <td>0.971014</td>
      <td>0.811594</td>
      <td>0.919048</td>
      <td>0.075170</td>
      <td>6</td>
      <td>0.978261</td>
      <td>0.964539</td>
      <td>1.000000</td>
      <td>0.980933</td>
      <td>0.014600</td>
    </tr>
    <tr>
      <th>21</th>
      <td>0.073574</td>
      <td>0.002803</td>
      <td>0.000837</td>
      <td>0.000021</td>
      <td>identity</td>
      <td>(5, 6)</td>
      <td>0.05</td>
      <td>{'activation': 'identity', 'hidden_layer_sizes...</td>
      <td>0.986111</td>
      <td>0.971014</td>
      <td>0.840580</td>
      <td>0.933333</td>
      <td>0.065179</td>
      <td>1</td>
      <td>0.985507</td>
      <td>0.971631</td>
      <td>1.000000</td>
      <td>0.985713</td>
      <td>0.011582</td>
    </tr>
    <tr>
      <th>22</th>
      <td>0.041393</td>
      <td>0.026982</td>
      <td>0.000766</td>
      <td>0.000025</td>
      <td>logistic</td>
      <td>(1,)</td>
      <td>0.01</td>
      <td>{'activation': 'logistic', 'hidden_layer_sizes...</td>
      <td>0.277778</td>
      <td>0.913043</td>
      <td>0.826087</td>
      <td>0.666667</td>
      <td>0.283103</td>
      <td>50</td>
      <td>0.282609</td>
      <td>0.872340</td>
      <td>0.929078</td>
      <td>0.694676</td>
      <td>0.292295</td>
    </tr>
    <tr>
      <th>23</th>
      <td>0.004731</td>
      <td>0.000962</td>
      <td>0.000746</td>
      <td>0.000015</td>
      <td>logistic</td>
      <td>(1,)</td>
      <td>0.05</td>
      <td>{'activation': 'logistic', 'hidden_layer_sizes...</td>
      <td>0.277778</td>
      <td>0.333333</td>
      <td>0.333333</td>
      <td>0.314286</td>
      <td>0.026370</td>
      <td>88</td>
      <td>0.239130</td>
      <td>0.333333</td>
      <td>0.333333</td>
      <td>0.301932</td>
      <td>0.044408</td>
    </tr>
    <tr>
      <th>24</th>
      <td>0.056559</td>
      <td>0.014107</td>
      <td>0.000802</td>
      <td>0.000024</td>
      <td>logistic</td>
      <td>(7,)</td>
      <td>0.01</td>
      <td>{'activation': 'logistic', 'hidden_layer_sizes...</td>
      <td>0.875000</td>
      <td>0.637681</td>
      <td>0.768116</td>
      <td>0.761905</td>
      <td>0.097305</td>
      <td>40</td>
      <td>0.891304</td>
      <td>0.687943</td>
      <td>1.000000</td>
      <td>0.859749</td>
      <td>0.129336</td>
    </tr>
    <tr>
      <th>25</th>
      <td>0.063974</td>
      <td>0.005955</td>
      <td>0.000828</td>
      <td>0.000012</td>
      <td>logistic</td>
      <td>(7,)</td>
      <td>0.05</td>
      <td>{'activation': 'logistic', 'hidden_layer_sizes...</td>
      <td>0.972222</td>
      <td>0.927536</td>
      <td>0.782609</td>
      <td>0.895238</td>
      <td>0.080887</td>
      <td>18</td>
      <td>0.985507</td>
      <td>0.978723</td>
      <td>0.992908</td>
      <td>0.985713</td>
      <td>0.005793</td>
    </tr>
    <tr>
      <th>26</th>
      <td>0.030395</td>
      <td>0.035163</td>
      <td>0.000807</td>
      <td>0.000035</td>
      <td>logistic</td>
      <td>(1, 6)</td>
      <td>0.01</td>
      <td>{'activation': 'logistic', 'hidden_layer_sizes...</td>
      <td>0.638889</td>
      <td>0.333333</td>
      <td>0.333333</td>
      <td>0.438095</td>
      <td>0.145036</td>
      <td>80</td>
      <td>0.644928</td>
      <td>0.333333</td>
      <td>0.262411</td>
      <td>0.413557</td>
      <td>0.166146</td>
    </tr>
    <tr>
      <th>27</th>
      <td>0.029712</td>
      <td>0.036393</td>
      <td>0.000793</td>
      <td>0.000018</td>
      <td>logistic</td>
      <td>(1, 6)</td>
      <td>0.05</td>
      <td>{'activation': 'logistic', 'hidden_layer_sizes...</td>
      <td>0.861111</td>
      <td>0.405797</td>
      <td>0.333333</td>
      <td>0.538095</td>
      <td>0.235161</td>
      <td>61</td>
      <td>0.876812</td>
      <td>0.418440</td>
      <td>0.333333</td>
      <td>0.542862</td>
      <td>0.238681</td>
    </tr>
    <tr>
      <th>28</th>
      <td>0.031180</td>
      <td>0.033444</td>
      <td>0.000786</td>
      <td>0.000022</td>
      <td>logistic</td>
      <td>(2, 5)</td>
      <td>0.01</td>
      <td>{'activation': 'logistic', 'hidden_layer_sizes...</td>
      <td>0.333333</td>
      <td>0.333333</td>
      <td>0.753623</td>
      <td>0.471429</td>
      <td>0.197408</td>
      <td>76</td>
      <td>0.333333</td>
      <td>0.333333</td>
      <td>0.900709</td>
      <td>0.522459</td>
      <td>0.267464</td>
    </tr>
    <tr>
      <th>29</th>
      <td>0.069245</td>
      <td>0.008611</td>
      <td>0.000856</td>
      <td>0.000012</td>
      <td>logistic</td>
      <td>(2, 5)</td>
      <td>0.05</td>
      <td>{'activation': 'logistic', 'hidden_layer_sizes...</td>
      <td>0.861111</td>
      <td>0.623188</td>
      <td>0.623188</td>
      <td>0.704762</td>
      <td>0.112933</td>
      <td>45</td>
      <td>0.891304</td>
      <td>0.602837</td>
      <td>0.645390</td>
      <td>0.713177</td>
      <td>0.127147</td>
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
      <td>0.005325</td>
      <td>0.001198</td>
      <td>0.000783</td>
      <td>0.000006</td>
      <td>tanh</td>
      <td>(2, 9)</td>
      <td>0.01</td>
      <td>{'activation': 'tanh', 'hidden_layer_sizes': (...</td>
      <td>0.333333</td>
      <td>0.333333</td>
      <td>0.333333</td>
      <td>0.333333</td>
      <td>0.000000</td>
      <td>81</td>
      <td>0.333333</td>
      <td>0.333333</td>
      <td>0.333333</td>
      <td>0.333333</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>59</th>
      <td>0.060108</td>
      <td>0.039002</td>
      <td>0.000828</td>
      <td>0.000025</td>
      <td>tanh</td>
      <td>(2, 9)</td>
      <td>0.05</td>
      <td>{'activation': 'tanh', 'hidden_layer_sizes': (...</td>
      <td>0.847222</td>
      <td>0.333333</td>
      <td>0.797101</td>
      <td>0.661905</td>
      <td>0.230765</td>
      <td>52</td>
      <td>0.898551</td>
      <td>0.333333</td>
      <td>0.943262</td>
      <td>0.725049</td>
      <td>0.277585</td>
    </tr>
    <tr>
      <th>60</th>
      <td>0.032669</td>
      <td>0.037862</td>
      <td>0.000818</td>
      <td>0.000039</td>
      <td>tanh</td>
      <td>(3, 8)</td>
      <td>0.01</td>
      <td>{'activation': 'tanh', 'hidden_layer_sizes': (...</td>
      <td>0.847222</td>
      <td>0.333333</td>
      <td>0.333333</td>
      <td>0.509524</td>
      <td>0.243925</td>
      <td>66</td>
      <td>0.905797</td>
      <td>0.333333</td>
      <td>0.333333</td>
      <td>0.524155</td>
      <td>0.269862</td>
    </tr>
    <tr>
      <th>61</th>
      <td>0.034406</td>
      <td>0.038783</td>
      <td>0.000827</td>
      <td>0.000044</td>
      <td>tanh</td>
      <td>(3, 8)</td>
      <td>0.05</td>
      <td>{'activation': 'tanh', 'hidden_layer_sizes': (...</td>
      <td>0.333333</td>
      <td>0.333333</td>
      <td>0.840580</td>
      <td>0.500000</td>
      <td>0.238250</td>
      <td>67</td>
      <td>0.333333</td>
      <td>0.333333</td>
      <td>1.000000</td>
      <td>0.555556</td>
      <td>0.314270</td>
    </tr>
    <tr>
      <th>62</th>
      <td>0.033985</td>
      <td>0.037829</td>
      <td>0.000819</td>
      <td>0.000032</td>
      <td>tanh</td>
      <td>(4, 7)</td>
      <td>0.01</td>
      <td>{'activation': 'tanh', 'hidden_layer_sizes': (...</td>
      <td>0.333333</td>
      <td>0.333333</td>
      <td>0.811594</td>
      <td>0.490476</td>
      <td>0.224636</td>
      <td>69</td>
      <td>0.333333</td>
      <td>0.333333</td>
      <td>0.943262</td>
      <td>0.536643</td>
      <td>0.287523</td>
    </tr>
    <tr>
      <th>63</th>
      <td>0.059563</td>
      <td>0.037497</td>
      <td>0.000823</td>
      <td>0.000022</td>
      <td>tanh</td>
      <td>(4, 7)</td>
      <td>0.05</td>
      <td>{'activation': 'tanh', 'hidden_layer_sizes': (...</td>
      <td>0.333333</td>
      <td>0.913043</td>
      <td>0.724638</td>
      <td>0.652381</td>
      <td>0.242776</td>
      <td>55</td>
      <td>0.333333</td>
      <td>0.851064</td>
      <td>0.801418</td>
      <td>0.661939</td>
      <td>0.233241</td>
    </tr>
    <tr>
      <th>64</th>
      <td>0.070369</td>
      <td>0.024599</td>
      <td>0.000822</td>
      <td>0.000012</td>
      <td>tanh</td>
      <td>(5, 6)</td>
      <td>0.01</td>
      <td>{'activation': 'tanh', 'hidden_layer_sizes': (...</td>
      <td>0.972222</td>
      <td>0.971014</td>
      <td>0.811594</td>
      <td>0.919048</td>
      <td>0.075170</td>
      <td>6</td>
      <td>0.963768</td>
      <td>0.985816</td>
      <td>0.992908</td>
      <td>0.980831</td>
      <td>0.012407</td>
    </tr>
    <tr>
      <th>65</th>
      <td>0.061088</td>
      <td>0.036350</td>
      <td>0.000838</td>
      <td>0.000029</td>
      <td>tanh</td>
      <td>(5, 6)</td>
      <td>0.05</td>
      <td>{'activation': 'tanh', 'hidden_layer_sizes': (...</td>
      <td>0.958333</td>
      <td>0.333333</td>
      <td>0.724638</td>
      <td>0.676190</td>
      <td>0.258240</td>
      <td>47</td>
      <td>0.963768</td>
      <td>0.333333</td>
      <td>0.957447</td>
      <td>0.751516</td>
      <td>0.295711</td>
    </tr>
    <tr>
      <th>66</th>
      <td>0.003851</td>
      <td>0.000330</td>
      <td>0.000749</td>
      <td>0.000009</td>
      <td>relu</td>
      <td>(1,)</td>
      <td>0.01</td>
      <td>{'activation': 'relu', 'hidden_layer_sizes': (...</td>
      <td>0.333333</td>
      <td>0.333333</td>
      <td>0.333333</td>
      <td>0.333333</td>
      <td>0.000000</td>
      <td>81</td>
      <td>0.333333</td>
      <td>0.333333</td>
      <td>0.333333</td>
      <td>0.333333</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>67</th>
      <td>0.003885</td>
      <td>0.000348</td>
      <td>0.000740</td>
      <td>0.000010</td>
      <td>relu</td>
      <td>(1,)</td>
      <td>0.05</td>
      <td>{'activation': 'relu', 'hidden_layer_sizes': (...</td>
      <td>0.333333</td>
      <td>0.333333</td>
      <td>0.333333</td>
      <td>0.333333</td>
      <td>0.000000</td>
      <td>81</td>
      <td>0.333333</td>
      <td>0.333333</td>
      <td>0.333333</td>
      <td>0.333333</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>68</th>
      <td>0.045143</td>
      <td>0.028196</td>
      <td>0.000798</td>
      <td>0.000035</td>
      <td>relu</td>
      <td>(7,)</td>
      <td>0.01</td>
      <td>{'activation': 'relu', 'hidden_layer_sizes': (...</td>
      <td>0.944444</td>
      <td>0.898551</td>
      <td>0.333333</td>
      <td>0.728571</td>
      <td>0.277125</td>
      <td>43</td>
      <td>0.971014</td>
      <td>0.907801</td>
      <td>0.333333</td>
      <td>0.737383</td>
      <td>0.286869</td>
    </tr>
    <tr>
      <th>69</th>
      <td>0.049885</td>
      <td>0.012048</td>
      <td>0.000862</td>
      <td>0.000018</td>
      <td>relu</td>
      <td>(7,)</td>
      <td>0.05</td>
      <td>{'activation': 'relu', 'hidden_layer_sizes': (...</td>
      <td>0.694444</td>
      <td>0.898551</td>
      <td>0.724638</td>
      <td>0.771429</td>
      <td>0.089783</td>
      <td>39</td>
      <td>0.673913</td>
      <td>0.971631</td>
      <td>0.985816</td>
      <td>0.877120</td>
      <td>0.143806</td>
    </tr>
    <tr>
      <th>70</th>
      <td>0.027544</td>
      <td>0.032807</td>
      <td>0.000791</td>
      <td>0.000023</td>
      <td>relu</td>
      <td>(1, 6)</td>
      <td>0.01</td>
      <td>{'activation': 'relu', 'hidden_layer_sizes': (...</td>
      <td>0.333333</td>
      <td>0.333333</td>
      <td>0.768116</td>
      <td>0.476190</td>
      <td>0.204215</td>
      <td>74</td>
      <td>0.333333</td>
      <td>0.333333</td>
      <td>0.893617</td>
      <td>0.520095</td>
      <td>0.264120</td>
    </tr>
    <tr>
      <th>71</th>
      <td>0.024087</td>
      <td>0.027217</td>
      <td>0.000793</td>
      <td>0.000020</td>
      <td>relu</td>
      <td>(1, 6)</td>
      <td>0.05</td>
      <td>{'activation': 'relu', 'hidden_layer_sizes': (...</td>
      <td>0.333333</td>
      <td>0.333333</td>
      <td>0.753623</td>
      <td>0.471429</td>
      <td>0.197408</td>
      <td>76</td>
      <td>0.333333</td>
      <td>0.333333</td>
      <td>0.914894</td>
      <td>0.527187</td>
      <td>0.274150</td>
    </tr>
    <tr>
      <th>72</th>
      <td>0.028927</td>
      <td>0.033614</td>
      <td>0.000794</td>
      <td>0.000021</td>
      <td>relu</td>
      <td>(2, 5)</td>
      <td>0.01</td>
      <td>{'activation': 'relu', 'hidden_layer_sizes': (...</td>
      <td>0.333333</td>
      <td>0.333333</td>
      <td>0.753623</td>
      <td>0.471429</td>
      <td>0.197408</td>
      <td>76</td>
      <td>0.333333</td>
      <td>0.333333</td>
      <td>0.914894</td>
      <td>0.527187</td>
      <td>0.274150</td>
    </tr>
    <tr>
      <th>73</th>
      <td>0.028115</td>
      <td>0.033275</td>
      <td>0.000790</td>
      <td>0.000013</td>
      <td>relu</td>
      <td>(2, 5)</td>
      <td>0.05</td>
      <td>{'activation': 'relu', 'hidden_layer_sizes': (...</td>
      <td>0.930556</td>
      <td>0.333333</td>
      <td>0.333333</td>
      <td>0.538095</td>
      <td>0.283480</td>
      <td>61</td>
      <td>0.884058</td>
      <td>0.333333</td>
      <td>0.333333</td>
      <td>0.516908</td>
      <td>0.259614</td>
    </tr>
    <tr>
      <th>74</th>
      <td>0.064091</td>
      <td>0.018651</td>
      <td>0.000852</td>
      <td>0.000044</td>
      <td>relu</td>
      <td>(3, 4)</td>
      <td>0.01</td>
      <td>{'activation': 'relu', 'hidden_layer_sizes': (...</td>
      <td>0.888889</td>
      <td>0.913043</td>
      <td>0.855072</td>
      <td>0.885714</td>
      <td>0.023609</td>
      <td>23</td>
      <td>0.862319</td>
      <td>0.921986</td>
      <td>0.992908</td>
      <td>0.925737</td>
      <td>0.053379</td>
    </tr>
    <tr>
      <th>75</th>
      <td>0.027491</td>
      <td>0.031803</td>
      <td>0.000825</td>
      <td>0.000038</td>
      <td>relu</td>
      <td>(3, 4)</td>
      <td>0.05</td>
      <td>{'activation': 'relu', 'hidden_layer_sizes': (...</td>
      <td>0.333333</td>
      <td>0.333333</td>
      <td>0.753623</td>
      <td>0.471429</td>
      <td>0.197408</td>
      <td>76</td>
      <td>0.333333</td>
      <td>0.333333</td>
      <td>0.914894</td>
      <td>0.527187</td>
      <td>0.274150</td>
    </tr>
    <tr>
      <th>76</th>
      <td>0.058697</td>
      <td>0.012225</td>
      <td>0.000828</td>
      <td>0.000032</td>
      <td>relu</td>
      <td>(11,)</td>
      <td>0.01</td>
      <td>{'activation': 'relu', 'hidden_layer_sizes': (...</td>
      <td>0.972222</td>
      <td>0.971014</td>
      <td>0.855072</td>
      <td>0.933333</td>
      <td>0.054749</td>
      <td>1</td>
      <td>0.985507</td>
      <td>0.971631</td>
      <td>1.000000</td>
      <td>0.985713</td>
      <td>0.011582</td>
    </tr>
    <tr>
      <th>77</th>
      <td>0.058315</td>
      <td>0.014197</td>
      <td>0.000859</td>
      <td>0.000011</td>
      <td>relu</td>
      <td>(11,)</td>
      <td>0.05</td>
      <td>{'activation': 'relu', 'hidden_layer_sizes': (...</td>
      <td>0.930556</td>
      <td>0.913043</td>
      <td>0.840580</td>
      <td>0.895238</td>
      <td>0.038903</td>
      <td>18</td>
      <td>1.000000</td>
      <td>0.985816</td>
      <td>1.000000</td>
      <td>0.995272</td>
      <td>0.006687</td>
    </tr>
    <tr>
      <th>78</th>
      <td>0.030161</td>
      <td>0.036703</td>
      <td>0.000795</td>
      <td>0.000020</td>
      <td>relu</td>
      <td>(1, 10)</td>
      <td>0.01</td>
      <td>{'activation': 'relu', 'hidden_layer_sizes': (...</td>
      <td>0.333333</td>
      <td>0.811594</td>
      <td>0.333333</td>
      <td>0.490476</td>
      <td>0.224636</td>
      <td>69</td>
      <td>0.333333</td>
      <td>0.808511</td>
      <td>0.333333</td>
      <td>0.491726</td>
      <td>0.224001</td>
    </tr>
    <tr>
      <th>79</th>
      <td>0.004346</td>
      <td>0.000320</td>
      <td>0.000779</td>
      <td>0.000004</td>
      <td>relu</td>
      <td>(1, 10)</td>
      <td>0.05</td>
      <td>{'activation': 'relu', 'hidden_layer_sizes': (...</td>
      <td>0.333333</td>
      <td>0.333333</td>
      <td>0.333333</td>
      <td>0.333333</td>
      <td>0.000000</td>
      <td>81</td>
      <td>0.333333</td>
      <td>0.333333</td>
      <td>0.333333</td>
      <td>0.333333</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>80</th>
      <td>0.028938</td>
      <td>0.035078</td>
      <td>0.000796</td>
      <td>0.000028</td>
      <td>relu</td>
      <td>(2, 9)</td>
      <td>0.01</td>
      <td>{'activation': 'relu', 'hidden_layer_sizes': (...</td>
      <td>0.333333</td>
      <td>0.333333</td>
      <td>0.782609</td>
      <td>0.480952</td>
      <td>0.211022</td>
      <td>71</td>
      <td>0.333333</td>
      <td>0.333333</td>
      <td>0.992908</td>
      <td>0.553191</td>
      <td>0.310926</td>
    </tr>
    <tr>
      <th>81</th>
      <td>0.030529</td>
      <td>0.035423</td>
      <td>0.000813</td>
      <td>0.000026</td>
      <td>relu</td>
      <td>(2, 9)</td>
      <td>0.05</td>
      <td>{'activation': 'relu', 'hidden_layer_sizes': (...</td>
      <td>0.333333</td>
      <td>0.782609</td>
      <td>0.333333</td>
      <td>0.480952</td>
      <td>0.211022</td>
      <td>71</td>
      <td>0.333333</td>
      <td>0.858156</td>
      <td>0.333333</td>
      <td>0.508274</td>
      <td>0.247404</td>
    </tr>
    <tr>
      <th>82</th>
      <td>0.026490</td>
      <td>0.029870</td>
      <td>0.000791</td>
      <td>0.000022</td>
      <td>relu</td>
      <td>(3, 8)</td>
      <td>0.01</td>
      <td>{'activation': 'relu', 'hidden_layer_sizes': (...</td>
      <td>0.972222</td>
      <td>0.333333</td>
      <td>0.333333</td>
      <td>0.552381</td>
      <td>0.303258</td>
      <td>60</td>
      <td>0.905797</td>
      <td>0.333333</td>
      <td>0.333333</td>
      <td>0.524155</td>
      <td>0.269862</td>
    </tr>
    <tr>
      <th>83</th>
      <td>0.070866</td>
      <td>0.012502</td>
      <td>0.000836</td>
      <td>0.000006</td>
      <td>relu</td>
      <td>(3, 8)</td>
      <td>0.05</td>
      <td>{'activation': 'relu', 'hidden_layer_sizes': (...</td>
      <td>0.861111</td>
      <td>0.942029</td>
      <td>0.797101</td>
      <td>0.866667</td>
      <td>0.058879</td>
      <td>28</td>
      <td>0.869565</td>
      <td>0.851064</td>
      <td>1.000000</td>
      <td>0.906876</td>
      <td>0.066280</td>
    </tr>
    <tr>
      <th>84</th>
      <td>0.053739</td>
      <td>0.034594</td>
      <td>0.000841</td>
      <td>0.000029</td>
      <td>relu</td>
      <td>(4, 7)</td>
      <td>0.01</td>
      <td>{'activation': 'relu', 'hidden_layer_sizes': (...</td>
      <td>0.888889</td>
      <td>0.536232</td>
      <td>0.333333</td>
      <td>0.590476</td>
      <td>0.230704</td>
      <td>58</td>
      <td>0.847826</td>
      <td>0.624113</td>
      <td>0.333333</td>
      <td>0.601758</td>
      <td>0.210635</td>
    </tr>
    <tr>
      <th>85</th>
      <td>0.049586</td>
      <td>0.031893</td>
      <td>0.000834</td>
      <td>0.000042</td>
      <td>relu</td>
      <td>(4, 7)</td>
      <td>0.05</td>
      <td>{'activation': 'relu', 'hidden_layer_sizes': (...</td>
      <td>0.861111</td>
      <td>0.333333</td>
      <td>0.724638</td>
      <td>0.642857</td>
      <td>0.223625</td>
      <td>56</td>
      <td>0.913043</td>
      <td>0.333333</td>
      <td>0.929078</td>
      <td>0.725152</td>
      <td>0.277135</td>
    </tr>
    <tr>
      <th>86</th>
      <td>0.078909</td>
      <td>0.000948</td>
      <td>0.000818</td>
      <td>0.000010</td>
      <td>relu</td>
      <td>(5, 6)</td>
      <td>0.01</td>
      <td>{'activation': 'relu', 'hidden_layer_sizes': (...</td>
      <td>0.972222</td>
      <td>0.942029</td>
      <td>0.753623</td>
      <td>0.890476</td>
      <td>0.096530</td>
      <td>21</td>
      <td>0.978261</td>
      <td>0.964539</td>
      <td>0.936170</td>
      <td>0.959657</td>
      <td>0.017527</td>
    </tr>
    <tr>
      <th>87</th>
      <td>0.052574</td>
      <td>0.033673</td>
      <td>0.000825</td>
      <td>0.000028</td>
      <td>relu</td>
      <td>(5, 6)</td>
      <td>0.05</td>
      <td>{'activation': 'relu', 'hidden_layer_sizes': (...</td>
      <td>0.972222</td>
      <td>0.927536</td>
      <td>0.333333</td>
      <td>0.747619</td>
      <td>0.290388</td>
      <td>41</td>
      <td>0.978261</td>
      <td>0.929078</td>
      <td>0.333333</td>
      <td>0.746891</td>
      <td>0.293118</td>
    </tr>
  </tbody>
</table>
<p>88 rows × 19 columns</p>
</div>



Métricas do modelo que o obteve a melhor média de acurácia entre os folds


```python
print("Melhor média de acurácia entre os folds = " + str(max(results['mean_train_score'])))
```

    Melhor média de acurácia entre os folds = 0.9952718676122932



```python
results.loc[results['mean_train_score']==max(results['mean_train_score'])]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
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
      <th>77</th>
      <td>0.058315</td>
      <td>0.014197</td>
      <td>0.000859</td>
      <td>0.000011</td>
      <td>relu</td>
      <td>(11,)</td>
      <td>0.05</td>
      <td>{'activation': 'relu', 'hidden_layer_sizes': (...</td>
      <td>0.930556</td>
      <td>0.913043</td>
      <td>0.84058</td>
      <td>0.895238</td>
      <td>0.038903</td>
      <td>18</td>
      <td>1.0</td>
      <td>0.985816</td>
      <td>1.0</td>
      <td>0.995272</td>
      <td>0.006687</td>
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
           hidden_layer_sizes=(5, 6), learning_rate='constant',
           learning_rate_init=0.05, max_iter=200, momentum=0.9,
           n_iter_no_change=10, nesterovs_momentum=True, power_t=0.5,
           random_state=None, shuffle=True, solver='lbfgs', tol=0.0001,
           validation_fraction=0.1, verbose=False, warm_start=False)



Como estamos trabalhando com pouco dados, escolhemos o solver 'LBFGS'.  
Convergindo mais rápido e trabalhando com pouca memŕia.
