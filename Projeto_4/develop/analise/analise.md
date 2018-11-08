
## Projeto Prático 4

**Universidade do Estado do Amazonas**  
**Escola Superior de Tecnologia**  
**Professora:** Elloá B. Guedes  
**Alunos:** Juliany Raiol, Raí Soledade, Richardson Souza  
**Disciplina:** Redes Neurais Artificiais

## Aprendizado de Máquina com tarefa de classificação aplicado no dataset  de variedades de trigo


## Introdução

Três variedades de trigo (Kama, Rosa e Canadian) possuem sementes muito parecidas,
entretanto diferentes. Um grupo de pesquisadores poloneses coletou 70 amostras de cada
tipo e, usando uma técnica particular de raio-X, coletou medidas geométricas destas
sementes, a citar: área, perímetro, compactude, comprimento, largura, coeficiente de
assimetria e comprimento do sulco da semente.


```python
# Módulos utilizados no projeto

import pandas as pd
from pandas.tools.plotting import parallel_coordinates
import matplotlib.pyplot as plt
```

## 1. Carregando o conjunto de dados


```python
names = ["Area", "Perimeter", "Compactness", "Length", "Width", "Asymmetry", "Groove", "Seed"]

df = pd.read_csv('../../data/seeds_dataset.txt', delim_whitespace=True, names = names)
```

## 2. Caracterização do conjunto de dados

### 2.1. Informações gerais

Visualizando as 10 primeiras linhas do DataFrame, podemos observar que:
1. O conjunto de dados é composto por oito atributos de entrada.  
2. Não existe dados faltantes, assim como descrito no repositório do conjunto de dados
3. Para construir os dados, sete parâmetros geométricos de grãos de trigo foram medidos.
4. Para cada semente são apresentado os valores para os atributos: Area, Perimeter, Compactness, Length(comprimento da amêndoa), Width(largura da amêndoa), Asymmetry, Groove(comprimento do sulco do núcleo) e Seed, que indica o tipo correspondente de amêndoas pertencentes a três variedades deferentes de trigo: Kama(1), Rosa(2) e Canadian(3), 70 elementos de cada, e corresponde ao atributo alvo.


```python
print("Dimensões do dataset", df.shape)
```

    Dimensões do dataset (210, 8)



```python
df.columns # mostra o nome das colunas do conjunto de dados
```




    Index(['Area', 'Perimeter', 'Compactness', 'Length', 'Width', 'Asymmetry',
           'Groove', 'Seed'],
          dtype='object')




```python
df.head(10) # Imprime as 10 primeiras linhas do conjunto de dados
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
      <th>Area</th>
      <th>Perimeter</th>
      <th>Compactness</th>
      <th>Length</th>
      <th>Width</th>
      <th>Asymmetry</th>
      <th>Groove</th>
      <th>Seed</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>15.26</td>
      <td>14.84</td>
      <td>0.8710</td>
      <td>5.763</td>
      <td>3.312</td>
      <td>2.221</td>
      <td>5.220</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>14.88</td>
      <td>14.57</td>
      <td>0.8811</td>
      <td>5.554</td>
      <td>3.333</td>
      <td>1.018</td>
      <td>4.956</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>14.29</td>
      <td>14.09</td>
      <td>0.9050</td>
      <td>5.291</td>
      <td>3.337</td>
      <td>2.699</td>
      <td>4.825</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>13.84</td>
      <td>13.94</td>
      <td>0.8955</td>
      <td>5.324</td>
      <td>3.379</td>
      <td>2.259</td>
      <td>4.805</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>16.14</td>
      <td>14.99</td>
      <td>0.9034</td>
      <td>5.658</td>
      <td>3.562</td>
      <td>1.355</td>
      <td>5.175</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>14.38</td>
      <td>14.21</td>
      <td>0.8951</td>
      <td>5.386</td>
      <td>3.312</td>
      <td>2.462</td>
      <td>4.956</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6</th>
      <td>14.69</td>
      <td>14.49</td>
      <td>0.8799</td>
      <td>5.563</td>
      <td>3.259</td>
      <td>3.586</td>
      <td>5.219</td>
      <td>1</td>
    </tr>
    <tr>
      <th>7</th>
      <td>14.11</td>
      <td>14.10</td>
      <td>0.8911</td>
      <td>5.420</td>
      <td>3.302</td>
      <td>2.700</td>
      <td>5.000</td>
      <td>1</td>
    </tr>
    <tr>
      <th>8</th>
      <td>16.63</td>
      <td>15.46</td>
      <td>0.8747</td>
      <td>6.053</td>
      <td>3.465</td>
      <td>2.040</td>
      <td>5.877</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9</th>
      <td>16.44</td>
      <td>15.25</td>
      <td>0.8880</td>
      <td>5.884</td>
      <td>3.505</td>
      <td>1.969</td>
      <td>5.533</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.tail() # Imprime as 5 últimas linhas do conjunto de dados
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
      <th>Area</th>
      <th>Perimeter</th>
      <th>Compactness</th>
      <th>Length</th>
      <th>Width</th>
      <th>Asymmetry</th>
      <th>Groove</th>
      <th>Seed</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>205</th>
      <td>12.19</td>
      <td>13.20</td>
      <td>0.8783</td>
      <td>5.137</td>
      <td>2.981</td>
      <td>3.631</td>
      <td>4.870</td>
      <td>3</td>
    </tr>
    <tr>
      <th>206</th>
      <td>11.23</td>
      <td>12.88</td>
      <td>0.8511</td>
      <td>5.140</td>
      <td>2.795</td>
      <td>4.325</td>
      <td>5.003</td>
      <td>3</td>
    </tr>
    <tr>
      <th>207</th>
      <td>13.20</td>
      <td>13.66</td>
      <td>0.8883</td>
      <td>5.236</td>
      <td>3.232</td>
      <td>8.315</td>
      <td>5.056</td>
      <td>3</td>
    </tr>
    <tr>
      <th>208</th>
      <td>11.84</td>
      <td>13.21</td>
      <td>0.8521</td>
      <td>5.175</td>
      <td>2.836</td>
      <td>3.598</td>
      <td>5.044</td>
      <td>3</td>
    </tr>
    <tr>
      <th>209</th>
      <td>12.30</td>
      <td>13.34</td>
      <td>0.8684</td>
      <td>5.243</td>
      <td>2.974</td>
      <td>5.637</td>
      <td>5.063</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>




```python
print(df.groupby('Seed').size()) # Agrupa a coluna referente ao atributo alvo
```

    Seed
    1    70
    2    70
    3    70
    dtype: int64


### 2.2. Características dos atributos

Analisando os valores que um atributo pode assumir(tipo) consideramos que:
1. Todos os atributos quantitativos.
2. E, por serem resultado de medições de parâmetros geométricos e estarem representados por valores reais, todos os atributos preditivos são também contínuos e o atributo alvo é discreto pois contêm um número finito.

| Atributo | Classificação |
| ---: | :--- |
| Area | Quantitativo contínuo |
| Perimeter | Quantitativo contínuo |
| Compactness | Quantitativo contínuo |
| Length | Quantitativo contínuo |
| Width | Quantitativo contínuo |
| Asymmetry | Quantitativo contínuo |
| Groove | Quantitativo contínuo |
| Seed | Quantitativo discreto |


```python
df.dtypes # mostra os tipos de dados presentes no conjunto de dados
```




    Area           float64
    Perimeter      float64
    Compactness    float64
    Length         float64
    Width          float64
    Asymmetry      float64
    Groove         float64
    Seed             int64
    dtype: object




```python
df.isnull().any() # verifica se há valores nulos no dataset
```




    Area           False
    Perimeter      False
    Compactness    False
    Length         False
    Width          False
    Asymmetry      False
    Groove         False
    Seed           False
    dtype: bool



### 2.3. Escala dos atributos  
Analisando os valores que um atributo pode assumir(escala) consideramos que:
1. Todos os atributos preditivos são racionais, pois carregam mais informação. Existe uma escala de razão entre os atributos.
2. Já o atributo alvo, por carregar uma menor quantidade de informação e por não existir uma relação de ordem entre seus valores, consideramos nominal. São trẽs valores nominais ou classes(1, 2 e 3) que representam as três variedades diferentes de trigo: Kama, Rosa e Canadian.

| Atributo | Classificação |
| ---: | :--- |
| Area | Racional |
| Perimeter | Racional |
| Compactness | Racional |
| Length | Racional |
| Width | Racional |
| Asymmetry | Racional |
| Groove | Racional |
| Seed | Nominal |

### 3. Exploração dos dados

Para uma facilidade no processo de análise, foi criado dataframes de cada tipo de semente. Dessa forma, foram gerados 3: kama, rosa e canadian. 


```python
kama     = df[df['Seed']==1]
rosa     = df[df['Seed']==2]
canadian = df[df['Seed']==3]
```

> Dados gerais da semente do tipo <strong>Kama</strong>


```python
kama.describe()
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
      <th>Area</th>
      <th>Perimeter</th>
      <th>Compactness</th>
      <th>Length</th>
      <th>Width</th>
      <th>Asymmetry</th>
      <th>Groove</th>
      <th>Seed</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>70.000000</td>
      <td>70.000000</td>
      <td>70.000000</td>
      <td>70.000000</td>
      <td>70.000000</td>
      <td>70.000000</td>
      <td>70.000000</td>
      <td>70.0</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>14.334429</td>
      <td>14.294286</td>
      <td>0.880070</td>
      <td>5.508057</td>
      <td>3.244629</td>
      <td>2.667403</td>
      <td>5.087214</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.215704</td>
      <td>0.576583</td>
      <td>0.016191</td>
      <td>0.231508</td>
      <td>0.177616</td>
      <td>1.173901</td>
      <td>0.263699</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>min</th>
      <td>11.230000</td>
      <td>12.630000</td>
      <td>0.839200</td>
      <td>4.902000</td>
      <td>2.850000</td>
      <td>0.765100</td>
      <td>4.519000</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>13.750000</td>
      <td>13.960000</td>
      <td>0.868850</td>
      <td>5.384500</td>
      <td>3.134250</td>
      <td>1.826500</td>
      <td>4.924500</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>14.355000</td>
      <td>14.320000</td>
      <td>0.880500</td>
      <td>5.534000</td>
      <td>3.243500</td>
      <td>2.545500</td>
      <td>5.094000</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>15.045000</td>
      <td>14.732500</td>
      <td>0.890400</td>
      <td>5.677000</td>
      <td>3.378500</td>
      <td>3.301000</td>
      <td>5.223500</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>max</th>
      <td>17.080000</td>
      <td>15.460000</td>
      <td>0.918300</td>
      <td>6.053000</td>
      <td>3.683000</td>
      <td>6.685000</td>
      <td>5.877000</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>



> Dados gerais da semente do tipo <strong>Rosa</strong>


```python
rosa.describe()
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
      <th>Area</th>
      <th>Perimeter</th>
      <th>Compactness</th>
      <th>Length</th>
      <th>Width</th>
      <th>Asymmetry</th>
      <th>Groove</th>
      <th>Seed</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>70.000000</td>
      <td>70.000000</td>
      <td>70.000000</td>
      <td>70.000000</td>
      <td>70.000000</td>
      <td>70.000000</td>
      <td>70.000000</td>
      <td>70.0</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>18.334286</td>
      <td>16.135714</td>
      <td>0.883517</td>
      <td>6.148029</td>
      <td>3.677414</td>
      <td>3.644800</td>
      <td>6.020600</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.439496</td>
      <td>0.616995</td>
      <td>0.015500</td>
      <td>0.268191</td>
      <td>0.185539</td>
      <td>1.181868</td>
      <td>0.253934</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>min</th>
      <td>15.380000</td>
      <td>14.660000</td>
      <td>0.845200</td>
      <td>5.363000</td>
      <td>3.231000</td>
      <td>1.472000</td>
      <td>5.144000</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>17.330000</td>
      <td>15.737500</td>
      <td>0.872525</td>
      <td>5.979250</td>
      <td>3.554250</td>
      <td>2.845500</td>
      <td>5.877500</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>18.720000</td>
      <td>16.210000</td>
      <td>0.882600</td>
      <td>6.148500</td>
      <td>3.693500</td>
      <td>3.609500</td>
      <td>5.981500</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>19.137500</td>
      <td>16.557500</td>
      <td>0.898225</td>
      <td>6.312000</td>
      <td>3.804750</td>
      <td>4.436000</td>
      <td>6.187750</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>max</th>
      <td>21.180000</td>
      <td>17.250000</td>
      <td>0.910800</td>
      <td>6.675000</td>
      <td>4.033000</td>
      <td>6.682000</td>
      <td>6.550000</td>
      <td>2.0</td>
    </tr>
  </tbody>
</table>
</div>



> Dados gerais da semente do tipo <strong>Canadian</strong>


```python
canadian.describe()
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
      <th>Area</th>
      <th>Perimeter</th>
      <th>Compactness</th>
      <th>Length</th>
      <th>Width</th>
      <th>Asymmetry</th>
      <th>Groove</th>
      <th>Seed</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>70.000000</td>
      <td>70.000000</td>
      <td>70.000000</td>
      <td>70.000000</td>
      <td>70.000000</td>
      <td>70.000000</td>
      <td>70.000000</td>
      <td>70.0</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>11.873857</td>
      <td>13.247857</td>
      <td>0.849409</td>
      <td>5.229514</td>
      <td>2.853771</td>
      <td>4.788400</td>
      <td>5.116400</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.723004</td>
      <td>0.340196</td>
      <td>0.021760</td>
      <td>0.138015</td>
      <td>0.147516</td>
      <td>1.336465</td>
      <td>0.162068</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>min</th>
      <td>10.590000</td>
      <td>12.410000</td>
      <td>0.808100</td>
      <td>4.899000</td>
      <td>2.630000</td>
      <td>1.661000</td>
      <td>4.745000</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>11.262500</td>
      <td>13.002500</td>
      <td>0.834000</td>
      <td>5.136250</td>
      <td>2.725500</td>
      <td>4.048750</td>
      <td>5.002000</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>11.835000</td>
      <td>13.250000</td>
      <td>0.849350</td>
      <td>5.224000</td>
      <td>2.834500</td>
      <td>4.839000</td>
      <td>5.091500</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>12.425000</td>
      <td>13.470000</td>
      <td>0.861825</td>
      <td>5.323750</td>
      <td>2.967000</td>
      <td>5.467250</td>
      <td>5.228500</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>max</th>
      <td>13.370000</td>
      <td>13.950000</td>
      <td>0.897700</td>
      <td>5.541000</td>
      <td>3.232000</td>
      <td>8.456000</td>
      <td>5.491000</td>
      <td>3.0</td>
    </tr>
  </tbody>
</table>
</div>



Fazendo uma análise da descrição de cada tipo de semente, podemos ver que a média do atributo **Area** é bem distinta entre os três tipos de semente.  
O atributo **Perimeter** e **Asymmetry** tem uma distinção mais discreta entre os três tipos. Porém, a média **Perimeter** da semente Rosa em relação aos tipos Kama e Canadian é bem diferenciado. 

Abaixo isolamos esses três atributos para melhor visualização:

#### MEDIDAS DE LOCALIDADE (MÉDIA, MEDIANA E QUARTIS E PERCETIS)

1 - Kama
2 - Rosa
3 - Canadian


```python
gb = df.groupby('Seed')[['Area','Perimeter', 'Asymmetry']]
gb.mean()
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
      <th>Area</th>
      <th>Perimeter</th>
      <th>Asymmetry</th>
    </tr>
    <tr>
      <th>Seed</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>14.334429</td>
      <td>14.294286</td>
      <td>2.667403</td>
    </tr>
    <tr>
      <th>2</th>
      <td>18.334286</td>
      <td>16.135714</td>
      <td>3.644800</td>
    </tr>
    <tr>
      <th>3</th>
      <td>11.873857</td>
      <td>13.247857</td>
      <td>4.788400</td>
    </tr>
  </tbody>
</table>
</div>



Mas, como já sabemos, a média tem uma sensibilidade aos **outliers**. Então, vamos visualizar a mediana desses atributos Area, Perimeter e Asymmtry.


```python
gb.median()
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
      <th>Area</th>
      <th>Perimeter</th>
      <th>Asymmetry</th>
    </tr>
    <tr>
      <th>Seed</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>14.355</td>
      <td>14.32</td>
      <td>2.5455</td>
    </tr>
    <tr>
      <th>2</th>
      <td>18.720</td>
      <td>16.21</td>
      <td>3.6095</td>
    </tr>
    <tr>
      <th>3</th>
      <td>11.835</td>
      <td>13.25</td>
      <td>4.8390</td>
    </tr>
  </tbody>
</table>
</div>



Vendo a mediana dos atributos acima, ainda mantemos a mesma conclusão da análise feita com a média. Mesmo assim, a mediana não elimina os **outliers** apenas minimiza sua presença.


```python
gb.std()
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
      <th>Area</th>
      <th>Perimeter</th>
      <th>Asymmetry</th>
    </tr>
    <tr>
      <th>Seed</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>1.215704</td>
      <td>0.576583</td>
      <td>1.173901</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.439496</td>
      <td>0.616995</td>
      <td>1.181868</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.723004</td>
      <td>0.340196</td>
      <td>1.336465</td>
    </tr>
  </tbody>
</table>
</div>



Os desvios padrões com valores acima de 1, indicam maior dispersão dos dados. Isso é visto com maior entoação no atributo Asymmetry em todos as sementes disponíveis no dataset. 

### 4. Visualização dos dados

### 4.1.  Histogramas


```python
fig,ax=plt.subplots(3,3,figsize=(16, 12))

# Histograma do atributo Area
kama.Area.plot(kind="hist", ax=ax[0][0],label="kama",color ='k',alpha=0.5,fontsize=10)
rosa.Area.plot(kind="hist", ax=ax[0][1],label="rosa",color='k',alpha=0.5,fontsize=10)
canadian.Area.plot( kind="hist",ax=ax[0][2],label="canadian",color='k',alpha=0.5,fontsize=10)

kama.Perimeter.plot(kind="hist", ax=ax[1][0],label="kama",color ='k',alpha=0.5,fontsize=10)
rosa.Perimeter.plot(kind="hist", ax=ax[1][1],label="rosa",color='k',alpha=0.5,fontsize=10)
canadian.Perimeter.plot( kind="hist",ax=ax[1][2],label="canadian",color='k', alpha=0.5,fontsize=10)

kama.Asymmetry.plot(kind="hist", ax=ax[2][0],label="kama",color ='k', alpha=0.5,fontsize=10)
rosa.Asymmetry.plot(kind="hist", ax=ax[2][1],label="rosa",color='k',alpha=0.5,fontsize=10)
canadian.Asymmetry.plot( kind="hist",ax=ax[2][2],label="canadian",color='k',alpha=0.5,fontsize=10)

ax[0][0].set(title='Area')
ax[0][1].set(title='Area')
ax[0][2].set(title='Area')
ax[1][0].set(title='Perimeter')
ax[1][1].set(title='Perimeter')
ax[1][2].set(title='Perimeter')
ax[2][0].set(title='Asymmetry')
ax[2][1].set(title='Asymmetry')
ax[2][2].set(title='Asymmetry')

ax[0][0].legend()
ax[0][1].legend()
ax[0][2].legend()
ax[1][0].legend()
ax[1][1].legend()
ax[1][2].legend()
ax[2][0].legend()
ax[2][1].legend()
ax[2][2].legend()

plt.savefig('histograma.png')
plt.show()
```


![png](output_36_0.png)


A distribuição dos valores em um conjunto de dados está associada ao valor da obliquidade. 

Quando visualizamos o histograma do atributo "Area", as sementes do tipo "Rosa" e "Canadian" apresentam distribuições do tipo multimodal, onde há o aparecimento de vários picos. Existe uma quantidade maior de valores não arrendondados nesses cenários. Por outro lado, a "Kama" mostra um frequência assimétrica, diferente das demais.

Em "Perimeter", as distribuições tendem a se centralizar, entretanto, ainda existe a presença de picos isolados.

No atributo "Asymmetry", a semente do tipo "Rosa" apresenta um gráfico do tipo "Platô", ou seja, há diversas misturas de distribuições com médias diferentes. A "Kama" é assimétrico e distorcido à direita, indica a ocorrência de altos valores com baixa frequência. "Canadian" continua a apresentar um gráfico multimodal, com a presença de outliers.

### 4.2. BOXPLOT


```python
fig,ax=plt.subplots(3,3,figsize=(16, 12))

# boxplot do atributo Area
kama.Area.plot(kind="box", ax=ax[0][0],label="kama",color ='k',sym='r+', fontsize=10)
rosa.Area.plot(kind="box", ax=ax[0][1],label="rosa",color='k',sym='r+', fontsize=10)
canadian.Area.plot( kind="box",ax=ax[0][2],label="canadian",color='k',sym='r+', fontsize=10)

# boxplot do atributo Perimeter
kama.Perimeter.plot(kind="box", ax=ax[1][0],label="kama",color ='k',sym='r+',fontsize=10)
rosa.Perimeter.plot(kind="box", ax=ax[1][1],label="rosa",color='k',sym='r+',fontsize=10)
canadian.Perimeter.plot( kind="box",ax=ax[1][2],label="canadian",color='k',sym='r+',fontsize=10)

# boxplot do atributo Asymmetry
kama.Asymmetry.plot(kind="box", ax=ax[2][0],label="kama",color ='k',sym='r+',fontsize=10)
rosa.Asymmetry.plot(kind="box", ax=ax[2][1],label="rosa",color='k',sym='r+',fontsize=10)
canadian.Asymmetry.plot( kind="box",ax=ax[2][2],label="canadian",color='k',sym='r+',fontsize=10)

ax[0][0].set(title='Area')
ax[0][1].set(title='Area')
ax[0][2].set(title='Area')
ax[1][0].set(title='Perimeter')
ax[1][1].set(title='Perimeter')
ax[1][2].set(title='Perimeter')
ax[2][0].set(title='Asymmetry')
ax[2][1].set(title='Asymmetry')
ax[2][2].set(title='Asymmetry')

plt.savefig('boxplot.png')
plt.show()
```


![png](output_39_0.png)


### 4.3. SCATTER PLOT


```python
fig, ax = plt.subplots(1,3,figsize=(21, 6))

kama.plot(x="Area", y="Perimeter", kind="scatter",ax=ax[0],label='kama',color='r',marker='s')
rosa.plot(x="Area",y="Perimeter",kind="scatter",ax=ax[0],label='rosa',color='g',marker='^')
canadian.plot(x="Area", y="Perimeter", kind="scatter", ax=ax[0], label='canadian', color='b',marker='o')

kama.plot(x="Area", y="Asymmetry", kind="scatter",ax=ax[1],label='kama',color='r',marker='s')
rosa.plot(x="Area",y="Asymmetry",kind="scatter",ax=ax[1],label='rosa',color='b',marker='^')
canadian.plot(x="Area", y="Asymmetry", kind="scatter", ax=ax[1], label='canadian', color='g',marker='o')

kama.plot(x="Perimeter", y="Asymmetry", kind="scatter",ax=ax[2],label='kama',color='r',marker='s')
rosa.plot(x="Perimeter",y="Asymmetry",kind="scatter",ax=ax[2],label='rosa',color='b',marker='^')
canadian.plot(x="Perimeter", y="Asymmetry", kind="scatter", ax=ax[2], label='canadian', color='g',marker='o')

ax[0].set(title='Area x Perimeter comparasion ', ylabel='Perimeter')
ax[1].set(title='Area x Asymmetry Comparasion',  ylabel='Asymetry')
ax[2].set(title='Perimeter x Asymmetry Comparasion',  ylabel='Asymetry')
ax[0].legend()
ax[1].legend()
ax[2].legend()
plt.show()
```


![png](output_41_0.png)


Os gráficos de dispersão referem-se como os dados estão dispostos. É também possível visualizar a correlação 
entre os atributos analisados.

O gráfico Area x Perimeter ilustra uma relação linear, visto que o valor de X aumenta juntamente com o de Y de maneira linear. A partir desse gráfico, é possível perceber que as sementes do tipo Canadian apresentam "Area" e "Perimeter" menores em relação os demais. 

Outro ponto de destaque é que através dos gráficos Area x Asymmetry e Perimeter x Asymmetry é possível identificar que não há correlação linear evidente entre os atributos. Tendo em vista que esses apresentam maior discrepância entre si. 


```python
fig, ax = plt.subplots(1,2,figsize=(21, 6))

kama.plot(x="Area", y="Groove", kind="scatter",ax=ax[0],label='kama',color='r',marker='s')
rosa.plot(x="Area",y="Groove",kind="scatter",ax=ax[0],label='rosa',color='b',marker='^')
canadian.plot(x="Area", y="Groove", kind="scatter", ax=ax[0], label='canadian', color='g',marker='o')


kama.plot(x="Perimeter", y="Groove", kind="scatter",ax=ax[1],label='kama',color='r',marker='s')
rosa.plot(x="Perimeter",y="Groove",kind="scatter",ax=ax[1],label='rosa',color='b',marker='^')
canadian.plot(x="Perimeter", y="Groove", kind="scatter", ax=ax[1], label='canadian', color='g',marker='o')

ax[0].set(title='Area x Groove ', ylabel='Perimeter')
ax[1].set(title='Perimeter x Groove',  ylabel='Groove')
ax[0].legend()
ax[1].legend()
plt.show()
```


![png](output_43_0.png)


Nesse esses é possível visualizar agrupamentos numa mesma região do gráfico.   

### 5. Correlação entre os atributos


```python
f,ax = plt.subplots(figsize=(10, 8))
sns.heatmap(df.iloc[:,0:8].corr(), annot=True, linewidths=.5, fmt= '.2f',ax=ax ,cmap="Blues")
plt.savefig('correlation.png')
plt.show()
```


![png](output_46_0.png)


Como os dados são derivados a partir medições de área e perimetro de cada semente, já é esperado que haja uma correlação entre os atributos de entrada Area, Perimeter, Length, Width e Groove.


```python
from pylab import *
 
def cface(ax, x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15,x16,x17,x18):
    # x1 = height  of upper face
    # x2 = overlap of lower face
    # x3 = half of vertical size of face
    # x4 = width of upper face
    # x5 = width of lower face
    # x6 = length of nose
    # x7 = vertical position of mouth
    # x8 = curvature of mouth
    # x9 = width of mouth
    # x10 = vertical position of eyes
    # x11 = separation of eyes
    # x12 = slant of eyes
    # x13 = eccentricity of eyes
    # x14 = size of eyes
    # x15 = position of pupils
    # x16 = vertical position of eyebrows
    # x17 = slant of eyebrows
    # x18 = size of eyebrows
     
    # transform some values so that input between 0,1 yields variety of output
    x3 = 1.9*(x3-.5)
    x4 = (x4+.25)
    x5 = (x5+.2)
    x6 = .3*(x6+.01)
    x8 = 5*(x8+.001)
    x11 /= 5
    x12 = 2*(x12-.5)
    x13 += .05
    x14 += .1
    x15 = .5*(x15-.5)
    x16 = .25*x16
    x17 = .5*(x17-.5)
    x18 = .5*(x18+.1)
 
    # top of face, in box with l=-x4, r=x4, t=x1, b=x3
    e = mpl.patches.Ellipse( (0,(x1+x3)/2), 2*x4, (x1-x3), fc='white', linewidth=2)
    ax.add_artist(e)
 
    # bottom of face, in box with l=-x5, r=x5, b=-x1, t=x2+x3
    e = mpl.patches.Ellipse( (0,(-x1+x2+x3)/2), 2*x5, (x1+x2+x3), fc='white', linewidth=2)
    ax.add_artist(e)
 
    # cover overlaps
    e = mpl.patches.Ellipse( (0,(x1+x3)/2), 2*x4, (x1-x3), fc='white', ec='none')
    ax.add_artist(e)
    e = mpl.patches.Ellipse( (0,(-x1+x2+x3)/2), 2*x5, (x1+x2+x3), fc='white', ec='none')
    ax.add_artist(e)
     
    # draw nose
    plot([0,0], [-x6/2, x6/2], 'k')
     
    # draw mouth
    p = mpl.patches.Arc( (0,-x7+.5/x8), 1/x8, 1/x8, theta1=270-180/pi*arctan(x8*x9), theta2=270+180/pi*arctan(x8*x9))
    ax.add_artist(p)
     
    # draw eyes
    p = mpl.patches.Ellipse( (-x11-x14/2,x10), x14, x13*x14, angle=-180/pi*x12, facecolor='white')
    ax.add_artist(p)
     
    p = mpl.patches.Ellipse( (x11+x14/2,x10), x14, x13*x14, angle=180/pi*x12, facecolor='white')
    ax.add_artist(p)
 
    # draw pupils
    p = mpl.patches.Ellipse( (-x11-x14/2-x15*x14/2, x10), .05, .05, facecolor='black')
    ax.add_artist(p)
    p = mpl.patches.Ellipse( (x11+x14/2-x15*x14/2, x10), .05, .05, facecolor='black')
    ax.add_artist(p)
     
    # draw eyebrows
    plot([-x11-x14/2-x14*x18/2,-x11-x14/2+x14*x18/2],[x10+x13*x14*(x16+x17),x10+x13*x14*(x16-x17)],'k')
    plot([x11+x14/2+x14*x18/2,x11+x14/2-x14*x18/2],[x10+x13*x14*(x16+x17),x10+x13*x14*(x16-x17)],'k')
    
fig = figure(figsize=(11,11))
                                                
for i in range(25):
    ax = fig.add_subplot(5,5,i+1,aspect='equal')
    cface(ax, .9, *rand(17))
    ax.axis([-1.2,1.2,-1.2,1.2])
    ax.set_xticks([])
    ax.set_yticks([])
 
fig.subplots_adjust(hspace=0, wspace=0)
```


![png](output_48_0.png)

