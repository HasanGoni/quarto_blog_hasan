* This is the tutorial of ``tensorflow_data_validation``. This tutorial is a part of coursera course ``Machine learning Data Lifecycle``


```python
import tensorflow as tf
import tensorflow_data_validation as tfdv
import pandas as pd
import numpy as np
```


```python
from tensorflow_metadata.proto.v0 import schema_pb2
```


```python
print(f'tensorflow version = {tf.__version__}')
print(f'tensorflow data validation version = {tfdv.__version__}')
```

    tensorflow version = 2.5.0
    tensorflow data validation version = 1.1.0
    

In this notebook we will look at the  ``__tensorflow_data_validation__`` module. We will see
1. Visulaize training data statistics
2. Infer data schema
3. Generate and visualize eval dataset
4. Calculate and display evaluation anomalies
5. Revising the schema
6. Fix anomalies in the Schema
7. Examining dataset slices


We will be using [Census Income Dataset](http://archive.ics.uci.edu/ml/datasets/Census+Income) 


```python
from pathlib import Path
Path.ls = lambda x:list(x.iterdir())
Path.cwd()
```




    WindowsPath('D:/Online_courses/coursera/ML_production/Machine Learning Data Lifecycle in Production/week1')




```python
df = pd.read_csv('../data/adult.data')
df.head()
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
      <th>age</th>
      <th>workclass</th>
      <th>fnlwgt</th>
      <th>education</th>
      <th>education-num</th>
      <th>marital-status</th>
      <th>occupation</th>
      <th>relationship</th>
      <th>race</th>
      <th>sex</th>
      <th>capital-gain</th>
      <th>capital-loss</th>
      <th>hours-per-week</th>
      <th>native-country</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>39</td>
      <td>State-gov</td>
      <td>77516</td>
      <td>Bachelors</td>
      <td>13</td>
      <td>Never-married</td>
      <td>Adm-clerical</td>
      <td>Not-in-family</td>
      <td>White</td>
      <td>Male</td>
      <td>2174</td>
      <td>0</td>
      <td>40</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>1</th>
      <td>50</td>
      <td>Self-emp-not-inc</td>
      <td>83311</td>
      <td>Bachelors</td>
      <td>13</td>
      <td>Married-civ-spouse</td>
      <td>Exec-managerial</td>
      <td>Husband</td>
      <td>White</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>13</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>2</th>
      <td>38</td>
      <td>Private</td>
      <td>215646</td>
      <td>HS-grad</td>
      <td>9</td>
      <td>Divorced</td>
      <td>Handlers-cleaners</td>
      <td>Not-in-family</td>
      <td>White</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>3</th>
      <td>53</td>
      <td>Private</td>
      <td>234721</td>
      <td>11th</td>
      <td>7</td>
      <td>Married-civ-spouse</td>
      <td>Handlers-cleaners</td>
      <td>Husband</td>
      <td>Black</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>4</th>
      <td>28</td>
      <td>Private</td>
      <td>338409</td>
      <td>Bachelors</td>
      <td>13</td>
      <td>Married-civ-spouse</td>
      <td>Prof-specialty</td>
      <td>Wife</td>
      <td>Black</td>
      <td>Female</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>Cuba</td>
      <td>&lt;=50K</td>
    </tr>
  </tbody>
</table>
</div>




```python
df = pd.read_csv('../data/adult.data', skipinitialspace=True)
df.head()
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
      <th>age</th>
      <th>workclass</th>
      <th>fnlwgt</th>
      <th>education</th>
      <th>education-num</th>
      <th>marital-status</th>
      <th>occupation</th>
      <th>relationship</th>
      <th>race</th>
      <th>sex</th>
      <th>capital-gain</th>
      <th>capital-loss</th>
      <th>hours-per-week</th>
      <th>native-country</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>39</td>
      <td>State-gov</td>
      <td>77516</td>
      <td>Bachelors</td>
      <td>13</td>
      <td>Never-married</td>
      <td>Adm-clerical</td>
      <td>Not-in-family</td>
      <td>White</td>
      <td>Male</td>
      <td>2174</td>
      <td>0</td>
      <td>40</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>1</th>
      <td>50</td>
      <td>Self-emp-not-inc</td>
      <td>83311</td>
      <td>Bachelors</td>
      <td>13</td>
      <td>Married-civ-spouse</td>
      <td>Exec-managerial</td>
      <td>Husband</td>
      <td>White</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>13</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>2</th>
      <td>38</td>
      <td>Private</td>
      <td>215646</td>
      <td>HS-grad</td>
      <td>9</td>
      <td>Divorced</td>
      <td>Handlers-cleaners</td>
      <td>Not-in-family</td>
      <td>White</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>3</th>
      <td>53</td>
      <td>Private</td>
      <td>234721</td>
      <td>11th</td>
      <td>7</td>
      <td>Married-civ-spouse</td>
      <td>Handlers-cleaners</td>
      <td>Husband</td>
      <td>Black</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>4</th>
      <td>28</td>
      <td>Private</td>
      <td>338409</td>
      <td>Bachelors</td>
      <td>13</td>
      <td>Married-civ-spouse</td>
      <td>Prof-specialty</td>
      <td>Wife</td>
      <td>Black</td>
      <td>Female</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>Cuba</td>
      <td>&lt;=50K</td>
    </tr>
  </tbody>
</table>
</div>



So at first we will be creating evaluation set


```python
from sklearn.model_selection import train_test_split
train, test = train_test_split(df, test_size=0.2, shuffle=False)
```


```python
train.head()
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
      <th>age</th>
      <th>workclass</th>
      <th>fnlwgt</th>
      <th>education</th>
      <th>education-num</th>
      <th>marital-status</th>
      <th>occupation</th>
      <th>relationship</th>
      <th>race</th>
      <th>sex</th>
      <th>capital-gain</th>
      <th>capital-loss</th>
      <th>hours-per-week</th>
      <th>native-country</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>39</td>
      <td>State-gov</td>
      <td>77516</td>
      <td>Bachelors</td>
      <td>13</td>
      <td>Never-married</td>
      <td>Adm-clerical</td>
      <td>Not-in-family</td>
      <td>White</td>
      <td>Male</td>
      <td>2174</td>
      <td>0</td>
      <td>40</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>1</th>
      <td>50</td>
      <td>Self-emp-not-inc</td>
      <td>83311</td>
      <td>Bachelors</td>
      <td>13</td>
      <td>Married-civ-spouse</td>
      <td>Exec-managerial</td>
      <td>Husband</td>
      <td>White</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>13</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>2</th>
      <td>38</td>
      <td>Private</td>
      <td>215646</td>
      <td>HS-grad</td>
      <td>9</td>
      <td>Divorced</td>
      <td>Handlers-cleaners</td>
      <td>Not-in-family</td>
      <td>White</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>3</th>
      <td>53</td>
      <td>Private</td>
      <td>234721</td>
      <td>11th</td>
      <td>7</td>
      <td>Married-civ-spouse</td>
      <td>Handlers-cleaners</td>
      <td>Husband</td>
      <td>Black</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>4</th>
      <td>28</td>
      <td>Private</td>
      <td>338409</td>
      <td>Bachelors</td>
      <td>13</td>
      <td>Married-civ-spouse</td>
      <td>Prof-specialty</td>
      <td>Wife</td>
      <td>Black</td>
      <td>Female</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>Cuba</td>
      <td>&lt;=50K</td>
    </tr>
  </tbody>
</table>
</div>




```python
test.head()
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
      <th>age</th>
      <th>workclass</th>
      <th>fnlwgt</th>
      <th>education</th>
      <th>education-num</th>
      <th>marital-status</th>
      <th>occupation</th>
      <th>relationship</th>
      <th>race</th>
      <th>sex</th>
      <th>capital-gain</th>
      <th>capital-loss</th>
      <th>hours-per-week</th>
      <th>native-country</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>26048</th>
      <td>30</td>
      <td>Private</td>
      <td>270886</td>
      <td>Some-college</td>
      <td>10</td>
      <td>Never-married</td>
      <td>Other-service</td>
      <td>Own-child</td>
      <td>White</td>
      <td>Female</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>26049</th>
      <td>21</td>
      <td>Private</td>
      <td>216129</td>
      <td>HS-grad</td>
      <td>9</td>
      <td>Never-married</td>
      <td>Other-service</td>
      <td>Own-child</td>
      <td>White</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>35</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>26050</th>
      <td>33</td>
      <td>Private</td>
      <td>189368</td>
      <td>Some-college</td>
      <td>10</td>
      <td>Married-civ-spouse</td>
      <td>Transport-moving</td>
      <td>Husband</td>
      <td>Black</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>United-States</td>
      <td>&gt;50K</td>
    </tr>
    <tr>
      <th>26051</th>
      <td>19</td>
      <td>?</td>
      <td>141418</td>
      <td>Some-college</td>
      <td>10</td>
      <td>Never-married</td>
      <td>?</td>
      <td>Own-child</td>
      <td>White</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>15</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>26052</th>
      <td>19</td>
      <td>Private</td>
      <td>306225</td>
      <td>HS-grad</td>
      <td>9</td>
      <td>Never-married</td>
      <td>Handlers-cleaners</td>
      <td>Own-child</td>
      <td>White</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>25</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
  </tbody>
</table>
</div>





So now we will see the first part. See train data statistics


```python
train_stats = tfdv.generate_statistics_from_dataframe(train)
tfdv.visualize_statistics(train_stats)
```


<iframe id='facets-iframe' width="100%" height="500px"></iframe>
        <script>
        facets_iframe = document.getElementById('facets-iframe');
        facets_html = '<script src="https://cdnjs.cloudflare.com/ajax/libs/webcomponentsjs/1.3.3/webcomponents-lite.js"><\/script><link rel="import" href="https://raw.githubusercontent.com/PAIR-code/facets/master/facets-dist/facets-jupyter.html"><facets-overview proto-input="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"></facets-overview>';
        facets_iframe.srcdoc = facets_html;
         facets_iframe.id = "";
         setTimeout(() => {
           facets_iframe.setAttribute('height', facets_iframe.contentWindow.document.body.offsetHeight + 'px')
         }, 1500)
         </script>


> So first point is done. From here we can see the statistics and also visualized the train data

# 2. Infer data Schema

* Infer schema actually created from train statistics
* It will just show you expected data type
* Let see it in action


```python
schema = tfdv.infer_schema(statistics=train_stats)
tfdv.display_schema(schema)
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
      <th>Type</th>
      <th>Presence</th>
      <th>Valency</th>
      <th>Domain</th>
    </tr>
    <tr>
      <th>Feature name</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>'age'</th>
      <td>INT</td>
      <td>required</td>
      <td></td>
      <td>-</td>
    </tr>
    <tr>
      <th>'workclass'</th>
      <td>STRING</td>
      <td>required</td>
      <td></td>
      <td>'workclass'</td>
    </tr>
    <tr>
      <th>'fnlwgt'</th>
      <td>INT</td>
      <td>required</td>
      <td></td>
      <td>-</td>
    </tr>
    <tr>
      <th>'education'</th>
      <td>STRING</td>
      <td>required</td>
      <td></td>
      <td>'education'</td>
    </tr>
    <tr>
      <th>'education-num'</th>
      <td>INT</td>
      <td>required</td>
      <td></td>
      <td>-</td>
    </tr>
    <tr>
      <th>'marital-status'</th>
      <td>STRING</td>
      <td>required</td>
      <td></td>
      <td>'marital-status'</td>
    </tr>
    <tr>
      <th>'occupation'</th>
      <td>STRING</td>
      <td>required</td>
      <td></td>
      <td>'occupation'</td>
    </tr>
    <tr>
      <th>'relationship'</th>
      <td>STRING</td>
      <td>required</td>
      <td></td>
      <td>'relationship'</td>
    </tr>
    <tr>
      <th>'race'</th>
      <td>STRING</td>
      <td>required</td>
      <td></td>
      <td>'race'</td>
    </tr>
    <tr>
      <th>'sex'</th>
      <td>STRING</td>
      <td>required</td>
      <td></td>
      <td>'sex'</td>
    </tr>
    <tr>
      <th>'capital-gain'</th>
      <td>INT</td>
      <td>required</td>
      <td></td>
      <td>-</td>
    </tr>
    <tr>
      <th>'capital-loss'</th>
      <td>INT</td>
      <td>required</td>
      <td></td>
      <td>-</td>
    </tr>
    <tr>
      <th>'hours-per-week'</th>
      <td>INT</td>
      <td>required</td>
      <td></td>
      <td>-</td>
    </tr>
    <tr>
      <th>'native-country'</th>
      <td>STRING</td>
      <td>required</td>
      <td></td>
      <td>'native-country'</td>
    </tr>
    <tr>
      <th>'label'</th>
      <td>STRING</td>
      <td>required</td>
      <td></td>
      <td>'label'</td>
    </tr>
  </tbody>
</table>
</div>


    C:\Users\hasan\.conda\envs\tensorflow_validation\lib\site-packages\tensorflow_data_validation\utils\display_util.py:180: FutureWarning: Passing a negative integer is deprecated in version 1.0 and will not be supported in future version. Instead, use None to not limit the column width.
      pd.set_option('max_colwidth', -1)
    


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
      <th>Values</th>
    </tr>
    <tr>
      <th>Domain</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>'workclass'</th>
      <td>'?', 'Federal-gov', 'Local-gov', 'Never-worked', 'Private', 'Self-emp-inc', 'Self-emp-not-inc', 'State-gov', 'Without-pay'</td>
    </tr>
    <tr>
      <th>'education'</th>
      <td>'10th', '11th', '12th', '1st-4th', '5th-6th', '7th-8th', '9th', 'Assoc-acdm', 'Assoc-voc', 'Bachelors', 'Doctorate', 'HS-grad', 'Masters', 'Preschool', 'Prof-school', 'Some-college'</td>
    </tr>
    <tr>
      <th>'marital-status'</th>
      <td>'Divorced', 'Married-AF-spouse', 'Married-civ-spouse', 'Married-spouse-absent', 'Never-married', 'Separated', 'Widowed'</td>
    </tr>
    <tr>
      <th>'occupation'</th>
      <td>'?', 'Adm-clerical', 'Armed-Forces', 'Craft-repair', 'Exec-managerial', 'Farming-fishing', 'Handlers-cleaners', 'Machine-op-inspct', 'Other-service', 'Priv-house-serv', 'Prof-specialty', 'Protective-serv', 'Sales', 'Tech-support', 'Transport-moving'</td>
    </tr>
    <tr>
      <th>'relationship'</th>
      <td>'Husband', 'Not-in-family', 'Other-relative', 'Own-child', 'Unmarried', 'Wife'</td>
    </tr>
    <tr>
      <th>'race'</th>
      <td>'Amer-Indian-Eskimo', 'Asian-Pac-Islander', 'Black', 'Other', 'White'</td>
    </tr>
    <tr>
      <th>'sex'</th>
      <td>'Female', 'Male'</td>
    </tr>
    <tr>
      <th>'native-country'</th>
      <td>'?', 'Cambodia', 'Canada', 'China', 'Columbia', 'Cuba', 'Dominican-Republic', 'Ecuador', 'El-Salvador', 'England', 'France', 'Germany', 'Greece', 'Guatemala', 'Haiti', 'Holand-Netherlands', 'Honduras', 'Hong', 'Hungary', 'India', 'Iran', 'Ireland', 'Italy', 'Jamaica', 'Japan', 'Laos', 'Mexico', 'Nicaragua', 'Outlying-US(Guam-USVI-etc)', 'Peru', 'Philippines', 'Poland', 'Portugal', 'Puerto-Rico', 'Scotland', 'South', 'Taiwan', 'Thailand', 'Trinadad&amp;Tobago', 'United-States', 'Vietnam', 'Yugoslavia'</td>
    </tr>
    <tr>
      <th>'label'</th>
      <td>'&lt;=50K', '&gt;50K'</td>
    </tr>
  </tbody>
</table>
</div>


# 3. Generate and visualize evaluation dataset statistics

- So we already used and saw train statistics what is special here to show.
- Actually tfdv give you to compare it with training and evaluation set together.

  - ``lhs_statistics``
  - ``rhs_statistics``
  - ``lhs_name``
  - ``rhs_name``



```python
test_stats = tfdv.generate_statistics_from_dataframe(test)
tfdv.visualize_statistics(lhs_statistics=test_stats,
                          rhs_statistics=train_stats,
                          lhs_name='TEST_DATASET',
                          rhs_name='TRAIN_DATASET')
```


<iframe id='facets-iframe' width="100%" height="500px"></iframe>
        <script>
        facets_iframe = document.getElementById('facets-iframe');
        facets_html = '<script src="https://cdnjs.cloudflare.com/ajax/libs/webcomponentsjs/1.3.3/webcomponents-lite.js"><\/script><link rel="import" href="https://raw.githubusercontent.com/PAIR-code/facets/master/facets-dist/facets-jupyter.html"><facets-overview proto-input="CsVlCgxURVNUX0RBVEFTRVQQ8TIavAcasgcKtgII8TIYASABLQAAgD8ypAIaGwkAAAAAAADwPxEAAAAAAADwPyFmZmZmZlqEQBobCQAAAAAAAPA/EQAAAAAAAPA/IWZmZmZmWoRAGhsJAAAAAAAA8D8RAAAAAAAA8D8hZmZmZmZahEAaGwkAAAAAAADwPxEAAAAAAADwPyFmZmZmZlqEQBobCQAAAAAAAPA/EQAAAAAAAPA/IWZmZmZmWoRAGhsJAAAAAAAA8D8RAAAAAAAA8D8hZmZmZmZahEAaGwkAAAAAAADwPxEAAAAAAADwPyFmZmZmZlqEQBobCQAAAAAAAPA/EQAAAAAAAPA/IWZmZmZmWoRAGhsJAAAAAAAA8D8RAAAAAAAA8D8hZmZmZmZahEAaGwkAAAAAAADwPxEAAAAAAADwPyFmZmZmZlqEQCABQPEyETIAK1qqMkNAGS2Gtizc7CpAKQAAAAAAADFAMQAAAAAAgEJAOQAAAAAAgFZAQqICGhsJAAAAAAAAMUARzczMzMxMOEAhpgpGJbVukUAaGwnNzMzMzEw4QBGamZmZmZk/QCH2l92TR42SQBobCZqZmZmZmT9AETMzMzMzc0NAIeLplbLMkZNAGhsJMzMzMzNzQ0ARmpmZmZkZR0Ahvw6cM2LBkkAaGwmamZmZmRlHQBEAAAAAAMBKQCFPa5p3HHyIQBobCQAAAAAAwEpAEWZmZmZmZk5AIcfctYR8339AGhsJZmZmZmZmTkARZmZmZmYGUUAhk6mCUUmydUAaGwlmZmZmZgZRQBGamZmZmdlSQCFwK/aX3adVQBobCZqZmZmZ2VJAEc3MzMzMrFRAIdyDns2qszNAGhsJzczMzMysVEARAAAAAACAVkAhBF+YTBWEJEBCpAIaGwkAAAAAAAAxQBEAAAAAAAA2QCFmZmZmZlqEQBobCQAAAAAAADZAEQAAAAAAADpAIWZmZmZmWoRAGhsJAAAAAAAAOkARAAAAAAAAPUAhZmZmZmZahEAaGwkAAAAAAAA9QBEAAAAAAIBAQCFmZmZmZlqEQBobCQAAAAAAgEBAEQAAAAAAgEJAIWZmZmZmWoRAGhsJAAAAAACAQkARAAAAAACAREAhZmZmZmZahEAaGwkAAAAAAIBEQBEAAAAAAIBGQCFmZmZmZlqEQBobCQAAAAAAgEZAEQAAAAAAAElAIWZmZmZmWoRAGhsJAAAAAAAASUARAAAAAAAATUAhZmZmZmZahEAaGwkAAAAAAABNQBEAAAAAAIBWQCFmZmZmZlqEQCABQgUKA2FnZRqPBhACIv0FCrYCCPEyGAEgAS0AAIA/MqQCGhsJAAAAAAAA8D8RAAAAAAAA8D8hZmZmZmZahEAaGwkAAAAAAADwPxEAAAAAAADwPyFmZmZmZlqEQBobCQAAAAAAAPA/EQAAAAAAAPA/IWZmZmZmWoRAGhsJAAAAAAAA8D8RAAAAAAAA8D8hZmZmZmZahEAaGwkAAAAAAADwPxEAAAAAAADwPyFmZmZmZlqEQBobCQAAAAAAAPA/EQAAAAAAAPA/IWZmZmZmWoRAGhsJAAAAAAAA8D8RAAAAAAAA8D8hZmZmZmZahEAaGwkAAAAAAADwPxEAAAAAAADwPyFmZmZmZlqEQBobCQAAAAAAAPA/EQAAAAAAAPA/IWZmZmZmWoRAGhsJAAAAAAAA8D8RAAAAAAAA8D8hZmZmZmZahEAgAUDxMhAJGhISB1ByaXZhdGUZAAAAAADjsUAaGxIQU2VsZi1lbXAtbm90LWluYxkAAAAAAHB+QBoUEglMb2NhbC1nb3YZAAAAAACAeEAaDBIBPxkAAAAAAIB3QBoUEglTdGF0ZS1nb3YZAAAAAABwcEAaFxIMU2VsZi1lbXAtaW5jGQAAAAAAYGtAGhYSC0ZlZGVyYWwtZ292GQAAAAAA4GdAGhYSC1dpdGhvdXQtcGF5GQAAAAAAABBAGhcSDE5ldmVyLXdvcmtlZBkAAAAAAAAAQCX4JfpAKu0BChIiB1ByaXZhdGUpAAAAAADjsUAKHwgBEAEiEFNlbGYtZW1wLW5vdC1pbmMpAAAAAABwfkAKGAgCEAIiCUxvY2FsLWdvdikAAAAAAIB4QAoQCAMQAyIBPykAAAAAAIB3QAoYCAQQBCIJU3RhdGUtZ292KQAAAAAAcHBAChsIBRAFIgxTZWxmLWVtcC1pbmMpAAAAAABga0AKGggGEAYiC0ZlZGVyYWwtZ292KQAAAAAA4GdAChoIBxAHIgtXaXRob3V0LXBheSkAAAAAAAAQQAobCAgQCCIMTmV2ZXItd29ya2VkKQAAAAAAAABAQgsKCXdvcmtjbGFzcxq/BxqyBwq2AgjxMhgBIAEtAACAPzKkAhobCQAAAAAAAPA/EQAAAAAAAPA/IWZmZmZmWoRAGhsJAAAAAAAA8D8RAAAAAAAA8D8hZmZmZmZahEAaGwkAAAAAAADwPxEAAAAAAADwPyFmZmZmZlqEQBobCQAAAAAAAPA/EQAAAAAAAPA/IWZmZmZmWoRAGhsJAAAAAAAA8D8RAAAAAAAA8D8hZmZmZmZahEAaGwkAAAAAAADwPxEAAAAAAADwPyFmZmZmZlqEQBobCQAAAAAAAPA/EQAAAAAAAPA/IWZmZmZmWoRAGhsJAAAAAAAA8D8RAAAAAAAA8D8hZmZmZmZahEAaGwkAAAAAAADwPxEAAAAAAADwPyFmZmZmZlqEQBobCQAAAAAAAPA/EQAAAAAAAPA/IWZmZmZmWoRAIAFA8TIRl7MXdaUzB0EZKO2GmRMW+kApAAAAAIDkykAxAAAAAPjNBUE5AAAAAO0sMUFCogIaGwkAAAAAgOTKQBFmZmZmloH+QCEMLzgEV7KcQBobCWZmZmaWgf5AEWZmZmZO0wxBIVH4ZsCTfqdAGhsJZmZmZk7TDEERzMzMzOgyFUEhZjwv4AckkkAaGwnMzMzM6DIVQRFmZmZmKvwbQSFsxvvyvmp3QBobCWZmZmYq/BtBEQAAAAC2YiFBIS4cFR65I1dAGhsJAAAAALZiIUERzMzMzFbHJEEh7jUEbLxyNkAaGwnMzMzMVsckQRGZmZmZ9ysoQSGvz7hmrssoQBobCZmZmZn3KyhBEWZmZmaYkCtBIT0PlMhTHQBAGhsJZmZmZpiQK0ERMzMzMzn1LkEhPQ+UyFMdAEAaGwkzMzMzOfUuQREAAAAA7SwxQSE9D5TIUx0AQEKkAhobCQAAAACA5MpAEQAAAADATO9AIWZmZmZmWoRAGhsJAAAAAMBM70ARAAAAANCj+UAhZmZmZmZahEAaGwkAAAAA0KP5QBEAAAAAUKL/QCFmZmZmZlqEQBobCQAAAABQov9AEQAAAADwRQNBIWZmZmZmWoRAGhsJAAAAAPBFA0ERAAAAAPjNBUEhZmZmZmZahEAaGwkAAAAA+M0FQREAAAAAwA4IQSFmZmZmZlqEQBobCQAAAADADghBEQAAAAB4/wpBIWZmZmZmWoRAGhsJAAAAAHj/CkERAAAAAHDjD0EhZmZmZmZahEAaGwkAAAAAcOMPQREAAAAA+CYUQSFmZmZmZlqEQBobCQAAAAD4JhRBEQAAAADtLDFBIWZmZmZmWoRAIAFCCAoGZm5sd2d0Gp8IEAIijQgKtgII8TIYASABLQAAgD8ypAIaGwkAAAAAAADwPxEAAAAAAADwPyFmZmZmZlqEQBobCQAAAAAAAPA/EQAAAAAAAPA/IWZmZmZmWoRAGhsJAAAAAAAA8D8RAAAAAAAA8D8hZmZmZmZahEAaGwkAAAAAAADwPxEAAAAAAADwPyFmZmZmZlqEQBobCQAAAAAAAPA/EQAAAAAAAPA/IWZmZmZmWoRAGhsJAAAAAAAA8D8RAAAAAAAA8D8hZmZmZmZahEAaGwkAAAAAAADwPxEAAAAAAADwPyFmZmZmZlqEQBobCQAAAAAAAPA/EQAAAAAAAPA/IWZmZmZmWoRAGhsJAAAAAAAA8D8RAAAAAAAA8D8hZmZmZmZahEAaGwkAAAAAAADwPxEAAAAAAADwPyFmZmZmZlqEQCABQPEyEBAaEhIHSFMtZ3JhZBkAAAAAACKgQBoXEgxTb21lLWNvbGxlZ2UZAAAAAADglkAaFBIJQmFjaGVsb3JzGQAAAAAAYJBAGhISB01hc3RlcnMZAAAAAACwdkAaFBIJQXNzb2Mtdm9jGQAAAAAAkHFAGg8SBDExdGgZAAAAAADAbEAaFRIKQXNzb2MtYWNkbRkAAAAAAMBrQBoPEgQxMHRoGQAAAAAAIGZAGhISBzd0aC04dGgZAAAAAAAgYEAaFhILUHJvZi1zY2hvb2wZAAAAAADAX0AaDhIDOXRoGQAAAAAAAFlAGg8SBDEydGgZAAAAAABAV0AaFBIJRG9jdG9yYXRlGQAAAAAAAFRAGhISBzV0aC02dGgZAAAAAACAU0AaEhIHMXN0LTR0aBkAAAAAAABHQBoUEglQcmVzY2hvb2wZAAAAAAAAJEAlvysHQSqDAwoSIgdIUy1ncmFkKQAAAAAAIqBAChsIARABIgxTb21lLWNvbGxlZ2UpAAAAAADglkAKGAgCEAIiCUJhY2hlbG9ycykAAAAAAGCQQAoWCAMQAyIHTWFzdGVycykAAAAAALB2QAoYCAQQBCIJQXNzb2Mtdm9jKQAAAAAAkHFAChMIBRAFIgQxMXRoKQAAAAAAwGxAChkIBhAGIgpBc3NvYy1hY2RtKQAAAAAAwGtAChMIBxAHIgQxMHRoKQAAAAAAIGZAChYICBAIIgc3dGgtOHRoKQAAAAAAIGBAChoICRAJIgtQcm9mLXNjaG9vbCkAAAAAAMBfQAoSCAoQCiIDOXRoKQAAAAAAAFlAChMICxALIgQxMnRoKQAAAAAAQFdAChgIDBAMIglEb2N0b3JhdGUpAAAAAAAAVEAKFggNEA0iBzV0aC02dGgpAAAAAACAU0AKFggOEA4iBzFzdC00dGgpAAAAAAAAR0AKGAgPEA8iCVByZXNjaG9vbCkAAAAAAAAkQEILCgllZHVjYXRpb24axgcasgcKtgII8TIYASABLQAAgD8ypAIaGwkAAAAAAADwPxEAAAAAAADwPyFmZmZmZlqEQBobCQAAAAAAAPA/EQAAAAAAAPA/IWZmZmZmWoRAGhsJAAAAAAAA8D8RAAAAAAAA8D8hZmZmZmZahEAaGwkAAAAAAADwPxEAAAAAAADwPyFmZmZmZlqEQBobCQAAAAAAAPA/EQAAAAAAAPA/IWZmZmZmWoRAGhsJAAAAAAAA8D8RAAAAAAAA8D8hZmZmZmZahEAaGwkAAAAAAADwPxEAAAAAAADwPyFmZmZmZlqEQBobCQAAAAAAAPA/EQAAAAAAAPA/IWZmZmZmWoRAGhsJAAAAAAAA8D8RAAAAAAAA8D8hZmZmZmZahEAaGwkAAAAAAADwPxEAAAAAAADwPyFmZmZmZlqEQCABQPEyESxTfHUDKCRAGT0uD5VD6ARAKQAAAAAAAPA/MQAAAAAAACRAOQAAAAAAADBAQqICGhsJAAAAAAAA8D8RAAAAAAAABEAhoBov3SSuS0AaGwkAAAAAAAAEQBEAAAAAAAAQQCFnZmZmZlpUQBobCQAAAAAAABBAEQAAAAAAABZAITEIrBxaFmxAGhsJAAAAAAAAFkARAAAAAAAAHEAhPQrXo3BjZkAaGwkAAAAAAAAcQBEAAAAAAAAhQCGe76fGSyZ0QBobCQAAAAAAACFAEQAAAAAAACRAIXsUrkfBJ6BAGhsJAAAAAAAAJEARAAAAAAAAJ0AhXI/C9eg4m0AaGwkAAAAAAAAnQBEAAAAAAAAqQCExCKwcWhZsQBobCQAAAAAAACpAEQAAAAAAAC1AIV66SQxCCJZAGhsJAAAAAAAALUARAAAAAAAAMEAhyHa+nxqlaUBCpAIaGwkAAAAAAADwPxEAAAAAAAAcQCFmZmZmZlqEQBobCQAAAAAAABxAEQAAAAAAACJAIWZmZmZmWoRAGhsJAAAAAAAAIkARAAAAAAAAIkAhZmZmZmZahEAaGwkAAAAAAAAiQBEAAAAAAAAiQCFmZmZmZlqEQBobCQAAAAAAACJAEQAAAAAAACRAIWZmZmZmWoRAGhsJAAAAAAAAJEARAAAAAAAAJEAhZmZmZmZahEAaGwkAAAAAAAAkQBEAAAAAAAAmQCFmZmZmZlqEQBobCQAAAAAAACZAEQAAAAAAACpAIWZmZmZmWoRAGhsJAAAAAAAAKkARAAAAAAAAKkAhZmZmZmZahEAaGwkAAAAAAAAqQBEAAAAAAAAwQCFmZmZmZlqEQCABQg8KDWVkdWNhdGlvbi1udW0a4gUQAiLLBQq2AgjxMhgBIAEtAACAPzKkAhobCQAAAAAAAPA/EQAAAAAAAPA/IWZmZmZmWoRAGhsJAAAAAAAA8D8RAAAAAAAA8D8hZmZmZmZahEAaGwkAAAAAAADwPxEAAAAAAADwPyFmZmZmZlqEQBobCQAAAAAAAPA/EQAAAAAAAPA/IWZmZmZmWoRAGhsJAAAAAAAA8D8RAAAAAAAA8D8hZmZmZmZahEAaGwkAAAAAAADwPxEAAAAAAADwPyFmZmZmZlqEQBobCQAAAAAAAPA/EQAAAAAAAPA/IWZmZmZmWoRAGhsJAAAAAAAA8D8RAAAAAAAA8D8hZmZmZmZahEAaGwkAAAAAAADwPxEAAAAAAADwPyFmZmZmZlqEQBobCQAAAAAAAPA/EQAAAAAAAPA/IWZmZmZmWoRAIAFA8TIQBxodEhJNYXJyaWVkLWNpdi1zcG91c2UZAAAAAADMp0AaGBINTmV2ZXItbWFycmllZBkAAAAAALKgQBoTEghEaXZvcmNlZBkAAAAAABCLQBoUEglTZXBhcmF0ZWQZAAAAAADgaEAaEhIHV2lkb3dlZBkAAAAAAIBnQBogEhVNYXJyaWVkLXNwb3VzZS1hYnNlbnQZAAAAAAAAUkAaHBIRTWFycmllZC1BRi1zcG91c2UZAAAAAAAAFEAly4NnQSrQAQodIhJNYXJyaWVkLWNpdi1zcG91c2UpAAAAAADMp0AKHAgBEAEiDU5ldmVyLW1hcnJpZWQpAAAAAACyoEAKFwgCEAIiCERpdm9yY2VkKQAAAAAAEItAChgIAxADIglTZXBhcmF0ZWQpAAAAAADgaEAKFggEEAQiB1dpZG93ZWQpAAAAAACAZ0AKJAgFEAUiFU1hcnJpZWQtc3BvdXNlLWFic2VudCkAAAAAAABSQAogCAYQBiIRTWFycmllZC1BRi1zcG91c2UpAAAAAAAAFEBCEAoObWFyaXRhbC1zdGF0dXMakgkQAiL/CAq2AgjxMhgBIAEtAACAPzKkAhobCQAAAAAAAPA/EQAAAAAAAPA/IWZmZmZmWoRAGhsJAAAAAAAA8D8RAAAAAAAA8D8hZmZmZmZahEAaGwkAAAAAAADwPxEAAAAAAADwPyFmZmZmZlqEQBobCQAAAAAAAPA/EQAAAAAAAPA/IWZmZmZmWoRAGhsJAAAAAAAA8D8RAAAAAAAA8D8hZmZmZmZahEAaGwkAAAAAAADwPxEAAAAAAADwPyFmZmZmZlqEQBobCQAAAAAAAPA/EQAAAAAAAPA/IWZmZmZmWoRAGhsJAAAAAAAA8D8RAAAAAAAA8D8hZmZmZmZahEAaGwkAAAAAAADwPxEAAAAAAADwPyFmZmZmZlqEQBobCQAAAAAAAPA/EQAAAAAAAPA/IWZmZmZmWoRAIAFA8TIQDxoaEg9FeGVjLW1hbmFnZXJpYWwZAAAAAAC4ikAaFxIMQ3JhZnQtcmVwYWlyGQAAAAAAQIpAGhkSDlByb2Ytc3BlY2lhbHR5GQAAAAAAiIlAGhASBVNhbGVzGQAAAAAAgIZAGhcSDEFkbS1jbGVyaWNhbBkAAAAAAKCFQBoYEg1PdGhlci1zZXJ2aWNlGQAAAAAA2INAGhwSEU1hY2hpbmUtb3AtaW5zcGN0GQAAAAAAYHlAGgwSAT8ZAAAAAACgd0AaGxIQVHJhbnNwb3J0LW1vdmluZxkAAAAAALBzQBocEhFIYW5kbGVycy1jbGVhbmVycxkAAAAAALBzQBoaEg9GYXJtaW5nLWZpc2hpbmcZAAAAAACAaEAaFxIMVGVjaC1zdXBwb3J0GQAAAAAAQGhAGhoSD1Byb3RlY3RpdmUtc2VydhkAAAAAAABfQBoaEg9Qcml2LWhvdXNlLXNlcnYZAAAAAAAAOUAaFxIMQXJtZWQtRm9yY2VzGQAAAAAAAPA/JfzFQ0EqugMKGiIPRXhlYy1tYW5hZ2VyaWFsKQAAAAAAuIpAChsIARABIgxDcmFmdC1yZXBhaXIpAAAAAABAikAKHQgCEAIiDlByb2Ytc3BlY2lhbHR5KQAAAAAAiIlAChQIAxADIgVTYWxlcykAAAAAAICGQAobCAQQBCIMQWRtLWNsZXJpY2FsKQAAAAAAoIVAChwIBRAFIg1PdGhlci1zZXJ2aWNlKQAAAAAA2INACiAIBhAGIhFNYWNoaW5lLW9wLWluc3BjdCkAAAAAAGB5QAoQCAcQByIBPykAAAAAAKB3QAofCAgQCCIQVHJhbnNwb3J0LW1vdmluZykAAAAAALBzQAogCAkQCSIRSGFuZGxlcnMtY2xlYW5lcnMpAAAAAACwc0AKHggKEAoiD0Zhcm1pbmctZmlzaGluZykAAAAAAIBoQAobCAsQCyIMVGVjaC1zdXBwb3J0KQAAAAAAQGhACh4IDBAMIg9Qcm90ZWN0aXZlLXNlcnYpAAAAAAAAX0AKHggNEA0iD1ByaXYtaG91c2Utc2VydikAAAAAAAA5QAobCA4QDiIMQXJtZWQtRm9yY2VzKQAAAAAAAPA/QgwKCm9jY3VwYXRpb24a+AQQAiLjBAq2AgjxMhgBIAEtAACAPzKkAhobCQAAAAAAAPA/EQAAAAAAAPA/IWZmZmZmWoRAGhsJAAAAAAAA8D8RAAAAAAAA8D8hZmZmZmZahEAaGwkAAAAAAADwPxEAAAAAAADwPyFmZmZmZlqEQBobCQAAAAAAAPA/EQAAAAAAAPA/IWZmZmZmWoRAGhsJAAAAAAAA8D8RAAAAAAAA8D8hZmZmZmZahEAaGwkAAAAAAADwPxEAAAAAAADwPyFmZmZmZlqEQBobCQAAAAAAAPA/EQAAAAAAAPA/IWZmZmZmWoRAGhsJAAAAAAAA8D8RAAAAAAAA8D8hZmZmZmZahEAaGwkAAAAAAADwPxEAAAAAAADwPyFmZmZmZlqEQBobCQAAAAAAAPA/EQAAAAAAAPA/IWZmZmZmWoRAIAFA8TIQBhoSEgdIdXNiYW5kGQAAAAAADqVAGhgSDU5vdC1pbi1mYW1pbHkZAAAAAADMmEAaFBIJT3duLWNoaWxkGQAAAAAAoI9AGhQSCVVubWFycmllZBkAAAAAALiFQBoPEgRXaWZlGQAAAAAAIHNAGhkSDk90aGVyLXJlbGF0aXZlGQAAAAAAQGtAJcVFEUEqmgEKEiIHSHVzYmFuZCkAAAAAAA6lQAocCAEQASINTm90LWluLWZhbWlseSkAAAAAAMyYQAoYCAIQAiIJT3duLWNoaWxkKQAAAAAAoI9AChgIAxADIglVbm1hcnJpZWQpAAAAAAC4hUAKEwgEEAQiBFdpZmUpAAAAAAAgc0AKHQgFEAUiDk90aGVyLXJlbGF0aXZlKQAAAAAAQGtAQg4KDHJlbGF0aW9uc2hpcBrIBBACIrsECrYCCPEyGAEgAS0AAIA/MqQCGhsJAAAAAAAA8D8RAAAAAAAA8D8hZmZmZmZahEAaGwkAAAAAAADwPxEAAAAAAADwPyFmZmZmZlqEQBobCQAAAAAAAPA/EQAAAAAAAPA/IWZmZmZmWoRAGhsJAAAAAAAA8D8RAAAAAAAA8D8hZmZmZmZahEAaGwkAAAAAAADwPxEAAAAAAADwPyFmZmZmZlqEQBobCQAAAAAAAPA/EQAAAAAAAPA/IWZmZmZmWoRAGhsJAAAAAAAA8D8RAAAAAAAA8D8hZmZmZmZahEAaGwkAAAAAAADwPxEAAAAAAADwPyFmZmZmZlqEQBobCQAAAAAAAPA/EQAAAAAAAPA/IWZmZmZmWoRAGhsJAAAAAAAA8D8RAAAAAAAA8D8hZmZmZmZahEAgAUDxMhAFGhASBVdoaXRlGQAAAAAAnrVAGhASBUJsYWNrGQAAAAAAOIRAGh0SEkFzaWFuLVBhYy1Jc2xhbmRlchkAAAAAAMBrQBodEhJBbWVyLUluZGlhbi1Fc2tpbW8ZAAAAAACATUAaEBIFT3RoZXIZAAAAAACASUAlt/KxQCqEAQoQIgVXaGl0ZSkAAAAAAJ61QAoUCAEQASIFQmxhY2spAAAAAAA4hEAKIQgCEAIiEkFzaWFuLVBhYy1Jc2xhbmRlcikAAAAAAMBrQAohCAMQAyISQW1lci1JbmRpYW4tRXNraW1vKQAAAAAAgE1AChQIBBAEIgVPdGhlcikAAAAAAIBJQEIGCgRyYWNlGpoDEAIijgMKtgII8TIYASABLQAAgD8ypAIaGwkAAAAAAADwPxEAAAAAAADwPyFmZmZmZlqEQBobCQAAAAAAAPA/EQAAAAAAAPA/IWZmZmZmWoRAGhsJAAAAAAAA8D8RAAAAAAAA8D8hZmZmZmZahEAaGwkAAAAAAADwPxEAAAAAAADwPyFmZmZmZlqEQBobCQAAAAAAAPA/EQAAAAAAAPA/IWZmZmZmWoRAGhsJAAAAAAAA8D8RAAAAAAAA8D8hZmZmZmZahEAaGwkAAAAAAADwPxEAAAAAAADwPyFmZmZmZlqEQBobCQAAAAAAAPA/EQAAAAAAAPA/IWZmZmZmWoRAGhsJAAAAAAAA8D8RAAAAAAAA8D8hZmZmZmZahEAaGwkAAAAAAADwPxEAAAAAAADwPyFmZmZmZlqEQCABQPEyEAIaDxIETWFsZRkAAAAAAAexQBoREgZGZW1hbGUZAAAAAADUoEAlkiqVQCooCg8iBE1hbGUpAAAAAAAHsUAKFQgBEAEiBkZlbWFsZSkAAAAAANSgQEIFCgNzZXgagQYa7gUKtgII8TIYASABLQAAgD8ypAIaGwkAAAAAAADwPxEAAAAAAADwPyFmZmZmZlqEQBobCQAAAAAAAPA/EQAAAAAAAPA/IWZmZmZmWoRAGhsJAAAAAAAA8D8RAAAAAAAA8D8hZmZmZmZahEAaGwkAAAAAAADwPxEAAAAAAADwPyFmZmZmZlqEQBobCQAAAAAAAPA/EQAAAAAAAPA/IWZmZmZmWoRAGhsJAAAAAAAA8D8RAAAAAAAA8D8hZmZmZmZahEAaGwkAAAAAAADwPxEAAAAAAADwPyFmZmZmZlqEQBobCQAAAAAAAPA/EQAAAAAAAPA/IWZmZmZmWoRAGhsJAAAAAAAA8D8RAAAAAAAA8D8hZmZmZmZahEAaGwkAAAAAAADwPxEAAAAAAADwPyFmZmZmZlqEQCABQPEyERWsRprb1Y9AGVuOdfjiRLtAINsuOQAAAADwafhAQpkCGhIRMzMzM/OHw0AhPgDQE7reuEAaGwkzMzMz84fDQBEzMzMz84fTQCEqHI1FaSdZQBobCTMzMzPzh9NAEczMzMzsS91AIUX6gGSekypAGhsJzMzMzOxL3UARMzMzM/OH40AhOyNrAbfg7D8aGwkzMzMz84fjQBEAAAAA8GnoQCE7I2sBt+DsPxobCQAAAADwaehAEczMzMzsS+1AITUjawG34Ow/GhsJzMzMzOxL7UARzczMzPQW8UAhQCNrAbfg7D8aGwnNzMzM9BbxQBEzMzMz84fzQCE1I2sBt+DsPxobCTMzMzPzh/NAEZmZmZnx+PVAITUjawG34Ow/GhsJmZmZmfH49UARAAAAAPBp+EAhdL1Gl1X0OkBCeRoJIWZmZmZmWoRAGgkhZmZmZmZahEAaCSFmZmZmZlqEQBoJIWZmZmZmWoRAGgkhZmZmZmZahEAaCSFmZmZmZlqEQBoJIWZmZmZmWoRAGgkhZmZmZmZahEAaCSFmZmZmZlqEQBoSEQAAAADwafhAIWZmZmZmWoRAIAFCDgoMY2FwaXRhbC1nYWluGoEGGu4FCrYCCPEyGAEgAS0AAIA/MqQCGhsJAAAAAAAA8D8RAAAAAAAA8D8hZmZmZmZahEAaGwkAAAAAAADwPxEAAAAAAADwPyFmZmZmZlqEQBobCQAAAAAAAPA/EQAAAAAAAPA/IWZmZmZmWoRAGhsJAAAAAAAA8D8RAAAAAAAA8D8hZmZmZmZahEAaGwkAAAAAAADwPxEAAAAAAADwPyFmZmZmZlqEQBobCQAAAAAAAPA/EQAAAAAAAPA/IWZmZmZmWoRAGhsJAAAAAAAA8D8RAAAAAAAA8D8hZmZmZmZahEAaGwkAAAAAAADwPxEAAAAAAADwPyFmZmZmZlqEQBobCQAAAAAAAPA/EQAAAAAAAPA/IWZmZmZmWoRAGhsJAAAAAAAA8D8RAAAAAAAA8D8hZmZmZmZahEAgAUDxMhFVh928a3tWQBl1xRykBnx5QCC3MDkAAAAAAASxQEKZAhoSEZqZmZmZOXtAIc3hiGRUObhAGhsJmpmZmZk5e0ARmpmZmZk5i0AhT5eLpqmDDUAaGwmamZmZmTmLQBE0MzMzM2uUQCHtSlylYDAZQBobCTQzMzMza5RAEZqZmZmZOZtAIWWheUMWWFdAGhsJmpmZmZk5m0ARAAAAAAAEoUAhGuC6CGTUZEAaGwkAAAAAAAShQBE0MzMzM2ukQCG4XWPkKplBQBobCTQzMzMza6RAEWdmZmZm0qdAIcetxfH75Pk/GhsJZ2ZmZmbSp0ARmpmZmZk5q0Ahx63F8fvk+T8aGwmamZmZmTmrQBHNzMzMzKCuQCHHrcXx++T5PxobCc3MzMzMoK5AEQAAAAAABLFAIcetxfH75Pk/QnkaCSFmZmZmZlqEQBoJIWZmZmZmWoRAGgkhZmZmZmZahEAaCSFmZmZmZlqEQBoJIWZmZmZmWoRAGgkhZmZmZmZahEAaCSFmZmZmZlqEQBoJIWZmZmZmWoRAGgkhZmZmZmZahEAaEhEAAAAAAASxQCFmZmZmZlqEQCABQg4KDGNhcGl0YWwtbG9zcxrHBxqyBwq2AgjxMhgBIAEtAACAPzKkAhobCQAAAAAAAPA/EQAAAAAAAPA/IWZmZmZmWoRAGhsJAAAAAAAA8D8RAAAAAAAA8D8hZmZmZmZahEAaGwkAAAAAAADwPxEAAAAAAADwPyFmZmZmZlqEQBobCQAAAAAAAPA/EQAAAAAAAPA/IWZmZmZmWoRAGhsJAAAAAAAA8D8RAAAAAAAA8D8hZmZmZmZahEAaGwkAAAAAAADwPxEAAAAAAADwPyFmZmZmZlqEQBobCQAAAAAAAPA/EQAAAAAAAPA/IWZmZmZmWoRAGhsJAAAAAAAA8D8RAAAAAAAA8D8hZmZmZmZahEAaGwkAAAAAAADwPxEAAAAAAADwPyFmZmZmZlqEQBobCQAAAAAAAPA/EQAAAAAAAPA/IWZmZmZmWoRAIAFA8TIR0WtZbUJGREAZAXx38HgCKUApAAAAAAAA8D8xAAAAAAAAREA5AAAAAADAWEBCogIaGwkAAAAAAADwPxGamZmZmZklQCGAt0CC4o9iQBobCZqZmZmZmSVAEZqZmZmZmTRAIWmR7Xw/4ntAGhsJmpmZmZmZNEARZ2ZmZmZmPkAhyZi7lpC4e0AaGwlnZmZmZmY+QBGamZmZmRlEQCFxGw3gDU+rQBobCZqZmZmZGURAEQAAAAAAAElAIcE5I0r77INAGhsJAAAAAAAASUARZ2ZmZmbmTUAhMuauJeQCikAaGwlnZmZmZuZNQBFnZmZmZmZRQCGtHFpkO/t1QBobCWdmZmZmZlFAEZqZmZmZ2VNAIS3/If32oVZAGhsJmpmZmZnZU0ARzczMzMxMVkAhIwaBlUPTQEAaGwnNzMzMzExWQBEAAAAAAMBYQCEPC7WmebdAQEKkAhobCQAAAAAAAPA/EQAAAAAAADhAIWZmZmZmWoRAGhsJAAAAAAAAOEARAAAAAACAQUAhZmZmZmZahEAaGwkAAAAAAIBBQBEAAAAAAABEQCFmZmZmZlqEQBobCQAAAAAAAERAEQAAAAAAAERAIWZmZmZmWoRAGhsJAAAAAAAAREARAAAAAAAAREAhZmZmZmZahEAaGwkAAAAAAABEQBEAAAAAAABEQCFmZmZmZlqEQBobCQAAAAAAAERAEQAAAAAAAEVAIWZmZmZmWoRAGhsJAAAAAAAARUARAAAAAAAASUAhZmZmZmZahEAaGwkAAAAAAABJQBEAAAAAAIBLQCFmZmZmZlqEQBobCQAAAAAAgEtAEQAAAAAAwFhAIWZmZmZmWoRAIAFCEAoOaG91cnMtcGVyLXdlZWsa5w0QAiLQDQq2AgjxMhgBIAEtAACAPzKkAhobCQAAAAAAAPA/EQAAAAAAAPA/IWZmZmZmWoRAGhsJAAAAAAAA8D8RAAAAAAAA8D8hZmZmZmZahEAaGwkAAAAAAADwPxEAAAAAAADwPyFmZmZmZlqEQBobCQAAAAAAAPA/EQAAAAAAAPA/IWZmZmZmWoRAGhsJAAAAAAAA8D8RAAAAAAAA8D8hZmZmZmZahEAaGwkAAAAAAADwPxEAAAAAAADwPyFmZmZmZlqEQBobCQAAAAAAAPA/EQAAAAAAAPA/IWZmZmZmWoRAGhsJAAAAAAAA8D8RAAAAAAAA8D8hZmZmZmZahEAaGwkAAAAAAADwPxEAAAAAAADwPyFmZmZmZlqEQBobCQAAAAAAAPA/EQAAAAAAAPA/IWZmZmZmWoRAIAFA8TIQKRoYEg1Vbml0ZWQtU3RhdGVzGQAAAAAAuLZAGhESBk1leGljbxkAAAAAAMBgQBoMEgE/GQAAAAAAQF1AGhYSC1BoaWxpcHBpbmVzGQAAAAAAgENAGhISB0dlcm1hbnkZAAAAAAAAP0AaFhILRWwtU2FsdmFkb3IZAAAAAAAAPkAaEBIFSW5kaWEZAAAAAAAAPUAaDxIEQ3ViYRkAAAAAAAA1QBoSEgdKYW1haWNhGQAAAAAAADRAGhASBUl0YWx5GQAAAAAAADJAGhESBkNhbmFkYRkAAAAAAAAyQBoSEgdWaWV0bmFtGQAAAAAAAC5AGh0SEkRvbWluaWNhbi1SZXB1YmxpYxkAAAAAAAAuQBoWEgtQdWVydG8tUmljbxkAAAAAAAAsQBoQEgVTb3V0aBkAAAAAAAAqQBoTEghDb2x1bWJpYRkAAAAAAAAqQBoQEgVKYXBhbhkAAAAAAAAoQBoSEgdFbmdsYW5kGQAAAAAAAChAGhASBUNoaW5hGQAAAAAAAChAGhQSCUd1YXRlbWFsYRkAAAAAAAAmQCXsmURBKvIHChgiDVVuaXRlZC1TdGF0ZXMpAAAAAAC4tkAKFQgBEAEiBk1leGljbykAAAAAAMBgQAoQCAIQAiIBPykAAAAAAEBdQAoaCAMQAyILUGhpbGlwcGluZXMpAAAAAACAQ0AKFggEEAQiB0dlcm1hbnkpAAAAAAAAP0AKGggFEAUiC0VsLVNhbHZhZG9yKQAAAAAAAD5AChQIBhAGIgVJbmRpYSkAAAAAAAA9QAoTCAcQByIEQ3ViYSkAAAAAAAA1QAoWCAgQCCIHSmFtYWljYSkAAAAAAAA0QAoUCAkQCSIFSXRhbHkpAAAAAAAAMkAKFQgKEAoiBkNhbmFkYSkAAAAAAAAyQAoWCAsQCyIHVmlldG5hbSkAAAAAAAAuQAohCAwQDCISRG9taW5pY2FuLVJlcHVibGljKQAAAAAAAC5AChoIDRANIgtQdWVydG8tUmljbykAAAAAAAAsQAoUCA4QDiIFU291dGgpAAAAAAAAKkAKFwgPEA8iCENvbHVtYmlhKQAAAAAAACpAChQIEBAQIgVKYXBhbikAAAAAAAAoQAoWCBEQESIHRW5nbGFuZCkAAAAAAAAoQAoUCBIQEiIFQ2hpbmEpAAAAAAAAKEAKGAgTEBMiCUd1YXRlbWFsYSkAAAAAAAAmQAoXCBQQFCIIUG9ydHVnYWwpAAAAAAAAIkAKFggVEBUiB0VjdWFkb3IpAAAAAAAAIkAKEwgWEBYiBFBlcnUpAAAAAAAAIEAKEwgXEBciBExhb3MpAAAAAAAAIEAKEwgYEBgiBEhvbmcpAAAAAAAAIEAKHggZEBkiD1RyaW5hZGFkJlRvYmFnbykAAAAAAAAcQAoVCBoQGiIGVGFpd2FuKQAAAAAAABxAChUIGxAbIgZQb2xhbmQpAAAAAAAAHEAKGAgcEBwiCU5pY2FyYWd1YSkAAAAAAAAcQAopCB0QHSIaT3V0bHlpbmctVVMoR3VhbS1VU1ZJLWV0YykpAAAAAAAAGEAKFAgeEB4iBUhhaXRpKQAAAAAAABhAChUIHxAfIgZHcmVlY2UpAAAAAAAAGEAKFQggECAiBkZyYW5jZSkAAAAAAAAYQAoZCCEQISIKWXVnb3NsYXZpYSkAAAAAAAAUQAoTCCIQIiIESXJhbikAAAAAAAAUQAoWCCMQIyIHSXJlbGFuZCkAAAAAAAAQQAoXCCQQJCIISG9uZHVyYXMpAAAAAAAAEEAKFwglECUiCFNjb3RsYW5kKQAAAAAAAAhAChYIJhAmIgdIdW5nYXJ5KQAAAAAAAAhAChcIJxAnIghDYW1ib2RpYSkAAAAAAAAIQAoXCCgQKCIIVGhhaWxhbmQpAAAAAAAAAEBCEAoObmF0aXZlLWNvdW50cnkamgMQAiKMAwq2AgjxMhgBIAEtAACAPzKkAhobCQAAAAAAAPA/EQAAAAAAAPA/IWZmZmZmWoRAGhsJAAAAAAAA8D8RAAAAAAAA8D8hZmZmZmZahEAaGwkAAAAAAADwPxEAAAAAAADwPyFmZmZmZlqEQBobCQAAAAAAAPA/EQAAAAAAAPA/IWZmZmZmWoRAGhsJAAAAAAAA8D8RAAAAAAAA8D8hZmZmZmZahEAaGwkAAAAAAADwPxEAAAAAAADwPyFmZmZmZlqEQBobCQAAAAAAAPA/EQAAAAAAAPA/IWZmZmZmWoRAGhsJAAAAAAAA8D8RAAAAAAAA8D8hZmZmZmZahEAaGwkAAAAAAADwPxEAAAAAAADwPyFmZmZmZlqEQBobCQAAAAAAAPA/EQAAAAAAAPA/IWZmZmZmWoRAIAFA8TIQAhoQEgU8PTUwSxkAAAAAADGzQBoPEgQ+NTBLGQAAAAAAAJlAJYgjmEAqJwoQIgU8PTUwSykAAAAAADGzQAoTCAEQASIEPjUwSykAAAAAAACZQEIHCgVsYWJlbAqIZgoNVFJBSU5fREFUQVNFVBDAywEavgcatAcKuAIIwMsBGAEgAS0AAIA/MqQCGhsJAAAAAAAA8D8RAAAAAAAA8D8hmpmZmZlZpEAaGwkAAAAAAADwPxEAAAAAAADwPyGamZmZmVmkQBobCQAAAAAAAPA/EQAAAAAAAPA/IZqZmZmZWaRAGhsJAAAAAAAA8D8RAAAAAAAA8D8hmpmZmZlZpEAaGwkAAAAAAADwPxEAAAAAAADwPyGamZmZmVmkQBobCQAAAAAAAPA/EQAAAAAAAPA/IZqZmZmZWaRAGhsJAAAAAAAA8D8RAAAAAAAA8D8hmpmZmZlZpEAaGwkAAAAAAADwPxEAAAAAAADwPyGamZmZmVmkQBobCQAAAAAAAPA/EQAAAAAAAPA/IZqZmZmZWaRAGhsJAAAAAAAA8D8RAAAAAAAA8D8hmpmZmZlZpEAgAUDAywERPhKV5WVQQ0AZZNqfGCVeK0ApAAAAAAAAMUAxAAAAAACAQkA5AAAAAACAVkBCogIaGwkAAAAAAAAxQBHNzMzMzEw4QCE5RUdy+VOxQBobCc3MzMzMTDhAEZqZmZmZmT9AIYNRSZ2AcrJAGhsJmpmZmZmZP0ARMzMzMzNzQ0AhImx4eqXAskAaGwkzMzMzM3NDQBGamZmZmRlHQCFiodY071yzQBobCZqZmZmZGUdAEQAAAAAAwEpAIXo2qz5X46hAGhsJAAAAAADASkARZmZmZmZmTkAhAU2EDU9XoEAaGwlmZmZmZmZOQBFmZmZmZgZRQCGNBvAWSECTQBobCWZmZmZmBlFAEZqZmZmZ2VJAIb2vA+eM6HhAGhsJmpmZmZnZUkARzczMzMysVEAhUMGopE4AZUAaGwnNzMzMzKxUQBEAAAAAAIBWQCFqrIvbaABMQEKkAhobCQAAAAAAADFAEQAAAAAAADZAIZmZmZmZWaRAGhsJAAAAAAAANkARAAAAAAAAOkAhmZmZmZlZpEAaGwkAAAAAAAA6QBEAAAAAAAA+QCGZmZmZmVmkQBobCQAAAAAAAD5AEQAAAAAAgEBAIZmZmZmZWaRAGhsJAAAAAACAQEARAAAAAACAQkAhmZmZmZlZpEAaGwkAAAAAAIBCQBEAAAAAAIBEQCGZmZmZmVmkQBobCQAAAAAAgERAEQAAAAAAgEZAIZmZmZmZWaRAGhsJAAAAAACARkARAAAAAAAASUAhmZmZmZlZpEAaGwkAAAAAAABJQBEAAAAAAABNQCGZmZmZmVmkQBobCQAAAAAAAE1AEQAAAAAAgFZAIZmZmZmZWaRAIAFCBQoDYWdlGpEGEAIi/wUKuAIIwMsBGAEgAS0AAIA/MqQCGhsJAAAAAAAA8D8RAAAAAAAA8D8hmpmZmZlZpEAaGwkAAAAAAADwPxEAAAAAAADwPyGamZmZmVmkQBobCQAAAAAAAPA/EQAAAAAAAPA/IZqZmZmZWaRAGhsJAAAAAAAA8D8RAAAAAAAA8D8hmpmZmZlZpEAaGwkAAAAAAADwPxEAAAAAAADwPyGamZmZmVmkQBobCQAAAAAAAPA/EQAAAAAAAPA/IZqZmZmZWaRAGhsJAAAAAAAA8D8RAAAAAAAA8D8hmpmZmZlZpEAaGwkAAAAAAADwPxEAAAAAAADwPyGamZmZmVmkQBobCQAAAAAAAPA/EQAAAAAAAPA/IZqZmZmZWaRAGhsJAAAAAAAA8D8RAAAAAAAA8D8hmpmZmZlZpEAgAUDAywEQCRoSEgdQcml2YXRlGQAAAABAsdFAGhsSEFNlbGYtZW1wLW5vdC1pbmMZAAAAAAAMoEAaFBIJTG9jYWwtZ292GQAAAAAAlJpAGgwSAT8ZAAAAAADQlkAaFBIJU3RhdGUtZ292GQAAAAAALJBAGhcSDFNlbGYtZW1wLWluYxkAAAAAAAiMQBoWEgtGZWRlcmFsLWdvdhkAAAAAAAiIQBoWEgtXaXRob3V0LXBheRkAAAAAAAAkQBoXEgxOZXZlci13b3JrZWQZAAAAAAAAFEAlEAr8QCrtAQoSIgdQcml2YXRlKQAAAABAsdFACh8IARABIhBTZWxmLWVtcC1ub3QtaW5jKQAAAAAADKBAChgIAhACIglMb2NhbC1nb3YpAAAAAACUmkAKEAgDEAMiAT8pAAAAAADQlkAKGAgEEAQiCVN0YXRlLWdvdikAAAAAACyQQAobCAUQBSIMU2VsZi1lbXAtaW5jKQAAAAAACIxAChoIBhAGIgtGZWRlcmFsLWdvdikAAAAAAAiIQAoaCAcQByILV2l0aG91dC1wYXkpAAAAAAAAJEAKGwgIEAgiDE5ldmVyLXdvcmtlZCkAAAAAAAAUQEILCgl3b3JrY2xhc3MawQcatAcKuAIIwMsBGAEgAS0AAIA/MqQCGhsJAAAAAAAA8D8RAAAAAAAA8D8hmpmZmZlZpEAaGwkAAAAAAADwPxEAAAAAAADwPyGamZmZmVmkQBobCQAAAAAAAPA/EQAAAAAAAPA/IZqZmZmZWaRAGhsJAAAAAAAA8D8RAAAAAAAA8D8hmpmZmZlZpEAaGwkAAAAAAADwPxEAAAAAAADwPyGamZmZmVmkQBobCQAAAAAAAPA/EQAAAAAAAPA/IZqZmZmZWaRAGhsJAAAAAAAA8D8RAAAAAAAA8D8hmpmZmZlZpEAaGwkAAAAAAADwPxEAAAAAAADwPyGamZmZmVmkQBobCQAAAAAAAPA/EQAAAAAAAPA/IZqZmZmZWaRAGhsJAAAAAAAA8D8RAAAAAAAA8D8hmpmZmZlZpEAgAUDAywERjAcsNk4oB0EZpCGuOEew+UApAAAAAID+x0AxAAAAAHjEBUE5AAAAAKGnNkFCogIaGwkAAAAAgP7HQBEAAAAAOHkDQSF4E1rhgHjEQBobCQAAAAA4eQNBEQAAAABEuRJBIZuSSoEFBshAGhsJAAAAAES5EkERAAAAAOy1G0Eh371jdRnNpUAaGwkAAAAA7LUbQREAAAAASlkiQSGExDwuevJ2QBobCQAAAABKWSJBEQAAAACe1yZBIXxsh/TM2lNAGhsJAAAAAJ7XJkERAAAAAPJVK0EhqxybPBdkIUAaGwkAAAAA8lUrQREAAAAARtQvQSERXZ2wnSEVQBobCQAAAABG1C9BEQAAAABNKTJBIRFdnbCdIRVAGhsJAAAAAE0pMkERAAAAAHdoNEEhEV2dsJ0hFUAaGwkAAAAAd2g0QREAAAAAoac2QSERXZ2wnSEVQEKkAhobCQAAAACA/sdAEQAAAACQOvBAIZmZmZmZWaRAGhsJAAAAAJA68EARAAAAAEAZ+kAhmZmZmZlZpEAaGwkAAAAAQBn6QBEAAAAA8AQAQSGZmZmZmVmkQBobCQAAAADwBABBEQAAAABYYwNBIZmZmZmZWaRAGhsJAAAAAFhjA0ERAAAAAHjEBUEhmZmZmZlZpEAaGwkAAAAAeMQFQREAAAAAAPYHQSGZmZmZmVmkQBobCQAAAAAA9gdBEQAAAACwxQpBIZmZmZmZWaRAGhsJAAAAALDFCkERAAAAAGCuD0EhmZmZmZlZpEAaGwkAAAAAYK4PQREAAAAAXA8UQSGZmZmZmVmkQBobCQAAAABcDxRBEQAAAAChpzZBIZmZmZmZWaRAIAFCCAoGZm5sd2d0GqEIEAIijwgKuAIIwMsBGAEgAS0AAIA/MqQCGhsJAAAAAAAA8D8RAAAAAAAA8D8hmpmZmZlZpEAaGwkAAAAAAADwPxEAAAAAAADwPyGamZmZmVmkQBobCQAAAAAAAPA/EQAAAAAAAPA/IZqZmZmZWaRAGhsJAAAAAAAA8D8RAAAAAAAA8D8hmpmZmZlZpEAaGwkAAAAAAADwPxEAAAAAAADwPyGamZmZmVmkQBobCQAAAAAAAPA/EQAAAAAAAPA/IZqZmZmZWaRAGhsJAAAAAAAA8D8RAAAAAAAA8D8hmpmZmZlZpEAaGwkAAAAAAADwPxEAAAAAAADwPyGamZmZmVmkQBobCQAAAAAAAPA/EQAAAAAAAPA/IZqZmZmZWaRAGhsJAAAAAAAA8D8RAAAAAAAA8D8hmpmZmZlZpEAgAUDAywEQEBoSEgdIUy1ncmFkGQAAAAAAesBAGhcSDFNvbWUtY29sbGVnZRkAAAAAAMO2QBoUEglCYWNoZWxvcnMZAAAAAADTsEAaEhIHTWFzdGVycxkAAAAAAECVQBoUEglBc3NvYy12b2MZAAAAAAA0kUAaDxIEMTF0aBkAAAAAAIiNQBoVEgpBc3NvYy1hY2RtGQAAAAAAaIpAGg8SBDEwdGgZAAAAAACgh0AaEhIHN3RoLTh0aBkAAAAAACiAQBoWEgtQcm9mLXNjaG9vbBkAAAAAABB8QBoOEgM5dGgZAAAAAADgeUAaDxIEMTJ0aBkAAAAAAEB1QBoUEglEb2N0b3JhdGUZAAAAAADQdEAaEhIHNXRoLTZ0aBkAAAAAAOBvQBoSEgcxc3QtNHRoGQAAAAAAgF5AGhQSCVByZXNjaG9vbBkAAAAAAIBEQCWn4QZBKoMDChIiB0hTLWdyYWQpAAAAAAB6wEAKGwgBEAEiDFNvbWUtY29sbGVnZSkAAAAAAMO2QAoYCAIQAiIJQmFjaGVsb3JzKQAAAAAA07BAChYIAxADIgdNYXN0ZXJzKQAAAAAAQJVAChgIBBAEIglBc3NvYy12b2MpAAAAAAA0kUAKEwgFEAUiBDExdGgpAAAAAACIjUAKGQgGEAYiCkFzc29jLWFjZG0pAAAAAABoikAKEwgHEAciBDEwdGgpAAAAAACgh0AKFggIEAgiBzd0aC04dGgpAAAAAAAogEAKGggJEAkiC1Byb2Ytc2Nob29sKQAAAAAAEHxAChIIChAKIgM5dGgpAAAAAADgeUAKEwgLEAsiBDEydGgpAAAAAABAdUAKGAgMEAwiCURvY3RvcmF0ZSkAAAAAANB0QAoWCA0QDSIHNXRoLTZ0aCkAAAAAAOBvQAoWCA4QDiIHMXN0LTR0aCkAAAAAAIBeQAoYCA8QDyIJUHJlc2Nob29sKQAAAAAAgERAQgsKCWVkdWNhdGlvbhrIBxq0Bwq4AgjAywEYASABLQAAgD8ypAIaGwkAAAAAAADwPxEAAAAAAADwPyGamZmZmVmkQBobCQAAAAAAAPA/EQAAAAAAAPA/IZqZmZmZWaRAGhsJAAAAAAAA8D8RAAAAAAAA8D8hmpmZmZlZpEAaGwkAAAAAAADwPxEAAAAAAADwPyGamZmZmVmkQBobCQAAAAAAAPA/EQAAAAAAAPA/IZqZmZmZWaRAGhsJAAAAAAAA8D8RAAAAAAAA8D8hmpmZmZlZpEAaGwkAAAAAAADwPxEAAAAAAADwPyGamZmZmVmkQBobCQAAAAAAAPA/EQAAAAAAAPA/IZqZmZmZWaRAGhsJAAAAAAAA8D8RAAAAAAAA8D8hmpmZmZlZpEAaGwkAAAAAAADwPxEAAAAAAADwPyGamZmZmVmkQCABQMDLAREm3K6moSkkQBna4JkkyX8EQCkAAAAAAADwPzEAAAAAAAAkQDkAAAAAAAAwQEKiAhobCQAAAAAAAPA/EQAAAAAAAARAIUOLbOf7KWVAGhsJAAAAAAAABEARAAAAAAAAEEAh8dJNYhAYcUAaGwkAAAAAAAAQQBEAAAAAAAAWQCFoke18PxWMQBobCQAAAAAAABZAEQAAAAAAABxAIRbZzvdTA4hAGhsJAAAAAAAAHEARAAAAAAAAIUAhLt0kBoEllEAaGwkAAAAAAAAhQBEAAAAAAAAkQCE/NV66SYLAQBobCQAAAAAAACRAEQAAAAAAACdAIQRWDi2y6bpAGhsJAAAAAAAAJ0ARAAAAAAAAKkAhvp8aL91Ei0AaGwkAAAAAAAAqQBEAAAAAAAAtQCEIrBxaZAe2QBobCQAAAAAAAC1AEQAAAAAAADBAIcDKoUW204hAQqQCGhsJAAAAAAAA8D8RAAAAAAAAHEAhmZmZmZlZpEAaGwkAAAAAAAAcQBEAAAAAAAAiQCGZmZmZmVmkQBobCQAAAAAAACJAEQAAAAAAACJAIZmZmZmZWaRAGhsJAAAAAAAAIkARAAAAAAAAIkAhmZmZmZlZpEAaGwkAAAAAAAAiQBEAAAAAAAAkQCGZmZmZmVmkQBobCQAAAAAAACRAEQAAAAAAACRAIZmZmZmZWaRAGhsJAAAAAAAAJEARAAAAAAAAJkAhmZmZmZlZpEAaGwkAAAAAAAAmQBEAAAAAAAAqQCGZmZmZmVmkQBobCQAAAAAAACpAEQAAAAAAACpAIZmZmZmZWaRAGhsJAAAAAAAAKkARAAAAAAAAMEAhmZmZmZlZpEAgAUIPCg1lZHVjYXRpb24tbnVtGuQFEAIizQUKuAIIwMsBGAEgAS0AAIA/MqQCGhsJAAAAAAAA8D8RAAAAAAAA8D8hmpmZmZlZpEAaGwkAAAAAAADwPxEAAAAAAADwPyGamZmZmVmkQBobCQAAAAAAAPA/EQAAAAAAAPA/IZqZmZmZWaRAGhsJAAAAAAAA8D8RAAAAAAAA8D8hmpmZmZlZpEAaGwkAAAAAAADwPxEAAAAAAADwPyGamZmZmVmkQBobCQAAAAAAAPA/EQAAAAAAAPA/IZqZmZmZWaRAGhsJAAAAAAAA8D8RAAAAAAAA8D8hmpmZmZlZpEAaGwkAAAAAAADwPxEAAAAAAADwPyGamZmZmVmkQBobCQAAAAAAAPA/EQAAAAAAAPA/IZqZmZmZWaRAGhsJAAAAAAAA8D8RAAAAAAAA8D8hmpmZmZlZpEAgAUDAywEQBxodEhJNYXJyaWVkLWNpdi1zcG91c2UZAAAAAABNx0AaGBINTmV2ZXItbWFycmllZBkAAAAAALHAQBoTEghEaXZvcmNlZBkAAAAAAPKrQBoUEglTZXBhcmF0ZWQZAAAAAADQiUAaEhIHV2lkb3dlZBkAAAAAACiJQBogEhVNYXJyaWVkLXNwb3VzZS1hYnNlbnQZAAAAAACgdUAaHBIRTWFycmllZC1BRi1zcG91c2UZAAAAAAAAMkAl/2ZmQSrQAQodIhJNYXJyaWVkLWNpdi1zcG91c2UpAAAAAABNx0AKHAgBEAEiDU5ldmVyLW1hcnJpZWQpAAAAAACxwEAKFwgCEAIiCERpdm9yY2VkKQAAAAAA8qtAChgIAxADIglTZXBhcmF0ZWQpAAAAAADQiUAKFggEEAQiB1dpZG93ZWQpAAAAAAAoiUAKJAgFEAUiFU1hcnJpZWQtc3BvdXNlLWFic2VudCkAAAAAAKB1QAogCAYQBiIRTWFycmllZC1BRi1zcG91c2UpAAAAAAAAMkBCEAoObWFyaXRhbC1zdGF0dXMalAkQAiKBCQq4AgjAywEYASABLQAAgD8ypAIaGwkAAAAAAADwPxEAAAAAAADwPyGamZmZmVmkQBobCQAAAAAAAPA/EQAAAAAAAPA/IZqZmZmZWaRAGhsJAAAAAAAA8D8RAAAAAAAA8D8hmpmZmZlZpEAaGwkAAAAAAADwPxEAAAAAAADwPyGamZmZmVmkQBobCQAAAAAAAPA/EQAAAAAAAPA/IZqZmZmZWaRAGhsJAAAAAAAA8D8RAAAAAAAA8D8hmpmZmZlZpEAaGwkAAAAAAADwPxEAAAAAAADwPyGamZmZmVmkQBobCQAAAAAAAPA/EQAAAAAAAPA/IZqZmZmZWaRAGhsJAAAAAAAA8D8RAAAAAAAA8D8hmpmZmZlZpEAaGwkAAAAAAADwPxEAAAAAAADwPyGamZmZmVmkQCABQMDLARAPGhkSDlByb2Ytc3BlY2lhbHR5GQAAAAAA9qlAGhcSDENyYWZ0LXJlcGFpchkAAAAAAHapQBoaEg9FeGVjLW1hbmFnZXJpYWwZAAAAAAAWqUAaFxIMQWRtLWNsZXJpY2FsGQAAAAAADKhAGhASBVNhbGVzGQAAAAAA5KZAGhgSDU90aGVyLXNlcnZpY2UZAAAAAADIpEAaHBIRTWFjaGluZS1vcC1pbnNwY3QZAAAAAADwmEAaDBIBPxkAAAAAAOSWQBobEhBUcmFuc3BvcnQtbW92aW5nGQAAAAAACJRAGhwSEUhhbmRsZXJzLWNsZWFuZXJzGQAAAAAAfJBAGhoSD0Zhcm1pbmctZmlzaGluZxkAAAAAAPCIQBoXEgxUZWNoLXN1cHBvcnQZAAAAAADwhkAaGhIPUHJvdGVjdGl2ZS1zZXJ2GQAAAAAAaIBAGhoSD1ByaXYtaG91c2Utc2VydhkAAAAAAABfQBoXEgxBcm1lZC1Gb3JjZXMZAAAAAAAAIEAlNxhDQSq6AwoZIg5Qcm9mLXNwZWNpYWx0eSkAAAAAAPapQAobCAEQASIMQ3JhZnQtcmVwYWlyKQAAAAAAdqlACh4IAhACIg9FeGVjLW1hbmFnZXJpYWwpAAAAAAAWqUAKGwgDEAMiDEFkbS1jbGVyaWNhbCkAAAAAAAyoQAoUCAQQBCIFU2FsZXMpAAAAAADkpkAKHAgFEAUiDU90aGVyLXNlcnZpY2UpAAAAAADIpEAKIAgGEAYiEU1hY2hpbmUtb3AtaW5zcGN0KQAAAAAA8JhAChAIBxAHIgE/KQAAAAAA5JZACh8ICBAIIhBUcmFuc3BvcnQtbW92aW5nKQAAAAAACJRACiAICRAJIhFIYW5kbGVycy1jbGVhbmVycykAAAAAAHyQQAoeCAoQCiIPRmFybWluZy1maXNoaW5nKQAAAAAA8IhAChsICxALIgxUZWNoLXN1cHBvcnQpAAAAAADwhkAKHggMEAwiD1Byb3RlY3RpdmUtc2VydikAAAAAAGiAQAoeCA0QDSIPUHJpdi1ob3VzZS1zZXJ2KQAAAAAAAF9AChsIDhAOIgxBcm1lZC1Gb3JjZXMpAAAAAAAAIEBCDAoKb2NjdXBhdGlvbhr6BBACIuUECrgCCMDLARgBIAEtAACAPzKkAhobCQAAAAAAAPA/EQAAAAAAAPA/IZqZmZmZWaRAGhsJAAAAAAAA8D8RAAAAAAAA8D8hmpmZmZlZpEAaGwkAAAAAAADwPxEAAAAAAADwPyGamZmZmVmkQBobCQAAAAAAAPA/EQAAAAAAAPA/IZqZmZmZWaRAGhsJAAAAAAAA8D8RAAAAAAAA8D8hmpmZmZlZpEAaGwkAAAAAAADwPxEAAAAAAADwPyGamZmZmVmkQBobCQAAAAAAAPA/EQAAAAAAAPA/IZqZmZmZWaRAGhsJAAAAAAAA8D8RAAAAAAAA8D8hmpmZmZlZpEAaGwkAAAAAAADwPxEAAAAAAADwPyGamZmZmVmkQBobCQAAAAAAAPA/EQAAAAAAAPA/IZqZmZmZWaRAIAFAwMsBEAYaEhIHSHVzYmFuZBkAAAAAAIHEQBoYEg1Ob3QtaW4tZmFtaWx5GQAAAAAAPrpAGhQSCU93bi1jaGlsZBkAAAAAALCvQBoUEglVbm1hcnJpZWQZAAAAAAB+pUAaDxIEV2lmZRkAAAAAALiTQBoZEg5PdGhlci1yZWxhdGl2ZRkAAAAAANiHQCWoExJBKpoBChIiB0h1c2JhbmQpAAAAAACBxEAKHAgBEAEiDU5vdC1pbi1mYW1pbHkpAAAAAAA+ukAKGAgCEAIiCU93bi1jaGlsZCkAAAAAALCvQAoYCAMQAyIJVW5tYXJyaWVkKQAAAAAAfqVAChMIBBAEIgRXaWZlKQAAAAAAuJNACh0IBRAFIg5PdGhlci1yZWxhdGl2ZSkAAAAAANiHQEIOCgxyZWxhdGlvbnNoaXAaygQQAiK9BAq4AgjAywEYASABLQAAgD8ypAIaGwkAAAAAAADwPxEAAAAAAADwPyGamZmZmVmkQBobCQAAAAAAAPA/EQAAAAAAAPA/IZqZmZmZWaRAGhsJAAAAAAAA8D8RAAAAAAAA8D8hmpmZmZlZpEAaGwkAAAAAAADwPxEAAAAAAADwPyGamZmZmVmkQBobCQAAAAAAAPA/EQAAAAAAAPA/IZqZmZmZWaRAGhsJAAAAAAAA8D8RAAAAAAAA8D8hmpmZmZlZpEAaGwkAAAAAAADwPxEAAAAAAADwPyGamZmZmVmkQBobCQAAAAAAAPA/EQAAAAAAAPA/IZqZmZmZWaRAGhsJAAAAAAAA8D8RAAAAAAAA8D8hmpmZmZlZpEAaGwkAAAAAAADwPxEAAAAAAADwPyGamZmZmVmkQCABQMDLARAFGhASBVdoaXRlGQAAAACAwtVAGhASBUJsYWNrGQAAAAAAWqNAGh0SEkFzaWFuLVBhYy1Jc2xhbmRlchkAAAAAAIiJQBodEhJBbWVyLUluZGlhbi1Fc2tpbW8ZAAAAAACAb0AaEBIFT3RoZXIZAAAAAACAa0AljhKxQCqEAQoQIgVXaGl0ZSkAAAAAgMLVQAoUCAEQASIFQmxhY2spAAAAAABao0AKIQgCEAIiEkFzaWFuLVBhYy1Jc2xhbmRlcikAAAAAAIiJQAohCAMQAyISQW1lci1JbmRpYW4tRXNraW1vKQAAAAAAgG9AChQIBBAEIgVPdGhlcikAAAAAAIBrQEIGCgRyYWNlGpwDEAIikAMKuAIIwMsBGAEgAS0AAIA/MqQCGhsJAAAAAAAA8D8RAAAAAAAA8D8hmpmZmZlZpEAaGwkAAAAAAADwPxEAAAAAAADwPyGamZmZmVmkQBobCQAAAAAAAPA/EQAAAAAAAPA/IZqZmZmZWaRAGhsJAAAAAAAA8D8RAAAAAAAA8D8hmpmZmZlZpEAaGwkAAAAAAADwPxEAAAAAAADwPyGamZmZmVmkQBobCQAAAAAAAPA/EQAAAAAAAPA/IZqZmZmZWaRAGhsJAAAAAAAA8D8RAAAAAAAA8D8hmpmZmZlZpEAaGwkAAAAAAADwPxEAAAAAAADwPyGamZmZmVmkQBobCQAAAAAAAPA/EQAAAAAAAPA/IZqZmZmZWaRAGhsJAAAAAAAA8D8RAAAAAAAA8D8hmpmZmZlZpEAgAUDAywEQAhoPEgRNYWxlGQAAAADABdFAGhESBkZlbWFsZRkAAAAAgNTAQCUILJVAKigKDyIETWFsZSkAAAAAwAXRQAoVCAEQASIGRmVtYWxlKQAAAACA1MBAQgUKA3NleBqEBhrxBQq4AgjAywEYASABLQAAgD8ypAIaGwkAAAAAAADwPxEAAAAAAADwPyGamZmZmVmkQBobCQAAAAAAAPA/EQAAAAAAAPA/IZqZmZmZWaRAGhsJAAAAAAAA8D8RAAAAAAAA8D8hmpmZmZlZpEAaGwkAAAAAAADwPxEAAAAAAADwPyGamZmZmVmkQBobCQAAAAAAAPA/EQAAAAAAAPA/IZqZmZmZWaRAGhsJAAAAAAAA8D8RAAAAAAAA8D8hmpmZmZlZpEAaGwkAAAAAAADwPxEAAAAAAADwPyGamZmZmVmkQBobCQAAAAAAAPA/EQAAAAAAAPA/IZqZmZmZWaRAGhsJAAAAAAAA8D8RAAAAAAAA8D8hmpmZmZlZpEAaGwkAAAAAAADwPxEAAAAAAADwPyGamZmZmVmkQCABQMDLARHEeMBihRGRQBm7QhglyDq9QCC+ugE5AAAAAPBp+EBCmQIaEhEzMzMz84fDQCHXN5hJsNbYQBobCTMzMzPzh8NAETMzMzPzh9NAIVDzM+B+6XpAGhsJMzMzM/OH00ARzMzMzOxL3UAhNMxJGNklO0AaGwnMzMzM7EvdQBEzMzMz84fjQCExYm5tlN8MQBobCTMzMzPzh+NAEQAAAADwaehAITFibm2U3wxAGhsJAAAAAPBp6EARzMzMzOxL7UAhK2JubZTfDEAaGwnMzMzM7EvtQBHNzMzM9BbxQCE3Ym5tlN8MQBobCc3MzMz0FvFAETMzMzPzh/NAIStibm2U3wxAGhsJMzMzM/OH80ARmZmZmfH49UAhK2JubZTfDEAaGwmZmZmZ8fj1QBEAAAAA8Gn4QCHQmjBmLLtgQEJ5GgkhmZmZmZlZpEAaCSGZmZmZmVmkQBoJIZmZmZmZWaRAGgkhmZmZmZlZpEAaCSGZmZmZmVmkQBoJIZmZmZmZWaRAGgkhmZmZmZlZpEAaCSGZmZmZmVmkQBoJIZmZmZmZWaRAGhIRAAAAAPBp+EAhmZmZmZlZpEAgAUIOCgxjYXBpdGFsLWdhaW4ahAYa8QUKuAIIwMsBGAEgAS0AAIA/MqQCGhsJAAAAAAAA8D8RAAAAAAAA8D8hmpmZmZlZpEAaGwkAAAAAAADwPxEAAAAAAADwPyGamZmZmVmkQBobCQAAAAAAAPA/EQAAAAAAAPA/IZqZmZmZWaRAGhsJAAAAAAAA8D8RAAAAAAAA8D8hmpmZmZlZpEAaGwkAAAAAAADwPxEAAAAAAADwPyGamZmZmVmkQBobCQAAAAAAAPA/EQAAAAAAAPA/IZqZmZmZWaRAGhsJAAAAAAAA8D8RAAAAAAAA8D8hmpmZmZlZpEAaGwkAAAAAAADwPxEAAAAAAADwPyGamZmZmVmkQBobCQAAAAAAAPA/EQAAAAAAAPA/IZqZmZmZWaRAGhsJAAAAAAAA8D8RAAAAAAAA8D8hmpmZmZlZpEAgAUDAywERBqEA2XGpVUAZZ1LmkuIbeUAgi8IBOQAAAAAABLFAQpkCGhIRmpmZmZk5e0AhbZC1zptE2EAaGwmamZmZmTl7QBGamZmZmTmLQCGO/gONDtwwQBobCZqZmZmZOYtAETQzMzMza5RAIdjuMUGAQDpAGhsJNDMzMzNrlEARmpmZmZk5m0AhG2MDh+i0dUAaGwmamZmZmTmbQBEAAAAAAAShQCHAT/qJ3ISDQBobCQAAAAAABKFAETQzMzMza6RAIZj409sNqWNAGhsJNDMzMzNrpEARZ2ZmZmbSp0AhYGxoza1BGUAaGwlnZmZmZtKnQBGamZmZmTmrQCFgbGjNrUEZQBobCZqZmZmZOatAEc3MzMzMoK5AIWBsaM2tQRlAGhsJzczMzMygrkARAAAAAAAEsUAhYGxoza1BGUBCeRoJIZmZmZmZWaRAGgkhmZmZmZlZpEAaCSGZmZmZmVmkQBoJIZmZmZmZWaRAGgkhmZmZmZlZpEAaCSGZmZmZmVmkQBoJIZmZmZmZWaRAGgkhmZmZmZlZpEAaCSGZmZmZmVmkQBoSEQAAAAAABLFAIZmZmZmZWaRAIAFCDgoMY2FwaXRhbC1sb3NzGskHGrQHCrgCCMDLARgBIAEtAACAPzKkAhobCQAAAAAAAPA/EQAAAAAAAPA/IZqZmZmZWaRAGhsJAAAAAAAA8D8RAAAAAAAA8D8hmpmZmZlZpEAaGwkAAAAAAADwPxEAAAAAAADwPyGamZmZmVmkQBobCQAAAAAAAPA/EQAAAAAAAPA/IZqZmZmZWaRAGhsJAAAAAAAA8D8RAAAAAAAA8D8hmpmZmZlZpEAaGwkAAAAAAADwPxEAAAAAAADwPyGamZmZmVmkQBobCQAAAAAAAPA/EQAAAAAAAPA/IZqZmZmZWaRAGhsJAAAAAAAA8D8RAAAAAAAA8D8hmpmZmZlZpEAaGwkAAAAAAADwPxEAAAAAAADwPyGamZmZmVmkQBobCQAAAAAAAPA/EQAAAAAAAPA/IZqZmZmZWaRAIAFAwMsBEaEA2XFtNERAGc0cTp5enShAKQAAAAAAAPA/MQAAAAAAAERAOQAAAAAAwFhAQqICGhsJAAAAAAAA8D8RmpmZmZmZJUAheVioNc07gkAaGwmamZmZmZklQBGamZmZmZk0QCG1hHzQs1mbQBobCZqZmZmZmTRAEWdmZmZmZj5AIfkx5q4lJJ1AGhsJZ2ZmZmZmPkARmpmZmZkZREAh46WbxCDUy0AaGwmamZmZmRlEQBEAAAAAAABJQCF0tRX7y36jQBobCQAAAAAAAElAEWdmZmZm5k1AIW6jAbwFyqdAGhsJZ2ZmZmbmTUARZ2ZmZmZmUUAh2r+y0qRklkAaGwlnZmZmZmZRQBGamZmZmdlTQCHRHcTOFLp2QBobCZqZmZmZ2VNAEc3MzMzMTFZAIWrH5ygafmNAGhsJzczMzMxMVkARAAAAAADAWEAhjL/Hc5DqWkBCpAIaGwkAAAAAAADwPxEAAAAAAAA4QCGZmZmZmVmkQBobCQAAAAAAADhAEQAAAAAAgEFAIZmZmZmZWaRAGhsJAAAAAACAQUARAAAAAAAAREAhmZmZmZlZpEAaGwkAAAAAAABEQBEAAAAAAABEQCGZmZmZmVmkQBobCQAAAAAAAERAEQAAAAAAAERAIZmZmZmZWaRAGhsJAAAAAAAAREARAAAAAAAAREAhmZmZmZlZpEAaGwkAAAAAAABEQBEAAAAAAABEQCGZmZmZmVmkQBobCQAAAAAAAERAEQAAAAAAAEhAIZmZmZmZWaRAGhsJAAAAAAAASEARAAAAAACAS0AhmZmZmZlZpEAaGwkAAAAAAIBLQBEAAAAAAMBYQCGZmZmZmVmkQCABQhAKDmhvdXJzLXBlci13ZWVrGooOEAIi8w0KuAIIwMsBGAEgAS0AAIA/MqQCGhsJAAAAAAAA8D8RAAAAAAAA8D8hmpmZmZlZpEAaGwkAAAAAAADwPxEAAAAAAADwPyGamZmZmVmkQBobCQAAAAAAAPA/EQAAAAAAAPA/IZqZmZmZWaRAGhsJAAAAAAAA8D8RAAAAAAAA8D8hmpmZmZlZpEAaGwkAAAAAAADwPxEAAAAAAADwPyGamZmZmVmkQBobCQAAAAAAAPA/EQAAAAAAAPA/IZqZmZmZWaRAGhsJAAAAAAAA8D8RAAAAAAAA8D8hmpmZmZlZpEAaGwkAAAAAAADwPxEAAAAAAADwPyGamZmZmVmkQBobCQAAAAAAAPA/EQAAAAAAAPA/IZqZmZmZWaRAGhsJAAAAAAAA8D8RAAAAAAAA8D8hmpmZmZlZpEAgAUDAywEQKhoYEg1Vbml0ZWQtU3RhdGVzGQAAAACAztZAGhESBk1leGljbxkAAAAAANB/QBoMEgE/GQAAAAAAIH1AGhYSC1BoaWxpcHBpbmVzGQAAAAAA4GNAGhISB0dlcm1hbnkZAAAAAACAWkAaERIGQ2FuYWRhGQAAAAAAwFlAGhYSC1B1ZXJ0by1SaWNvGQAAAAAAAFlAGhISB0VuZ2xhbmQZAAAAAACAU0AaFhILRWwtU2FsdmFkb3IZAAAAAAAAU0AaDxIEQ3ViYRkAAAAAAIBSQBoQEgVJbmRpYRkAAAAAAMBRQBoQEgVTb3V0aBkAAAAAAMBQQBoQEgVDaGluYRkAAAAAAIBPQBoSEgdKYW1haWNhGQAAAAAAgE5AGhASBUl0YWx5GQAAAAAAgEtAGh0SEkRvbWluaWNhbi1SZXB1YmxpYxkAAAAAAIBLQBoREgZQb2xhbmQZAAAAAACASkAaFBIJR3VhdGVtYWxhGQAAAAAAgEpAGhISB1ZpZXRuYW0ZAAAAAAAASkAaEBIFSmFwYW4ZAAAAAAAASUAlBrpEQSqVCAoYIg1Vbml0ZWQtU3RhdGVzKQAAAACAztZAChUIARABIgZNZXhpY28pAAAAAADQf0AKEAgCEAIiAT8pAAAAAAAgfUAKGggDEAMiC1BoaWxpcHBpbmVzKQAAAAAA4GNAChYIBBAEIgdHZXJtYW55KQAAAAAAgFpAChUIBRAFIgZDYW5hZGEpAAAAAADAWUAKGggGEAYiC1B1ZXJ0by1SaWNvKQAAAAAAAFlAChYIBxAHIgdFbmdsYW5kKQAAAAAAgFNAChoICBAIIgtFbC1TYWx2YWRvcikAAAAAAABTQAoTCAkQCSIEQ3ViYSkAAAAAAIBSQAoUCAoQCiIFSW5kaWEpAAAAAADAUUAKFAgLEAsiBVNvdXRoKQAAAAAAwFBAChQIDBAMIgVDaGluYSkAAAAAAIBPQAoWCA0QDSIHSmFtYWljYSkAAAAAAIBOQAoUCA4QDiIFSXRhbHkpAAAAAACAS0AKIQgPEA8iEkRvbWluaWNhbi1SZXB1YmxpYykAAAAAAIBLQAoVCBAQECIGUG9sYW5kKQAAAAAAgEpAChgIERARIglHdWF0ZW1hbGEpAAAAAACASkAKFggSEBIiB1ZpZXRuYW0pAAAAAAAASkAKFAgTEBMiBUphcGFuKQAAAAAAAElAChcIFBAUIghDb2x1bWJpYSkAAAAAAABHQAoVCBUQFSIGVGFpd2FuKQAAAAAAAEZAChMIFhAWIgRJcmFuKQAAAAAAAENAChQIFxAXIgVIYWl0aSkAAAAAAABDQAoXCBgQGCIIUG9ydHVnYWwpAAAAAAAAPEAKGAgZEBkiCU5pY2FyYWd1YSkAAAAAAAA7QAoTCBoQGiIEUGVydSkAAAAAAAA3QAoVCBsQGyIGR3JlZWNlKQAAAAAAADdAChUIHBAcIgZGcmFuY2UpAAAAAAAAN0AKFggdEB0iB0lyZWxhbmQpAAAAAAAANEAKFggeEB4iB0VjdWFkb3IpAAAAAAAAM0AKFwgfEB8iCFRoYWlsYW5kKQAAAAAAADBAChcIIBAgIghDYW1ib2RpYSkAAAAAAAAwQAoeCCEQISIPVHJpbmFkYWQmVG9iYWdvKQAAAAAAAChAChMIIhAiIgRIb25nKQAAAAAAAChAChkIIxAjIgpZdWdvc2xhdmlhKQAAAAAAACZAChMIJBAkIgRMYW9zKQAAAAAAACRAChYIJRAlIgdIdW5nYXJ5KQAAAAAAACRAChcIJhAmIghTY290bGFuZCkAAAAAAAAiQAoXCCcQJyIISG9uZHVyYXMpAAAAAAAAIkAKKQgoECgiGk91dGx5aW5nLVVTKEd1YW0tVVNWSS1ldGMpKQAAAAAAACBACiEIKRApIhJIb2xhbmQtTmV0aGVybGFuZHMpAAAAAAAA8D9CEAoObmF0aXZlLWNvdW50cnkanAMQAiKOAwq4AgjAywEYASABLQAAgD8ypAIaGwkAAAAAAADwPxEAAAAAAADwPyGamZmZmVmkQBobCQAAAAAAAPA/EQAAAAAAAPA/IZqZmZmZWaRAGhsJAAAAAAAA8D8RAAAAAAAA8D8hmpmZmZlZpEAaGwkAAAAAAADwPxEAAAAAAADwPyGamZmZmVmkQBobCQAAAAAAAPA/EQAAAAAAAPA/IZqZmZmZWaRAGhsJAAAAAAAA8D8RAAAAAAAA8D8hmpmZmZlZpEAaGwkAAAAAAADwPxEAAAAAAADwPyGamZmZmVmkQBobCQAAAAAAAPA/EQAAAAAAAPA/IZqZmZmZWaRAGhsJAAAAAAAA8D8RAAAAAAAA8D8hmpmZmZlZpEAaGwkAAAAAAADwPxEAAAAAAADwPyGamZmZmVmkQCABQMDLARACGhASBTw9NTBLGQAAAADAV9NAGg8SBD41MEsZAAAAAABhuEAlOlWYQConChAiBTw9NTBLKQAAAADAV9NAChMIARABIgQ+NTBLKQAAAAAAYbhAQgcKBWxhYmVs"></facets-overview>';
        facets_iframe.srcdoc = facets_html;
         facets_iframe.id = "";
         setTimeout(() => {
           facets_iframe.setAttribute('height', facets_iframe.contentWindow.document.body.offsetHeight + 'px')
         }, 1500)
         </script>


# 4. Calcualte and display evaluation anomalies


```python
anomalies = tfdv.validate_statistics(statistics=test_stats,
                                     schema=schema)
tfdv.display_anomalies(anomalies)
```

    C:\Users\hasan\.conda\envs\tensorflow_validation\lib\site-packages\tensorflow_data_validation\utils\display_util.py:217: FutureWarning: Passing a negative integer is deprecated in version 1.0 and will not be supported in future version. Instead, use None to not limit the column width.
      pd.set_option('max_colwidth', -1)
    


<h4 style="color:green;">No anomalies found.</h4>


> Nice We have no anomaly found. But main point whether we can detect any anomly if there is anomaly. let's check. Now we will add some rows to check whether anomaly detection can detect it or not.


```python
def add_extra_rows(df):
    rows = [
        {
            'age': 46, 
            'fnlwgt': 257473, 
            'education': 'Bachelors', 
            'education-num': 8,
            'marital-status': 'Married-civ-spouse', 
            'occupation': 'Plumber', 
            'relationship': 'Husband', 
            'race': 'Other', 
            'sex': 'Male',
            'capital-gain': 1000, 
            'capital-loss': 0, 
            'hours-per-week': 41, 
            'native-country': 'Australia',
            'label': '>50K'
        },
        {
            'age': 0, 
            'workclass': 'Private', 
            'fnlwgt': 257473, 
            'education': 'Masters', 
            'education-num': 8,
            'marital-status': 'Married-civ-spouse', 
            'occupation': 'Adm-clerical', 
            'relationship': 'Wife', 
            'race': 'Asian', 
            'sex': 'Female',
            'capital-gain': 0, 
            'capital-loss': 0, 
            'hours-per-week': 40, 
            'native-country': 'Pakistan',
            'label': '>50K'
        },
        {
            'age': 1000, 
            'workclass': 'Private', 
            'fnlwgt': 257473, 
            'education': 'Masters', 
            'education-num': 8,
            'marital-status': 'Married-civ-spouse', 
            'occupation': 'Prof-specialty', 
            'relationship': 'Husband', 
            'race': 'Black', 
            'sex': 'Male',
            'capital-gain': 0, 
            'capital-loss': 0, 
            'hours-per-week': 20, 
            'native-country': 'Cameroon',
            'label': '<=50K'
        },
        {
            'age': 25, 
            'workclass': '?', 
            'fnlwgt': 257473, 
            'education': 'Masters', 
            'education-num': 8,
            'marital-status': 'Married-civ-spouse', 
            'occupation': 'gamer', 
            'relationship': 'Husband', 
            'race': 'Asian', 
            'sex': 'Female',
            'capital-gain': 0, 
            'capital-loss': 0, 
            'hours-per-week': 50, 
            'native-country': 'Mongolia',
            'label': '<=50K'
        }
    ]
    
    df = df.append(rows, ignore_index=True)
    
    return df
```




```python
test = add_extra_rows(test)
```


```python
test_stats = tfdv.generate_statistics_from_dataframe(test)
anomalies = tfdv.validate_statistics(statistics=test_stats, schema=schema)
tfdv.display_anomalies(anomalies)

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
      <th>Anomaly short description</th>
      <th>Anomaly long description</th>
    </tr>
    <tr>
      <th>Feature name</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>'race'</th>
      <td>Unexpected string values</td>
      <td>Examples contain values missing from the schema: Asian (&lt;1%).</td>
    </tr>
    <tr>
      <th>'native-country'</th>
      <td>Unexpected string values</td>
      <td>Examples contain values missing from the schema: Australia (&lt;1%), Cameroon (&lt;1%), Mongolia (&lt;1%), Pakistan (&lt;1%).</td>
    </tr>
    <tr>
      <th>'occupation'</th>
      <td>Unexpected string values</td>
      <td>Examples contain values missing from the schema: Plumber (&lt;1%), gamer (&lt;1%).</td>
    </tr>
    <tr>
      <th>'workclass'</th>
      <td>Multiple errors</td>
      <td>The feature has a shape, but it's not always present (if the feature is nested, then it should always be present at each nested level) or its value lengths vary. The feature was present in fewer examples than expected: minimum fraction = 1.000000, actual = 0.999847</td>
    </tr>
  </tbody>
</table>
</div>


* As anomaly is very much specific to problems, sometimes anomaly shows something important to the data. However TFDV provides some useful functionality for anomaly removal
* TFDV provides a set of utility methods and parameters that you can use for revising the inferred schema. This [reference](https://www.tensorflow.org/tfx/data_validation/anomalies) lists down the type of anomalies and the parameters that you can edit but we'll focus only on a couple here.

 You can relax the minimum fraction of values that must come from the domain of a particular feature (as described by `ENUM_TYPE_UNEXPECTED_STRING_VALUES` in the [reference](https://www.tensorflow.org/tfx/data_validation/anomalies)):

```python
tfdv.get_feature(schema, 'feature_column_name').distribution_constraints.min_domain_mass = <float: 0.0 to 1.0>
```

- You can add a new value to the domain of a particular feature:

```python
tfdv.get_domain(schema, 'feature_column_name').value.append('string')
```

## We will try here to fix anomaly in this schema



```python
# filter the age range
test = test [test['age'] > 16]
test = test[test['age'] < 91]

# drop missing values
test.dropna(inplace=True)
```


```python
test_stats = tfdv.generate_statistics_from_dataframe(test)
anomalies = tfdv.validate_statistics(statistics=test_stats, schema=schema)
tfdv.display_anomalies(anomalies)
```

    C:\Users\hasan\.conda\envs\tensorflow_validation\lib\site-packages\tensorflow_data_validation\utils\display_util.py:217: FutureWarning: Passing a negative integer is deprecated in version 1.0 and will not be supported in future version. Instead, use None to not limit the column width.
      pd.set_option('max_colwidth', -1)
    


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
      <th>Anomaly short description</th>
      <th>Anomaly long description</th>
    </tr>
    <tr>
      <th>Feature name</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>'race'</th>
      <td>Unexpected string values</td>
      <td>Examples contain values missing from the schema: Asian (&lt;1%).</td>
    </tr>
    <tr>
      <th>'native-country'</th>
      <td>Unexpected string values</td>
      <td>Examples contain values missing from the schema: Mongolia (&lt;1%).</td>
    </tr>
    <tr>
      <th>'occupation'</th>
      <td>Unexpected string values</td>
      <td>Examples contain values missing from the schema: gamer (&lt;1%).</td>
    </tr>
  </tbody>
</table>
</div>


So something is changed. We did it manually, however let see we can change it


```python
# Relax the minimum fraction of values that must come from the domain for the feature `native-country`
country_feature = tfdv.get_feature(schema, 'native-country')
country_feature.distribution_constraints.min_domain_mass = 0.9

# Relax the minimum fraction of values that must come from the domain for the feature occupation
occupation_feature = tfdv.get_feature(schema, 'occupation')
occupation_feature.distribution_constraints.min_domain_mass = 0.9
```

> If more rigid needs to be expected


```python
race_domain = tfdv.get_domain(schema, 'race')
race_domain.value.append('Asian')
```

> If some numerical values range needs to be restricted, then can be done also


```python
tfdv.set_domain(schema, 'age', schema_pb2.IntDomain(name='age', min=17, max=90))
# Display the modified schema, Notice the domain column of age
tfdv.display_schema(schema)
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
      <th>Type</th>
      <th>Presence</th>
      <th>Valency</th>
      <th>Domain</th>
    </tr>
    <tr>
      <th>Feature name</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>'age'</th>
      <td>INT</td>
      <td>required</td>
      <td></td>
      <td>min: 17; max: 90</td>
    </tr>
    <tr>
      <th>'workclass'</th>
      <td>STRING</td>
      <td>required</td>
      <td></td>
      <td>'workclass'</td>
    </tr>
    <tr>
      <th>'fnlwgt'</th>
      <td>INT</td>
      <td>required</td>
      <td></td>
      <td>-</td>
    </tr>
    <tr>
      <th>'education'</th>
      <td>STRING</td>
      <td>required</td>
      <td></td>
      <td>'education'</td>
    </tr>
    <tr>
      <th>'education-num'</th>
      <td>INT</td>
      <td>required</td>
      <td></td>
      <td>-</td>
    </tr>
    <tr>
      <th>'marital-status'</th>
      <td>STRING</td>
      <td>required</td>
      <td></td>
      <td>'marital-status'</td>
    </tr>
    <tr>
      <th>'occupation'</th>
      <td>STRING</td>
      <td>required</td>
      <td></td>
      <td>'occupation'</td>
    </tr>
    <tr>
      <th>'relationship'</th>
      <td>STRING</td>
      <td>required</td>
      <td></td>
      <td>'relationship'</td>
    </tr>
    <tr>
      <th>'race'</th>
      <td>STRING</td>
      <td>required</td>
      <td></td>
      <td>'race'</td>
    </tr>
    <tr>
      <th>'sex'</th>
      <td>STRING</td>
      <td>required</td>
      <td></td>
      <td>'sex'</td>
    </tr>
    <tr>
      <th>'capital-gain'</th>
      <td>INT</td>
      <td>required</td>
      <td></td>
      <td>-</td>
    </tr>
    <tr>
      <th>'capital-loss'</th>
      <td>INT</td>
      <td>required</td>
      <td></td>
      <td>-</td>
    </tr>
    <tr>
      <th>'hours-per-week'</th>
      <td>INT</td>
      <td>required</td>
      <td></td>
      <td>-</td>
    </tr>
    <tr>
      <th>'native-country'</th>
      <td>STRING</td>
      <td>required</td>
      <td></td>
      <td>'native-country'</td>
    </tr>
    <tr>
      <th>'label'</th>
      <td>STRING</td>
      <td>required</td>
      <td></td>
      <td>'label'</td>
    </tr>
  </tbody>
</table>
</div>


    C:\Users\hasan\.conda\envs\tensorflow_validation\lib\site-packages\tensorflow_data_validation\utils\display_util.py:180: FutureWarning: Passing a negative integer is deprecated in version 1.0 and will not be supported in future version. Instead, use None to not limit the column width.
      pd.set_option('max_colwidth', -1)
    


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
      <th>Values</th>
    </tr>
    <tr>
      <th>Domain</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>'workclass'</th>
      <td>'?', 'Federal-gov', 'Local-gov', 'Never-worked', 'Private', 'Self-emp-inc', 'Self-emp-not-inc', 'State-gov', 'Without-pay'</td>
    </tr>
    <tr>
      <th>'education'</th>
      <td>'10th', '11th', '12th', '1st-4th', '5th-6th', '7th-8th', '9th', 'Assoc-acdm', 'Assoc-voc', 'Bachelors', 'Doctorate', 'HS-grad', 'Masters', 'Preschool', 'Prof-school', 'Some-college'</td>
    </tr>
    <tr>
      <th>'marital-status'</th>
      <td>'Divorced', 'Married-AF-spouse', 'Married-civ-spouse', 'Married-spouse-absent', 'Never-married', 'Separated', 'Widowed'</td>
    </tr>
    <tr>
      <th>'occupation'</th>
      <td>'?', 'Adm-clerical', 'Armed-Forces', 'Craft-repair', 'Exec-managerial', 'Farming-fishing', 'Handlers-cleaners', 'Machine-op-inspct', 'Other-service', 'Priv-house-serv', 'Prof-specialty', 'Protective-serv', 'Sales', 'Tech-support', 'Transport-moving'</td>
    </tr>
    <tr>
      <th>'relationship'</th>
      <td>'Husband', 'Not-in-family', 'Other-relative', 'Own-child', 'Unmarried', 'Wife'</td>
    </tr>
    <tr>
      <th>'race'</th>
      <td>'Amer-Indian-Eskimo', 'Asian-Pac-Islander', 'Black', 'Other', 'White', 'Asian'</td>
    </tr>
    <tr>
      <th>'sex'</th>
      <td>'Female', 'Male'</td>
    </tr>
    <tr>
      <th>'native-country'</th>
      <td>'?', 'Cambodia', 'Canada', 'China', 'Columbia', 'Cuba', 'Dominican-Republic', 'Ecuador', 'El-Salvador', 'England', 'France', 'Germany', 'Greece', 'Guatemala', 'Haiti', 'Holand-Netherlands', 'Honduras', 'Hong', 'Hungary', 'India', 'Iran', 'Ireland', 'Italy', 'Jamaica', 'Japan', 'Laos', 'Mexico', 'Nicaragua', 'Outlying-US(Guam-USVI-etc)', 'Peru', 'Philippines', 'Poland', 'Portugal', 'Puerto-Rico', 'Scotland', 'South', 'Taiwan', 'Thailand', 'Trinadad&amp;Tobago', 'United-States', 'Vietnam', 'Yugoslavia'</td>
    </tr>
    <tr>
      <th>'label'</th>
      <td>'&lt;=50K', '&gt;50K'</td>
    </tr>
  </tbody>
</table>
</div>



```python
updated_anomalies = tfdv.validate_statistics(test_stats, schema)
tfdv.display_anomalies(updated_anomalies)
```

    C:\Users\hasan\.conda\envs\tensorflow_validation\lib\site-packages\tensorflow_data_validation\utils\display_util.py:217: FutureWarning: Passing a negative integer is deprecated in version 1.0 and will not be supported in future version. Instead, use None to not limit the column width.
      pd.set_option('max_colwidth', -1)
    


<h4 style="color:green;">No anomalies found.</h4>


> We can actually go further can remove all anomalies, but we will not do it further. Let's consider now last part. 

# 7. Examining dataset slices

``get_feature_value_slicer`` from ``slicing_util``, Normally dictionary features argument. If we want full domain, then we can just tell ``None``, means if we use None, then both male and female in this case, if we consider the ``sex`` feature


```python
from tensorflow_data_validation.utils import slicing_util
slice_fn = slicing_util.get_feature_value_slicer(features={'sex':None})
```

> Now we can see the statistics


```python
slice_stats_options = tfdv.StatsOptions(schema=schema,
                                        slice_functions=[slice_fn],
                                        infer_type_from_schema=True)
```


```python
# Convert dataframe to CSV since `slice_functions` works only with `tfdv.generate_statistics_from_csv`
CSV_PATH = 'slice_sample.csv'
train.to_csv(CSV_PATH)

# Calculate statistics for the sliced dataset
sliced_stats = tfdv.generate_statistics_from_csv(CSV_PATH, stats_options=slice_stats_options)
```

    WARNING:apache_beam.runners.interactive.interactive_environment:Dependencies required for Interactive Beam PCollection visualization are not available, please use: `pip install apache-beam[interactive]` to install necessary dependencies to enable all data visualization features.
    



    WARNING:root:Make sure that locally built Python SDK docker image has Python 3.7 interpreter.
    WARNING:apache_beam.io.tfrecordio:Couldn't find python-snappy so the implementation of _TFRecordUtil._masked_crc32c is not as fast as it could be.
    

    WARNING:tensorflow:From C:\Users\hasan\.conda\envs\tensorflow_validation\lib\site-packages\tensorflow_data_validation\utils\stats_util.py:247: tf_record_iterator (from tensorflow.python.lib.io.tf_record) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use eager execution and: 
    `tf.data.TFRecordDataset(path)`
    

    WARNING:tensorflow:From C:\Users\hasan\.conda\envs\tensorflow_validation\lib\site-packages\tensorflow_data_validation\utils\stats_util.py:247: tf_record_iterator (from tensorflow.python.lib.io.tf_record) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use eager execution and: 
    `tf.data.TFRecordDataset(path)`
    


```python
print(f'Datasets generated: {[sliced.name for sliced in sliced_stats.datasets]}')
```

    Datasets generated: ['All Examples', 'sex_Male', 'sex_Female']
    


```python


print(f'Type of sliced_stats elements: {type(sliced_stats.datasets[0])}')
```

    Type of sliced_stats elements: <class 'tensorflow_metadata.proto.v0.statistics_pb2.DatasetFeatureStatistics'>
    


```python
from tensorflow_metadata.proto.v0.statistics_pb2 import DatasetFeatureStatisticsList

# Convert `Male` statistics (index=1) to the correct type and get the dataset name
male_stats_list = DatasetFeatureStatisticsList()
male_stats_list.datasets.extend([sliced_stats.datasets[1]])
male_stats_name = sliced_stats.datasets[1].name

# Convert `Female` statistics (index=2) to the correct type and get the dataset name
female_stats_list = DatasetFeatureStatisticsList()
female_stats_list.datasets.extend([sliced_stats.datasets[2]])
female_stats_name = sliced_stats.datasets[2].name

# Visualize the two slices side by side
tfdv.visualize_statistics(
    lhs_statistics=male_stats_list,
    rhs_statistics=female_stats_list,
    lhs_name=male_stats_name,
    rhs_name=female_stats_name
)
```


<iframe id='facets-iframe' width="100%" height="500px"></iframe>
        <script>
        facets_iframe = document.getElementById('facets-iframe');
        facets_html = '<script src="https://cdnjs.cloudflare.com/ajax/libs/webcomponentsjs/1.3.3/webcomponents-lite.js"><\/script><link rel="import" href="https://raw.githubusercontent.com/PAIR-code/facets/master/facets-dist/facets-jupyter.html"><facets-overview proto-input="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"></facets-overview>';
        facets_iframe.srcdoc = facets_html;
         facets_iframe.id = "";
         setTimeout(() => {
           facets_iframe.setAttribute('height', facets_iframe.contentWindow.document.body.offsetHeight + 'px')
         }, 1500)
         </script>



* If we want to be more specific, then we can map the specific value to the feature name. For example, if we want just `Male`, then we  can declare it as `features={'sex': [b'Male']}`. Notice that the string literal needs to be passed in as bytes with the `b'` prefix.

* We can also pass in several features if we want. For example, if we want to slice through both the `sex` and `race` features, then we can do `features={'sex': None, 'race': None}`.


