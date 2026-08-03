# Feature Preprocessing and Generation

* This is a second part of the class note of how to proceed to a data science project. First part can be found [here](https://Hasangoni.github.io/2022/11/19/Explanatory_data_analysis.html). In this part I will try to write down, how prprocessing steps vary based on model selection. What type of preprocessing needs to be done on different types of data and how to handle missing values in a project.

# Employed model impacts choice of preprocessing

* Normally if we have a model non linear dependency between features and target, we should use one type of preprocessing
  * for example one hot encoding
  * however random forest is not sensitive to one hot encoding and doesnot require it

* Feature generation sometimes gives you a peak in performance
  * for example if you have a feature which is a combination of two features, it can give you a peak in performance
  * however it is not always the case, sometimes it can give you a dip in performance
  * so it is always good to try out feature generation and see if it gives you a peak in performance

* Depends on the model you are using and the type of data you have you need to choose the type of preprocessing(feature preprocessing and feature generation)
  * for example if you are using a linear model, you should use one hot encoding
  * if you are using a tree based model, you should not use one hot encoding

# Feature preporocessings and feature generation

## Numeric features

### Prprocessing

* Model Types

  * Tree based models
    * Normally if features are multiplied with some numbers, then model performance doesnot change, thats Noralization doesnot necessary for tree based models

  * Non-tree based models
    * But in case of non tree based model scaling and normalization can affect the model
    * Some non tree based models are
      * k nearest neighbors
      * linear models
      * nueral networks

* Preprocessing: Scaling

  * To[0, 1] sklearn.preprocessing.MinMaxScaler
    $$ X = (X - X.min()) / (X.max() - X.min()) $$

  * To [mean=0 std=1] sklearn.preprocessing.StandardScaler
    $$ X = (X - X.mean()) / X.std() $$
  
* Preprocessing: Outliers

  * Can be found both in features and outputs
  * If outliers are present in features, then we can use robust scaler
  * We can also use lower bound and upper bound between two values, may be 0.01 and 0.99 percentile, well known in finance called __winsorization__

    * ```python
        UPPER_BOUND, LOWER_BOUND = np.percentile(X, [1, 99])
        y = np.clip(y, UPPER_BOUND, LOWER_BOUND)
        pd.Series(y).hist(bins=30)
        ```

  * Another way to preprocess is rank transformation, which actually sets spaces between proper assorted values to to equal, it could be a better option from MInMaxScaler, if we have outliers in features, because rank transformation will move the outlier to closer to object
    * Linear models, neural networks, k nearest neighbors can benefits from this, if we have no times to handle outliers manually

      ```python
         scipy.stats.rankdata
      ```

    * One important things about rank transformaiton, when applying to test data, you need to store the created mapping from feature values to rank values. Or alternatively you can concatenate train and test data and apply rank transformation to the whole dataset, and then split it back to train and test

* Preprocessing: Transformation

  * There is one more preprocessing which often helps to non tree based models, specially neural networks.
    * It is called log transformation

        ```python
          np.log(1 + X)
        ```

    * Or Raising to power <1:

        ```python
          np.sqrt(X, 2/3) 
        ```

    * Both this transformations could be helpful, because they drive too big values closer to features average value, so values near 0 a bit distuinguishable. Despite simplicity, this transformation could be helpful in many cases for neural network. __Also data seems to be normllay distributed after this transformation (if not try to use other transformation and see histogram[Andrew ng])__

  * __Sometimes it is benefecial to apply concatenated dataframes produced be different preprocessing, or to mix models training different pre-porcessed data. Specially Linear models, KNN and Neural networks can benefit from this__

### Feature generation

* Ways to proceed :
  * prior knowledge
  * EDA(exploratory data analysis)

  * Example:

    * We have Real state price and real state squared data, from this we can create a new feature which is price per square meter
    * Forest cover type prediction dataset, if we have horizontal and vertical distance from hydrology, we can create a new feature which is distance from hydrology, $$ combined = \sqrt{horizontal\_distance\_to\_hydrology^2 + vertical\_distance\_to\_hydrology^2} $$

    This type of featur generation is not only helpful for neural network and linear models, but also for tree based models, sometimes we can have a peak in performance with less amount of trees with thiess type of feature generation

    * Another example is, if we have a price of a product, we can take fraction of the price. for example if the price is 9.99 the fractionaly part of the price is 0.99. This type of feature of features also can distinguish between robot and human. Normally in finance, this is very helpful. As a human we normally use whole number.
  
  * This is just some example, one can be very creative and can get a peak in model performance with feature generation

## Categorical and Ordinal features

### Preprocessing

* In Titanice dataset, caegorical features are Sex, Cabin and Embarked and ordinal features are Pclass, which stands for passenger class. We have a value [1,2,3] for PClas, which is ordered based on the price of the ticket, so it is ordinal feature.
* One needs to be very careful here, because if we put PClass as a numeric features, we say the difference between class 1 and class 2 is the same as the difference between class 2 and class 3, which is not true.
* Other ordinal features are
  * Driver's license: A, B, C, D, E, F, G
  * Education: Elementary, Middle, High, College, Master, PhD
* For Ordinal features one can use LabelEncoder, which will assign a number to each category, but it will not be ordered, so we need to use OrdinalEncoder, In case of tree based models, such model helps, but in case of non tree based models, you need to use the preprocess the data in a different way.
* For Titanic dataset again, let's see the Embarked feature, which has category S, C, Q.
  
  * Alphabetical :
    [S, C, Q] -> [2, 1, 3]

    ```python
      from sklearn.preprocessing import LabelEncoder
      le = LabelEncoder()
      le.fit_transform(df['Embarked'])
    ```

  * Order of apperance:

    [S, C, Q] -> [1, 2, 3] S comes first to the data and C comes after that

    ```python
    pandas.factorize
    ```

  * Frequency Encodeing:
    [S, C, Q] -> [0.5, 0.3, 0.2] S is the most frequent category and Q is the least frequent category

    ```python
    encoding = df.groupby('Embarked').size() / len(df)
    df['enc'] = df['Embarked'].map(encoding)
    ```

    * One important thing regarding frequency encoding, if we have multiple features with same frequency, then the features are not distinguishable, so we need to rank transformation

    ```python
    encoding = df.groupby('Embarked').size() / len(df)
    df['enc'] = df['Embarked'].map(encoding)
    from scipy.stats import rankdata
    ```

  * There are different encodings possible, one can be very creative here, one need to try different encodings here.

  * Categorical features for non-tree based models

    * One hot encoding

    ```python
    pd.get_dummies, sklearn.preprocessing.OneHotEncoder
    ```

    * Tree methods will slow down here, because they will have to split on each category, so one hot encoding is not good for tree based models
    * If too many unique categories, then one hot encoding will create too many features, To work with them we need to know how to handle sparse data.XGBoost, LightGBM, CatBoost can handle sparse data, but other models cannot handle sparse data, so we need to use some other methods to handle sparse data.
  
### Feature generation

  * Feature interaction between several categorica features -> normally usefull linear model and KNN, e.g. Titanic dataset, sex and Pclass. We can concatenate both features and then one hot encode them. Advanced feature generation will be discussed in the next sections.

## Datetime and coordinates

### Feature generation:Datetime

* Most of the time, we have a datetime feature, which is a string, we need to convert it to a datetime object, so we can extract different features from it, such as year, month, day, hour, minute, second, day of week, day of year, week of year, quarter, etc.
* Normally it falls into three categories

  * Peridocity: year, month, day, hour, minute, second, day of week, day of year, week of year, quarter, etc.

    * helpful to capture repetative patterns in the data

  * Time Since:
    * Row independent moment, e.g. time since last purchase, time since last login, time since last visit, day since lasth holiday, day to next holiday, etc.
    * Row dependent moment, e.g. since 00:00:00 UTC, 1 January 1970, etc.
  
  * Difference between dates:

    * datetime_feature_1 - datretime_feature_2. e.g. churn prediciton. The likelihood, the customers will churn. feature: last_purchase_date, last_call_date, date_diff = last_purchase_date - last_call_date

  * After generating features in datetime, we will get numeric or categorical features, so we need to apply the same preprocessing as we did for numeric and categorical features.

### Feature generation: Coordinates

* Distance:
  * Real state price prediction. We can create new features to calculate distance from the city center, distance from the airport, distance from the sea, etc. If such data is not available we can extract some features from the maps. We can convert the map into differnt squares and extract the most expensive flats in this square and then calculate the distance from the center of the square. We can also use the same method to calculate the distance from the airport, sea, etc.
    * Create cluster from datapoints and calculate the center of the cluster and then calculate the distance from the center of the cluster.
    * Find some very special areas and add distance from these areas.

* Aggreagated statistics, objects sorrounding area:
  * Calcualte number of important place in sorrounding area, which can be interpreted as the quality of the area.
  * Or mean price of the sorrounding area. How expensive sorrounding area is.

* In case of decision trees, slightyl rotated coordinates can be helpful, because decision trees are not able to capture the rotation of the data.

## Missing Values

* Examples of missing values

  * NA Values
  * Empty string
  * -1
  * Very large number
  * -99999 (or less)
  * 999
  * 99

* how to know, whether this is a missing value or not. plot histogram, and see some values are very different from other values, So missing values can be hidden means may be used other values in plac of missing values.

* Fillna approaches:

1. -999, -1 etc -> for Neural networks it will be difficult
2. mean, median, mode -> for trees it will be harder to find differnt splits
3. Reconsturct value

* Feature generation:

* Need to create a new column named as is_missing, which will be 1 if the value is missing and 0 if the value is not missing. This will help the model to capture the missing values. -> help both tree an neural networks but only problem is number of columns will increated

* Carful during missing value imputation, e.g. we have two features, one is datetime and temperature of whole year. If we impute the missing values with mean, then the mean will be calculated from the whole year, which is not correct, because the temperature is not same in the whole year. So we need to calculate the mean for each month and then impute the missing values with the mean of the month. Then we creeate a new feature, which will be the difference between the mean of the month and the actual value. In case of imputation with mean, it will be 0, as we already impute it with mean, which is 0 , so new feature will not create some sense, or in other way new feature will not help the model. So we need to be careful during imputation. __Unfortunately we don't have enough time to be careful here__.

* For example we have a categorical featue, we have created a numeric feature from this categorical feature. So A->1, B->9, B->?, if we impute missing value with some outer range value like -999 in missing value. Then the category will be going to closer to -999 and the more missing value and category will go more closer to this big or small number.  In case of mean and median imputation it will also a problem. So we need to be careful during imputation. __Simple solution will be ignore missing value, while calculating the mean and median__.

* XGBoost, LightGBM, CatBoost can handle missing values.
* Sometimes outliers can be handled as missing values.
* Treat missing values which appeear in test but not in trainig data.

  * Create a category which will tell how frequently occures this category. If there is some correlation between occurence frequeny and target, then it will sucessly tell the prediction.

### Summray missing values

* The choice of filling depends on situation
* Usual way -999, mean, median, mode but careful
* Missing values sometimes hidden.
* Binary feature is_missing is helpful
* avoid missing values during feature generation
* XGBoost can handle missing values


 __Next post can be found [here](https://Hasangoni.github.io/2022/11/19/Exploring_anonymized_data.html)__