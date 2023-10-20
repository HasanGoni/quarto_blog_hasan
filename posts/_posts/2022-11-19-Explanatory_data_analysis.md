# Explanatory Data Analysis

* This is a summary of the course [How to win a data science competition: Learn from top Kagglers](https://www.coursera.org/learn/competitive-data-science). The course is available on coursera. The course is taught by Yandex and MIPT. This is my note taken from that course.

* As the summary seems a little bit long. So I tried to convert it different parts. This is introduction part. In this part I will try to explain what is EDA and how should one proceed with EDA in a project.

  * [Feature preprocessing and generation](https://Hasangoni.github.io/2022/11/19/Feature_preprocessing_and_generation.html)
  * [Exploring anonymized data](https://Hasangoni.github.io/2022/11/19/Exploring_anonymized_data.html)
  * [Feature extraction from text](https://Hasangoni.github.io/2022/11/19/Feature_extraction_from_text.html)
  * [Feature extraction from images](https://Hasangoni.github.io/2022/11/19/Feature_extraction_images.html)

# Explantory data analysis

1. What are the most important things we need to pay attention to when we are doing exploratory data analysis?
2. This knowledge help to create a good model

## What is EDA

* Understand the data
* Build intuition about the data
* Generatge hypothesis of possible new features
* Find some insights about the data

## Visualization

* One EDA tool is visualization -> see the patterns-> what are these patterns -> why we see them
* e.g. There was a competition in kaggle, where you just don't need anaything, if you just understand it very well
  * Goal of the competition: whether a person will use the promo, or not
  * Feature of the person are the inputs, e.g. age, sex, marital status, etc. and there are features which described the promo.
  * But two intersting features were

    * Number of promos sent by the person before
    * NUmber of promos the person has used before
  * If we sort the rows based on number of promos sent by the person before and we calcualte the difference between number of promos used by the person. Most of the time this difference is equal to target values.
  * Although it is not given what to do with the Nan values, but we can see that without doing modeling we can get 80% accuracy. It is only possible after doing EDA.
  * Actually this is very good example of data leakage, done by the person who created the dataset.

## Building intuition


1. Getting domain knowledge 
2. Checking the data is intuitive
3. Understand how data is generated

### 1. Getting domain knowledge

* What is our goal?
* What data we have?
* how people actually tackle this kind of problem to build this problem to build baseline?


* So may be first goal is to read wikipedia, do google search, read some blogs, etc. to get some domain knowledge.
* e.g. We need to predict advisers's cost

  * First to understand this is about web adverstisemnt. 
  * By seeing column names we can understand, that data is exported from Google AdWords system. 
  * After reading several articles we get the meaning of the columns.

### 2. Checking the data is intuitive
 
* After getting the domain knowledge, we can check the data is intuitive or not. Whether it agrees with our domain knowledge or not.
  * e.g. if this is a age data, we can be sure that the age is between 0 and 100. If it is not, then we need to check it. If we get a value 336, may be it's a typo and it should be 33.6. If it is not, then we need to check it. 
  * The value of Clicks columns should be less or equal to Impressions. If it is not, then we need to check it. We need to check whethe it is a error or some kind of logic there. 
  * > **Note**: Fastai founder Jeremy says there is outlier in statistics, but in case of data science, if there is some anomaly, we need to check it and go deep down to understand it.
  * We also createa a new feature with incorrect, whether 0 or 1, based on the above conditions.

### 3. Understand how data is generated

* Algorithm for sampling object from database.
* Random sample, or over sampling, or under sampling
* Validation scheme can only ge genreated if we understand how data is generated 
* If theere is a diffrence between train and test, then we need to check it, try to understand why it is done and whether to gain any intuition about it. 

__Next post can be found [here](https://Hasangoni.github.io/2022/11/19/Feature_preprocessing_and_generation.html)__
