# EDA on Text data

* This is a part of series. [First](https://Hasangoni.github.io/2022/11/19/Explanatory_data_analysis), [Second](https://Hasangoni.github.io/2022/11/19/Feature_preprocessing_and_generation.html), [Third](https://Hasangoni.github.io/2022/11/19/Exploring_anonymized_data.html) parts are connected in link.   In this part I will try to write down about text data
# Feature extraction from text

1. Bag of words

2. Embeddings(~word2vec)

## 1. Bag of words

1.1. CountVectorizer

* each word is separated and count number of occurences

```python
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)
print(vectorizer.get_feature_names())
print(X.toarray())
```

* We may be need to do some post processing. As we know KNN, neural networks are sensitive to the scale of the features. So we need to scale the features. We can use TF-IDF to do this.

1.2. TfidfVectorizer

* What actually is just not frequency but normalized frequency.
* Term frequency:

```python
tf = 1/ x.sum[axis=1](:,None)
x = x * tf
```

* Inverse document frequency:

```python
idf = np.log(x.shape[0]/(x>0).sum(axis=0)))
x = x*idf
sklearn.feature_extraction.text.TfidfVectorizer
```

1.3 N-grams

* Not only words but n-consequent words

```python
sklearn.feature_extraction.text.CountVectorizer(ngram_range=(1,2)) 
# may be parameter analyzer
```

### Text Preprocessing

* Actually before applying any Bag of words we need to preprocess the text. We need to remove the stop words, stemming, lemmatization, etc.* Conventionally preprocessing are

  * Tokenization -> Very very sunny day -> [Very, very, sunny, day]
  * Lowercasing -> [very, very, sunny, day] -> [very, very, sunny, day] ->CountVectorizer from sklearn will automatically do this
  * Removing punctuation
  * Removing stopwords -> [The cow jumped over the moon] -> [cow, jumped, moon]
    * Ariticles or preprositon words
    * Very common words
    * Can be used NLTK library
    * sklearn.feature_extraction.text.CountVectorizer(max_df)
    * max_df is the frequency threshold, after which the word is removed

  * Stemming/Lemmatization

  * Stemming
    * [democracy, democratic, democratization] -> [democr]
    * [Saw] -> [s]
  * Lemitization
    * [democracy, democratic, democratization] -> [democracy]
    * [Saw, sawing, sawed] -> [see or saw] depending on text

## Summray of Bag of words Pipeline

1. Preprocessing
   Lowercasing, removing punctuation, removing stopwords, stemming/lemmatization
2. N-grams helps to get local context
3. Post processing TF-IDF

## 2. Embeddings

### Word2vec

* Vector representation of words and text
* Each word is represented as a vector, in some sophisticated way, which could have 100 dimensions or more.
* Same words will have similar vectors. king->queen
* Also addition and subtraction of vectors will have some meaning. -> king + woman - man = queen
* Several implementaton of word2vec
  * Word2vec
  * Glove
  * FastText
* Sentences
  * Doc2vec
* Based on situation we can use word or sentence embeddings. Actually try both and take the best one.
* All the preprocessing steps can be applied to the text before applying word2vec.

### Comparion Bag of words and Word2vec

* Bag of words
  * Very large vector
  * meaning is easy value in vector is known
* Word2vec
  * Relative Small vector
  * Values of vector can be interpreted only some cases
  * The words with simlar meaning will have similar embeddings

__Next post can be found [here](https://Hasangoni.github.io/2022/11/19/Feature_extraction_images.html)__