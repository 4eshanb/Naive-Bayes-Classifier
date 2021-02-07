# Naive Bayes Calssifier for Movie Reviews

### Text Classification
Text classification requires assigning a category from a predefined set to each document of interest. In the case,
the reviews are modeled as a feature vector. These feature vector representations are used to make predictions about unlabeled (unknown rating) reviews.
To do this, a Naive Bayes classifier is built and trained.

### Data  
There is a total of 500 movie reviews: 250 that have been given a positive rating (an overall score
\> 5 out of 10) and 250 that have been given a negative rating (an overall score < 5). 
These reviews havebeen split into training, development, and testing files. 

Each dataset contains a list of reviews. Each review has two facets, Overall, which specifies whether the
review is positive or negative, based on the score from 1-10 and Text, which gives the textual review. Each
review is separated by a period character (.)

### Features
For each review, a feature vector is created as a representation of the text. Each element of this vector represents
a key/value pair. The key is the name of the feature. The value can be any
real number. Binary features are a common special case of features when the values are restricted to the set
f0,1g. These are useful for indicating whether a feature is present or not in the document. For example, we
could model the presence or absence of the word throw with a feature named UNI throw and a value of 1 to
indicate we’ve seen it and a 0 to indicate we have not.

In some cases, we can also use the value to give an estimate on the strength of the feature. 
For example, we might want to take into account that a word was seen more than once in a document. Although any
value is possible, most text classifiers work best when the values are between ±1 or between 0 and 1. To
ensure this we will normalize our values to be within this range.

In NLTK, feature vectors are represented using the built-in dict class, which is Python’s implementation
of a hash table.
In this part of the assignment, create four different feature sets of the text:
>1. word_features: unigram, bigram and trigram word features

>2. word_pos features: unigram, bigram and trigram word features and unigram, bigram and trigram
part-of-speech features

>3. word_pos_liwc_features: unigram, bigram and trigram word features and unigram, bigram and trigram
part-of-speech features and the liwc category features.

>4. word_pos_opinion_features: unigram, bigram and trigram word features and unigram, bigram and
trigram part-of-speech features and the opinion lexicon binary features.

The value of the feature should be the relative frequency of that feature within the document,
except for LIWC and Opinion Lexicon features. 
For example, if the word the occurred 10 times and there were 200 words overall in the document, then the feature value for UNI the should be 10/200 = 0:05. 
If the bigram feature BIGRAM the food was seen once out of 60 bigrams in the document, the value should be 1/60 = 0:0167.


For each feature set and dataset, the labels and feature vectors are written to a file named
FEATURE SET-DATASET-features.txt, where each line represents a document. FEATURE SET
is one of the following: word features, word pos features, word pos liwc features,
word pos opinion features. 
DATASET is one of the following: training, development, testing.

The first token on the line is the class label (positive or negative for the imdb reviews). 
Each additional token should is a feature name:value pair separated by whitespace.

Look for these files in the features folder.

# LIWC and Opinion Lexicon
LIWC categories are used as another feature. A Python module that will take a tokenized input string and return a Counter
object with various counts and features of the input text. 
Features used:
> Negative Emotion
  Positive Emotion
  Negations
  Sadness
  Positive feelings
  Assent
  Leisure

The Opinion Lexicon is a set of positive words and a set of negative words For the features using this
lexicon each word in both the positive and negative list of words will be a binary feature. So if the word
appears in the review it will have a value of 1 and if it does not it will have a value of 0. 

http://www.liwc.net/
https://docs.python.org/2/library/collections.html
http://www.liwc.net/LIWC2007LanguageManual.pdf
https://www.kaggle.com/nltkdata/opinion-lexicon
