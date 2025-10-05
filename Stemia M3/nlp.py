import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from nltk.corpus import stopwords
from string import punctuation 
from nltk.stem import LancasterStemmer, SnowballStemmer, PorterStemmer,WordNetLemmatizer

class Preprocess:
    def __init__(self,method='WordNetLemmatizer')->None:
        """
        Preprocesses the inputs in your dataset to be suitable for machine learning models 
        Preprocessing involves the following steps:
        1) Convert all words to lower case for uniformity (normalization)
        2) Split each word in the sentence and remove stop words and punctuation marks (tokenization)
        3) Stem each word (stemming)
        4) Uses tf-idf vectorization technique to preserve the semantic meaning of the words in the sentence
        5) Label encode the categories 
        
        Attributes:
        self.method(str) - The method by which you would want to stem
        self.methods(list) - The possible methods which you can use for stemmming
        self.stemmers(dict) - The appropriate nltk stemmers matching to the methods
        self.stemmer(nltk.stem.stemmer) - The stemmer object which will be used to stem the words
        self.stuff_to_be_removed(list) - The list of characters which should be removed 
        self.isFitted(bool) - Indicates if the label encoders and tf-idf vectorizers are fitted
        
        Params:
        method(str) - the method by which you would like to stem your inputs
        Returns:
        None 
        """
        self.method = method 
        self.methods = ['LancesterStemmer','PorterStemmer','SnowballStemmer','WordNetLemmatizer']
        if method not in self.methods:
            raise ValueError(f'The method should be from the following methods {self.methods}')
        self.stuff_to_be_removed = list(stopwords.words('english'))+list(punctuation)
        self.stemmers = {
            'PorterStemmer':PorterStemmer(),
            'LancesterStemmer':LancasterStemmer(),
            'SnowballStemmer':SnowballStemmer(language='english'),
            'WordNetLemmatizer':WordNetLemmatizer()
        }
        self.stemmer = self.stemmers[self.method]
        self.isFitted = False

    def preprocess(self,message:str)->str:
        """
        Stems and removes stopwords and punctuation from the given message 
        Params:
        message(str) - The message which you want to preprocess
        Returns:
        str - The preprocessed message        
        """
        # Convert message to lower case 
        message = message.lower()
        # Remove all the links from the messages 
        message = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|'\
                       '(?:%[0-9a-fA-F][0-9a-fA-F]))+','', message)
        # Remove all the mentions
        message =re.sub("(@[A-Za-z0-9_]+)","", message)

        # Remove stopwords and perform stemming
        if self.method == 'WordNetLemmatizer':
            message = ' '.join([self.stemmer.lemmatize(word) for word in message.split() if word not in self.stuff_to_be_removed])
        else:
            message = ' '.join([self.stemmer.stem(word) for word in message.split() if word not in self.stuff_to_be_removed])
        # Return the message
        return message

    def fit(self,X:pd.Series,y:pd.Series)->None:
        """
        Fits the tf-idf vectorizer and label encoders according the training data 
        Params:
        X(pd.Series) - The column of the dataframe containing the message contents
        y(pd.Series) - The column of the dataframe containing the labels
        Returns:
        None
        """
        # Preprocess the message first
        X = X.apply(lambda x: self.preprocess(x))
        # Initialize the label encoder and tfidf vectorizer 
        self.labelEncoder = LabelEncoder()
        self.vectorizer = TfidfVectorizer()
        # Fit the label encoder and vectorizer 
        self.vectorizer.fit(X)
        self.labelEncoder.fit(y)
        self.isFitted=True

    def transform(self,X:pd.Series,y:pd.Series)->tuple:
        """
        Transforms the given messages and labels to be suitable for machine learning models.
        Removes stop words and punctuation from the message and then stems it and applies 
        tf-idf vectorization to the message.
        Label encodes the message category
        Params:
        X(pd.Series or str) - The message or column of messages you want to transform
        y(pd.Series or list or np.array) - The labels which you want to transform 
        Returns:
        tuple containing the transformed messages and labels        
        """
        # Check if it is fitted
        if not self.isFitted:
            raise NotImplementedError('Please use fit function first')
        # Preprocess the messages first and apply tfidf vectorization
        if isinstance(X,pd.Series):
            X = X.apply(lambda x: self.preprocess(x))
            vector = self.vectorizer.transform(X)
        else:
            X = self.preprocess(X)
            vector = self.vectorizer.transform([X])
        # convert tfidf sparse matrix to an array
        vector = vector.toarray()
        # Apply label encoding 
        if y is not None:
            labels = self.labelEncoder.transform(y)
            return vector,labels
        else:
            return vector

    def fit_transform(self,X:pd.Series,y:pd.Series)->tuple:
        """
        Fits and transforms the data to be suitable for machine learning models
        Params:
        X(pd.Series or str) - The message or column of messages you want to transform
        y(pd.Series or list or np.array) - The labels which you want to transform 
        Returns:
        tuple containing the transformed messages and labels   
        """
        # Call the fit function first
        self.fit(X,y)
        # Call the transform function 
        vectors,labels = self.transform(X,y)
        return vectors,labels
