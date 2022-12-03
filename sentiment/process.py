from keras_preprocessing.sequence import pad_sequences
import pandas as pd
import keras
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer as Tokenizer2
import pickle
from nepalitokenizer import NepaliTokenizer
from nepali_stemmer.stemmer import NepStemmer

# loading the sentimental classification model
reconstructed_model = keras.models.load_model("NLP_Trained_models/sentimental_classification/sentimental_v2.h5")

# loading the tokenizer
with open('NLP_Trained_models/sentimental_classification/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# maximum padding length for the padded sequnces
maximum_padding_length = 10

def text_preprocessing(text):
    """
        Preprocessing of the text
        
        Arguments:
            text : nepali sentences
        
        Output:
            text : preprocessed nepali sentences
    """
    nepali_stopwords = stopwords.words('nepali')
    nepstem = NepStemmer()
    tokenize = NepaliTokenizer()
    # Tokenize the reviews
    text = tokenize.tokenizer(text)
    # Remove the nepali stopwords
    text = [word for word in text if word not in nepali_stopwords]
    text = ' '.join(text)
    # Stemming the nepali words
    text = nepstem.stem(text)
    # Remove the leading and trailing spaces
    text = text.split()
    text = ' '.join(text)
  
    return text



def testOwnString(own_string):
    """
        Sentimental Classification of the custom string
        
        Argument:
            own_string : nepali sentence entered by the user

        Output:
            own_pred : 1-d array of length 3 [negative, positive, neutral] of the assosciated probabililties 
    """
    # preprocessing of the text
    own_string = text_preprocessing(own_string)
    own_string_list = [own_string]
    # Conversion of the string into the one hot vectors and padding
    own_string_series = pd.Series(own_string_list)
    own_string_sequences = tokenizer.texts_to_sequences(own_string_series)
    own_string_padded = pad_sequences(own_string_sequences, maxlen = maximum_padding_length, padding = "post", truncating = "post")
    # sentimental prediction of the sentences using the model
    own_pred = reconstructed_model.predict(own_string_padded)
    return own_pred