from keras_preprocessing.sequence import pad_sequences
import pandas as pd
import keras
from sklearn.model_selection import train_test_split
from Nepali_nlp import Tokenizer, Stem
from nltk.corpus import stopwords
import sklearn
from keras.preprocessing.text import Tokenizer as Tokenizer2
import pickle

reconstructed_model = keras.models.load_model(
    "Nepali-Language-Processing\\Classification\\Nepali Sentimental Classification\\sentimental_v1.h5")
#    "../Nepali-Language-Processing/Classification/Nepali Sentimental Classification/sentimental_v1.h5")

maximum_padding_length = 10


def text_preprocessing(text):
    nepali_stopwords = stopwords.words('nepali')
    # Tokenize the reviews
    text = Tokenizer().word_tokenize(text)
    # Remove the nepali stopwords
    text = [word for word in text if word not in nepali_stopwords]
    # Stemming the nepali words
    text = Stem().rootify(text)
    text = ' '.join(text)
    # Remove the leading and trailing spaces
    text = text.split()
    text = ' '.join(text)

    return text


def create_tokenizer():
    data = pd.read_csv(
        "../Nepali-Language-Processing/Classification/Nepali Sentimental Classification/sentimental.csv")
    unique_words_count = 3179

    tokenizer = Tokenizer2(num_words=unique_words_count)

    data.rename(columns={"Data ": "Data"}, inplace=True)
    update_list = [(475, 1), (796, 0), (1083, 0), (1441, 2),
                   (1695, 2), (1715, 2), (2080, 0), (2083, 1)]

    def update_label_for_nan_values(data, update_list):
        for index, label in update_list:
            data.at[index, "Label"] = label

    update_label_for_nan_values(data, update_list)
    # drop all the null values
    data.dropna(axis=0, inplace=True)

    X = data["Data"].apply(text_preprocessing)
    y = data["Label"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)
    tokenizer.fit_on_texts(X_train)

    with open('tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)


# create_tokenizer()
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)


def preprocess(own_string):
    own_string = text_preprocessing(own_string)
    own_string_list = [own_string]
    own_string_series = pd.Series(own_string_list)
    # print(own_string_series)
    own_string_sequences = tokenizer.texts_to_sequences(own_string_series)
    # print(own_string_sequences)
    own_string_padded = pad_sequences(
        own_string_sequences, maxlen=maximum_padding_length, padding="post", truncating="post")
    # print(own_string_padded)
    return own_string_padded
