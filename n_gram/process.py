import joblib
from .ngram import nGramLangugageModel, preprocessText

vocabulary = joblib.load('NLP_Trained_models/n_gram/Nepali_Corpus/vocabulary.pkl')
n_gram_counts_list = joblib.load('NLP_Trained_models/n_gram/Nepali_Corpus/n_gram_counts_list.pkl')


model = nGramLangugageModel()

#previous_tokens = ['त्यो', 'त']


def suggest(previous_tokens, n_gram_index):
    print("Suggest")
    return model.return_suggestions(previous_tokens, vocabulary, n_gram_counts_list,
                                    n_gram_index=n_gram_index, no_suggestions=2)


def preprocess(input):
    tokenized_input = input.split(' ')
    preProcessModel = preprocessText()
    print("Preprocess:")
    return tokenized_input
    return preProcessModel.preprocess_sample(tokenized_input, vocabulary)
