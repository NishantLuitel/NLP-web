import joblib
from .ngram import nGramLangugageModel

vocabulary = joblib.load('NLP_Trained_models/n_gram/Nepali_Corpus/vocabulary.pkl')
n_gram_counts_list = joblib.load('NLP_Trained_models/n_gram/Nepali_Corpus/n_gram_counts_list.pkl')

model = nGramLangugageModel()

#previous_tokens = ['त्यो', 'त']

def suggest(previous_tokens):
    """
        Return no_suggestions of suggested words 

        Arguments:
            previous_tokens : previous n-gram
        
        Output:
            list of suggested words 
    """
    return model.return_suggestions(previous_tokens, vocabulary, n_gram_counts_list,
                                    n_gram_index=[0, 1], no_suggestions=5)


