import math
import random
from collections import Counter, defaultdict
import pickle
import sys
import os
sys.path.append(os.path.dirname(__file__))

with open('NLP_Trained_models/spellchecker/models/saved_model_knlm2', 'rb') as inputfile:
    kn_lm2 = pickle.load(inputfile)
