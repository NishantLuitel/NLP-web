class nGramLangugageModel:
    def __init__(self):
        pass

    def count_n_grams(self, data, n, start_token="", end_token=""):
        """
        Count all the n-grams of the data

        Input :
          data = list of tokenized sentences
          n = 1 for unigram, 2 for bigram and 3 for trigram and so on......

        Output :
          n_grams : dictionary with key of n-gram and value of the count of the corresponding n-gram
        """

        n_grams = {}

        for sentence in data:
            # prepend start token n times and append end token one time
            sentence = [start_token] * n + sentence + [end_token]

            # convert list to tuple so that the sequence of words can be used as a key in the dictionary
            sentence = tuple(sentence)

            # use value of m to denote the number of n grams in the current sentence
            m = len(sentence) if n == 1 else len(sentence) - 1

            for i in range(m):
                n_gram = sentence[i: i + n]

                # check if the n-gram is in the dictionary
                # if present, increase the value else se the value of n-gram to 1
                if n_gram in n_grams.keys():
                    n_grams[n_gram] += 1
                else:
                    n_grams[n_gram] = 1

        return n_grams

    def estimate_probability(self, word, previous_n_gram, n_gram_counts, n_plus1_gram_counts, vocabulary_size, k=1.0):
        """
        Estimate the probabilities of the next word using the n-gram counts with k-smoothing

        Input :
          word : next word to be predicted
          previous_n_gram : given input words by the user of n-gram
          n_gram_counts : output of the count_n_gram functions for n-gram
          n_plus1_gram_counts : output of the count_n_gram functions for (n+1)-gram
          vocabulary_size : length of the vocabulary
          k : constant for k-smoothing

        Output:
          probability : probability that the "word" appear after the "previous_n_gram"
        """

        # convert the list to tuple to use it as a dictionary key
        previous_n_gram = tuple(previous_n_gram)

        # set the denominator
        # if the previous n-gram exists in the dictionary of n-gram counts, get its coun
        # else set the count to zero
        # use the dictionary that has counts for n-grams
        previous_n_gram_count = n_gram_counts[previous_n_gram] if previous_n_gram in n_gram_counts else 0

        # calculate the denominator useing the count fo the previous n gram and apply k-smoothing
        denominator = previous_n_gram_count + k * vocabulary_size

        # define n plus 1 gram as the previous n-gram plus the current word as a tuple
        n_plus1_gram = previous_n_gram + (word, )

        # set the count to the count in the dictionary
        # 0 if not in the dictionary
        # use the dictionary that has counts for the n-gram plus current word
        n_plus1_gram_count = n_plus1_gram_counts[n_plus1_gram] if n_plus1_gram in n_plus1_gram_counts else 0

        # define the numerator using the counf of the n-gram plus current word and apply smoothing
        numerator = n_plus1_gram_count + k

        # calculate the probability as the numberator divided by the denominator
        probability = numerator / denominator

        return probability

    def estimate_probabilities(self, previous_n_gram, n_gram_counts, n_plus1_gram_counts, vocabulary, k=1.0):
        """
        Estimate the probabilities of next words using the n-gram counts with k-smoothing

        Input :
          previous_n_gram : given input words by the user of n-gram
          n_gram_counts : output of the count_n_gram functions for n-gram
          n_plus1_gram_counts : output of the count_n_gram functions for (n+1)-gram
          vocabulary : list of unique words in the training datasets
          k : constant for k-smoothing

        Output:
          probabilities : dictionary of probability that the "word" in "vocabulary" appear after the "previous_n_gram"
        """
        # convert the list to tuple to use it as a dictionary key
        previous_n_gram = tuple(previous_n_gram)

        # add  and  to the vocabulary
        #  is not needed as it should not appear as the next word
        vocabulary = vocabulary + ["", ""]
        vocabulary_size = len(vocabulary)

        probabilities = {}
        for word in vocabulary:
            probability = self.estimate_probability(
                word, previous_n_gram, n_gram_counts, n_plus1_gram_counts, vocabulary_size, k=k)
            probabilities[word] = probability

        return probabilities

    def suggest_a_word(self, previous_tokens, n_gram_counts, n_plus1_gram_counts, vocabulary, k=1.0, start_with=None):
        """
        Get suggestion for the next word

        Input :
          previous_n_gram : given input words by the user of n-gram
          n_gram_counts : output of the count_n_gram functions for n-gram
          n_plus1_gram_counts : output of the count_n_gram functions for (n+1)-gram
          vocabulary : list of unique words in the training datasets
          k : constant for k-smoothing
          start_with : starting letters of the word to be suggested

        Output :
          suggestion : word in vocabulary with the highest probability
          max_prob : corresponding probabilirt of the suggested word after the given n-gram
        """

        # length of previous words
        n = len(list(n_gram_counts.keys())[0])

        # From the words that the user already typed, get the most recent 'n' words as the previous n-gram
        previous_n_gram = previous_tokens[-n:]

        # Estimate the probabilities that each word in the vocabular is the next word
        probabilities = self.estimate_probabilities(
            previous_n_gram, n_gram_counts, n_plus1_gram_counts, vocabulary, k=k)

        # Words with highest probability will be set to suggestion
        suggestion = None

        # Initialie the value for maximum probability
        max_prob = 0

        # For each word and its probability in the probabilities dictionary
        for word, prob in probabilities.items():
            # if the optional start with string is set
            if start_with != None:
                # Check if the beginning of word does not match with the letters in 'start_with'
                if not word.startswith(start_with):
                    # if they don't match, skip this word and move onto the next word
                    continue

            # Check if this word's probability is greater than the current maximum probability
            if prob > max_prob:
                # if so, save this word for the best suggestion
                suggestion = word
                max_prob = prob

        return suggestion, max_prob

    def suggest_words(self, previous_tokens, n_gram_counts, n_plus1_gram_counts, vocabulary, k=1.0, start_with=None, no_suggestions=3):
        """
        Get suggestion for the next word

        Input :
          previous_n_gram : given input words by the user of n-gram
          n_gram_counts : output of the count_n_gram functions for n-gram
          n_plus1_gram_counts : output of the count_n_gram functions for (n+1)-gram
          vocabulary : list of unique words in the training datasets
          k : constant for k-smoothing
          start_with : starting letters of the word to be suggested
          no_suggestions : No. of suggestions provided for the next word

        Output :
          suggestions : dictionary of (no_suggestions) of "word" in "vocabulary" with the highest probability
        """

        # length of previous words
        n = len(list(n_gram_counts.keys())[0])

        # From the words that the user already typed, get the most recent 'n' words as the previous n-gram
        previous_n_gram = previous_tokens[-n:]

        # Estimate the probabilities that each word in the vocabular is the next word
        probabilities = self.estimate_probabilities(
            previous_n_gram, n_gram_counts, n_plus1_gram_counts, vocabulary, k=k)

        # Words with highest probability will be set to suggestion
        suggestions = {}

        # For each word and its probability in the probabilities dictionary
        for word, prob in probabilities.items():
            # if the optional start with string is set
            if start_with != None:
                # Check if the beginning of word does not match with the letters in 'start_with'
                if not word.startswith(start_with):
                    # if they don't match, skip this word and move onto the next word
                    continue

            if len(suggestions) < no_suggestions:
                suggestions[word] = prob
            else:
                # find the suggestions (key, value) with the smallest probability
                suggestions_list = list(suggestions.items())
                suggest_key = suggestions_list[0][0]
                suggest_prob = suggestions_list[0][1]
                for i in range(1, no_suggestions):
                    if suggest_prob > suggestions_list[i][1]:
                        suggest_key = suggestions_list[i][0]
                        suggest_prob = suggestions_list[i][1]
                # replace if the smallest probability is smaller than the probability of the current word
                if(suggest_prob < prob):
                    suggestions.pop(suggest_key)
                    suggestions[word] = prob

        # sort the suggestions in descending order
        suggestions = sorted(suggestions.items(), key=lambda x: x[1], reverse=True)

        return suggestions

    def display_suggestions(self, previous_tokens, vocabulary, n_gram_counts_list, n_gram_index, k=1.0, start_with=None, no_suggestions=3):
        """
        Display suggestion for the next word

        Input :
          previous_tokens : given input words by the user
          vocabulary : list of unique words in the training datasets
          n_gram_counts_list : list of output of the count_n_gram functions for n-gram where n = 1, 2, 3, 4, 5
          n_gram_index : index for the n-gram, where = 1 for unigram, = 2 for bigram and so on....
          k : constant for k-smoothing
          start_with : starting letters of the word to be suggested
          no_suggestions : No. of suggestions provided for the next word

        Output :
          Display the suggested words
        """
        print(f"The previous words are {previous_tokens}, the suggestions are:")

        for index in n_gram_index:
            tmp_suggest = self.suggest_words(
                previous_tokens, n_gram_counts_list[index], n_gram_counts_list[index + 1], vocabulary, k=k, start_with=start_with, no_suggestions=no_suggestions)

    def return_suggestions(self, previous_tokens, vocabulary, n_gram_counts_list, n_gram_index, k = 1.0, start_with = None, no_suggestions = 3):
      """
      Display suggestion for the next word

      Input :
        previous_tokens : given input words by the user
        vocabulary : list of unique words in the training datasets
        n_gram_counts_list : list of output of the count_n_gram functions for n-gram where n = 1, 2, 3, 4, 5
        n_gram_index : index for the n-gram, where = 1 for unigram, = 2 for bigram and so on....
        k : constant for k-smoothing
        start_with : starting letters of the word to be suggested
        no_suggestions : No. of suggestions provided for the next word

      Output :
        suggestions : no_suggestions * 2 suggest words list
      """
      suggestions = []
      for index in n_gram_index:
        tmp_suggest = self.suggest_words(previous_tokens, n_gram_counts_list[index], n_gram_counts_list[index + 1], vocabulary, k = k, start_with = start_with, no_suggestions = no_suggestions)
        for suggest in tmp_suggest:
          suggestions.append(suggest)
      
      suggestions.sort(key = lambda x : x[1], reverse = True)

      return suggestions[0:no_suggestions]
