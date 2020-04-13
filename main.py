from collections import defaultdict, Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

import math
import os
import pickle
import re


class NaiveBayesClassifier:
    def __init__(self, retrain=False):
        # stopwords that are excluded in processing text
        self.stopwords = set(stopwords.words('english'))

        # The filename of the pickle file that contains the saved params
        self.saved_params_filename = '_nbc_params_v2.pickle'

        # Model parameters that is used to classify input texts
        self.params = {
            # The prior probability of positive and negative reviews
            'pos_prior': 0,
            'neg_prior': 0,

            # The number of words in all documents in positive and negative reviews
            'n_pos_words': 0,
            'n_neg_words': 0,

            # The number of unique words across positive and negative reviews
            'n_vocab': 0,

            # Dictionary that contains the likelihood of all the words in the vocabulary
            'pos_likelihood': defaultdict(float),
            'neg_likelihood': defaultdict(float),
        }

        # Initialize model params (load saved params or retrain)
        self._init_params()

    def _init_params(self):
        if os.path.isfile(self.saved_params_filename):
            with open(self.saved_params_filename, 'rb') as f:
                self.params = pickle.load(f)
                print('INFO - Loaded model parameters from pickle.')
        else:
            print('INFO - No saved model parameters. Model will be retrained.')
            self._train_model()

    def _train_model(self):
        # read all training samples (positive and negative reviews)
        pos_train = get_samples('aclImdb/train/pos')
        neg_train = get_samples('aclImdb/train/neg')

        # count length
        n_pos_train = len(pos_train)
        n_neg_train = len(neg_train)
        n_samples = n_pos_train + n_neg_train

        # compute class prior probabilities
        pos_prior = math.log(n_pos_train / n_samples)
        neg_prior = math.log(n_neg_train / n_samples)

        # bag of words container of positive and negative samples
        pos_bow = Counter()
        neg_bow = Counter()

        print('INFO - Generating BOW for positive training samples')
        for sample in pos_train:
            sample_contents = sample[1]
            sample_tokens = self.tokenize(sample_contents)
            pos_bow += Counter(sample_tokens)

        print('INFO - Generating BOW for negative training samples')
        for sample in neg_train:
            sample_contents = sample[1]
            sample_tokens = self.tokenize(sample_contents)
            neg_bow += Counter(sample_tokens)

        # count the number of words for each classes
        # TODO : stopwords are excluded in this count so maybe correct this ?
        n_pos_words = sum(pos_bow.values())
        n_neg_words = sum(neg_bow.values())

        # collate all unique words in a set
        print('INFO - Collating vocabulary of unique words')
        vocabulary = set(pos_bow.keys()).union(set(neg_bow.keys()))
        n_vocab = len(vocabulary)

        # compute likelihood of all the words
        pos_likelihood = defaultdict(float)
        neg_likelihood = defaultdict(float)

        pos_denominator = n_pos_words + n_vocab
        neg_denominator = n_neg_words + n_vocab

        print('INFO - Calculating words likelihoods for positive training samples')
        for word in vocabulary:
            frequency = pos_bow[word] + 1
            pos_likelihood[word] = math.log(frequency / pos_denominator)

        print('INFO - Calculating words likelihoods for negative training samples')
        for word in vocabulary:
            frequency = neg_bow[word] + 1
            neg_likelihood[word] = math.log(frequency / neg_denominator)

        # prepare computed model parameters for serialization
        self.params['pos_prior'] = pos_prior
        self.params['neg_prior'] = neg_prior
        self.params['n_pos_words'] = n_pos_words
        self.params['n_neg_words'] = n_neg_words
        self.params['n_vocab'] = n_vocab
        self.params['pos_likelihood'] = pos_likelihood
        self.params['neg_likelihood'] = neg_likelihood

        # Save model parameters for later use
        with open(self.saved_params_filename, 'wb') as f:
            pickle.dump(self.params, f)

    def tokenize(self, text):
        # clean out unnecessary whitespace in text and lowercase text
        text = text.strip().lower()

        # use nltk to tokenize text
        tokens = word_tokenize(text)

        # filter out stopwords
        tokens = [token for token in tokens if token not in self.stopwords]

        # filter out punctuation marks
        return [token for token in tokens if re.search(r'[a-z]', token)]

    def classify(self, text):
        '''
        Classify the given text whether it is a positive or negative review.
        Returns 1 if the text is positive and -1 if it's negative.
        '''

        words = self.tokenize(text)

        # Initialize scores to the prior probabilities of the classes
        positive = self.params['pos_prior']
        negative = self.params['neg_prior']

        # The default likelihood of an unknown word
        def_pos_ll = math.log(1 / self.params['n_pos_words'])
        def_neg_ll = math.log(1 / self.params['n_neg_words'])

        for word in words:
            positive += self.params['pos_likelihood'].get(word, def_pos_ll)
            negative += self.params['neg_likelihood'].get(word, def_neg_ll)

            # program sanity check
            if positive == 0 or negative == 0 or positive == negative:
                raise AssertionError(
                    'Error : Probabilities resulted to zeroes')

        return 1 if positive > negative else -1


def get_samples(directory):
    assert os.path.isdir(directory)

    pattern = re.compile(r'(\d+)_(\d+)\.txt')
    samples = []

    print('INFO - Getting samples from directory: ' + directory)
    for name in os.listdir(directory):
        match = pattern.match(name)

        if match:
            review_id = int(match.group(1))
            rating = int(match.group(2))

            with open(os.path.join(directory, name), 'r') as f:
                contents = ' '.join(f.readlines())
                review_classification = 1 if rating >= 7 else -1

                samples.append((review_id, contents, review_classification))

    return samples


if __name__ == '__main__':
    model = NaiveBayesClassifier()
    pos_samples = get_samples('aclImdb/test/pos')
    neg_samples = get_samples('aclImdb/test/neg')
    all_samples = pos_samples + neg_samples

    # accuracy stuff
    tp = 0  # true positives
    tn = 0  # true negatives
    fp = 0  # false positives
    fn = 0  # false negatives

    print('INFO - Testing classifier against test samples...')
    for sample in all_samples:
        _, text, actual = sample[0], sample[1], sample[2]
        prediction = model.classify(text)

        if actual is 1:
            if prediction is 1:
                tp += 1
            else:
                fn += 1
        else:
            if prediction is 1:
                fp += 1
            else:
                tn += 1

    # compute accuracy stuff
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f_measure = (2 * precision * recall) / (precision + recall)
    accuracy = (tp + tn) / (tp + fp + tn + fn)

    # report accuracy stuff
    print('EVALUATION - Precision: %.4f' % precision)
    print('EVALUATION - Recall: %.4f' % recall)
    print('EVALUATION - F-measure: %.4f' % f_measure)
    print('EVALUATION - Accuracy: %.4f' % accuracy)
