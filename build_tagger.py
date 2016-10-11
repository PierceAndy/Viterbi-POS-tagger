from itertools import product, tee
from math import log
from sys import float_info, argv
import csv
from nltk.data import load
from nltk.tag.util import str2tuple
from tqdm import tqdm


def main():
    training_file, model_file = argv[1:4]

    transition_probabilities, emission_probabilities = train(training_file)
    save_model(transition_probabilities, emission_probabilities, model_file)


def train(training_file):
    """
    Trains hidden Markov model with a tagged training corpus, and returns its transition and emission probabilities.

    The transition probability keeps track of the probability of a part-of-speech tag given the previous tag.
    This is stored in a dictionary with key: tuple(tag1, tag2), value: maximum likelihood estimate of tag2 given tag1.

    The emission probability keeps track of the probability that given a tag, it will be associated with a given token.
    This is stored in a dictionary with key: tuple(token, tag), value: maximum likelihood estimate of tag given token.

    To mitigate division by 0 errors, should the count for a certain POS tag be 0, the probabilities related to it
    would be treated as 0.

    :param training_file: the location of the training file
    :return: a tuple of dictionaries tracking transition probabilities, and emission probabilities.
    """
    tag_counts, transition_counts, emission_counts = read_training_data(training_file)

    transition_probabilities = {}
    emission_probabilities = {}

    for tag1, tag2 in transition_counts.keys():
        transition_probabilities[(tag1, tag2)] = transition_counts[(tag1, tag2)] / tag_counts[tag1] \
            if tag_counts[tag1] else 0

    for token, tag in emission_counts.keys():
        emission_probabilities[(token, tag)] = emission_counts[(token, tag)] / tag_counts[tag] \
            if tag_counts[tag] else 0

    transition_probabilities = apply_log_scale(transition_probabilities)
    emission_probabilities = apply_log_scale(emission_probabilities)

    return transition_probabilities, emission_probabilities


def read_training_data(training_file):
    """
    Extracts part-of-speech (POS) tag, transition between tags, and emission counts from a tagged training corpus.

    The POS tag count keeps track of the number of times a given POS tag occurs in the training data.
    This is stored in a dictionary with POS tag keys and integer count values.

    The transition counts keep track of how often the first tag is followed by a second tag.
    This is stored in a dictionary with tuple(tag1, tag2) keys and the number of times tag2 is followed by tag1 values.

    The emission count keeps track of the number of times a word and its associated tag occurs in the data.
    This is stored in a dictionary with tuple(word, POS tag) keys and integer count values.

    The training file is expected to be a training set of POS-tagged sentences, separated by newline characters.
    Additional custom tags, "START" and "END", are included to indicate the start and end of each sentence.

    :param training_file: the location of the training file
    :return: a tuple of dictionaries tracking tag counts, transition counts, and emission counts
    """
    tag_types = list(load('help/tagsets/upenn_tagset.pickle').keys()) + ["START", "END", "-LRB-", "-RRB-", "#"]
    tag_types = [x for x in tag_types
                 if x not in ["(", ")", "--"]]  # The tagset in nltk uses different notations
    tag_type_permutations = list(product(tag_types, repeat=2))

    tag_counts = dict.fromkeys(tag_types, 0)
    transition_counts = dict.fromkeys(tag_type_permutations, 0)
    emission_counts = {}

    with open(training_file, "r") as training_data:
        for line in tqdm(training_data, total=rawcount(training_file), desc="Training"):

            tagged_tokens = tuple(str2tuple(tagged_token) for tagged_token in line.split())
            tag_sequence = ("START",) + tuple(tagged_token[1] for tagged_token in tagged_tokens) + ("END",)

            for tag in tag_sequence:
                tag_counts[tag] += 1

            for tag_pair in pairwise(tag_sequence):
                transition_counts[tag_pair] += 1

            for tagged_token in tagged_tokens:
                if tagged_token in emission_counts:
                    emission_counts[tagged_token] += 1
                else:
                    emission_counts[tagged_token] = 1

    return tag_counts, transition_counts, emission_counts


def save_model(transition_probabilities, emission_probabilities, model_file):
    """
    Save hidden Markov model transition and emission probabilities to a CSV file.

    :param transition_probabilities: transition probabilities of a trained hidden Markov model
    :param emission_probabilities: emission probabilities of a trained hidden Markov model
    :param model_file: the file to write the hidden Markov model probabilities to
    """
    with open(model_file, 'w') as csvfile:
        model = csv.writer(csvfile, delimiter=' ', newline='')

        # Saving transition probabilities
        model.writerow([len(transition_probabilities)])
        for tag1, tag2 in transition_probabilities.keys():
            model.writerow([tag1, tag2, transition_probabilities[(tag1, tag2)]])

        # Saving emission probabilities
        model.writerow([len(emission_probabilities)])
        for token, tag in emission_probabilities.keys():
            model.writerow([token, tag, emission_probabilities[(token, tag)]])


def pairwise(iterable):
    """
    Returns an iterator of pairs to enable pairwise traversal of an iterable.

    Source: https://docs.python.org/3/library/itertools.html#itertools-recipes

    :param iterable: an iterable object
    :return: an iterator of tuples containing the i-th and i-th+1 elements of the input iterable
    """
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def apply_log_scale(dictionary):
    """
    Applies logarithm to each value in the dictionary.
    If value is not valid, returns the logarithm of the smallest floating point number possible.

    :param dictionary: a dictionary containing values to apply logarithm to
    :return: a dictionary with logarithmic values
    """
    for key, value in dictionary.items():
        dictionary[key] = log(value) if value else log(float_info.min)

    return dictionary


def rawcount(filename):
    """
    Returns the number of lines in a text file in a lightweight and efficient manner.

    Source: http://stackoverflow.com/a/27518377

    :param filename: the location of the text file
    :return: the number of lines in the text file
    """
    f = open(filename, 'rb')
    lines = 0
    buf_size = 1024 * 1024
    read_f = f.raw.read

    buf = read_f(buf_size)
    while buf:
        lines += buf.count(b'\n')
        buf = read_f(buf_size)

    f.close()

    return lines


if __name__ == '__main__':
    main()
