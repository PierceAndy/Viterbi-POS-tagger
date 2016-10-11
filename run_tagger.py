from itertools import tee
from math import log
from sys import float_info, argv
import numpy as np
import csv
from nltk.data import load
from nltk.tag.util import tuple2str
from tqdm import tqdm


def main():
    test_file, model_file, output_file = argv[1:5]

    transition_probabilities, emission_probabilities = read_model(model_file)
    test(test_file, output_file, transition_probabilities, emission_probabilities)


def read_model(model_file):
    """
    Read hidden Markov model transition and emission probabilities from a CSV file.

    :param model_file: the file to read the hidden Markov model probabilities from
    :return: a tuple of dictionaries of transition and emission probabilities of a hidden Markov model
    """
    with open(model_file, 'r') as csvfile:
        model = csv.reader(csvfile, delimiter=' ', newline='')

        transition_probabilities = {}
        emission_probabilities = {}

        number_of_transition_probabilities = int(next(model)[0])
        for i in range(number_of_transition_probabilities):
            tag1, tag2, probability = next(model)
            transition_probabilities[(tag1, tag2)] = float(probability)

        number_of_emission_probabilities = int(next(model)[0])
        for i in range(number_of_emission_probabilities):
            token, tag, probability = next(model)
            emission_probabilities[(token, tag)] = float(probability)

    return transition_probabilities, emission_probabilities


def test(test_file, output_file, transition_probabilities, emission_probabilities):
    """
    Performs part-of-speech tagging on each word in a sentence in the test file using Viterbi algorithm.

    Probabilities are calculated on a logarithmic scale to avoid problems associated with multiplying small floating
    point numbers.

    For cases where emission probabilities of a token given a tag were not encountered in training data, the emission
    probability is treated as 0. As probabilities are being calculated on a logarithmic scale, and log(0) is undefined,
    log(float_info.min) is used as a substitute, representing the logarithmic value as it approaches 0.

    Side effects of substituting "0" probabilities with log(float_info.min):
    - in cases where there are no unknown tokens (defined as tokens not encountered in the tagged training corpus),
    log(float_info.min) serves as a heavy penalty to a path's probability where a "0" probability occurs, relative to
    other paths without a "0" probability. When considering the best paths for a token's tag types, a path with
    log(float_info.min) in its probability will effectively fare poorer in comparisons with paths without such "0"
    probabilities, as log(float_info.min) will be several magnitudes smaller than the smallest probabilities.
    - in cases where there are unknown tokens, every emission probability related to that token given a tag type will be
    "0" (i.e. log(float_info.min)). As path probabilities are being calculated on a logarithmic scale, probabilities are
    being added and not multiplied, thus encountering a single unknown token in a sequence will not render all paths in
    the trellis 0. Instead, log(float_info.min) would be a flat penalty across all paths, thus effectively cancelling
    each other out in comparisons. In which case where unknown tokens are encountered, emission probabilities of that
    token are not considered. Rather, only the tag types from the previous token, and the transition probability between
    tag types play a part in determining the best paths.

    Viterbi nodes are initialized to -float_info.max such that in comparisons to determine the best paths, it will be
    replaced by even the worst of path probabilities, as -float_info.max is a googol googol googol (that's 300 zeroes)
    magnitudes smaller than log(float_info.min). Even if a sentence contains several unknown tokens, the resulting path
    probabilities would still be larger than -float_info.max.

    :param test_file: the file containing sentences to perform part-of-speech tagging on
    :param output_file: the file to write the part-of-speech tagged sentences to
    :param transition_probabilities: transition probabilities of a trained hidden Markov model, on a logarithmic scale
    :param emission_probabilities: emission probabilities of a trained hidden Markov model, on a logarithmic scale
    """
    with open(test_file, "r") as test_data, open(output_file, "w") as output:
        for line in tqdm(test_data, total=rawcount(test_file), desc="Testing "):

            tokens = tuple(line.split())
            tag_types = list(load('help/tagsets/upenn_tagset.pickle').keys()) + ["-LRB-", "-RRB-", "#"]
            tag_types = [x for x in tag_types
                         if x not in ["(", ")", "--"]]  # The tagset in nltk uses different notations

            # Initialize required arrays to model the Viterbi trellis for given test input.
            # The viterbi array keeps track of the best probability path to a token's tag type from the previous token.
            # For each best path, the backpointer array keeps track of the tag type in the previous token.
            viterbi = np.full((len(tokens), len(tag_types)), -float_info.max)
            backpointer = np.full((len(tokens), len(tag_types)), -1, dtype=np.int)

            # Initialize paths in trellis from start to tag types (states) corresponding to first token (observation)
            for t_index, tag in enumerate(tag_types):
                viterbi[0][t_index] = transition_probabilities[("START", tag)] \
                                      + emission_probabilities.get((tokens[0], tag), log(float_info.min))

            # Iteratively fill out Viterbi path probabilities between tag types of each token and the tag types of the
            # token immediately preceding it in the sequence
            if len(tokens) > 1:
                for token_index, (prev_token, curr_token) in enumerate(pairwise(tokens)):
                    for ctag_index, curr_tag in enumerate(tag_types):
                        for ptag_index, prev_tag in enumerate(tag_types):
                            temp_viterbi = viterbi[token_index][ptag_index] \
                                          + transition_probabilities[(prev_tag, curr_tag)] \
                                          + emission_probabilities.get((curr_token, curr_tag), log(float_info.min))

                            if temp_viterbi >= viterbi[token_index+1][ctag_index]:
                                viterbi[token_index + 1][ctag_index] = temp_viterbi
                                backpointer[token_index+1][ctag_index] = ptag_index

            # Determine the best terminating path from the last token
            last_token_index = len(tokens)-1
            end_viterbi = -float_info.max
            end_backpointer = -1
            for tag_index, prev_tag in enumerate(tag_types):
                temp_viterbi = viterbi[last_token_index][tag_index] \
                               + transition_probabilities[(prev_tag, "END")]

                if temp_viterbi >= end_viterbi:
                    end_viterbi = temp_viterbi
                    end_backpointer = tag_index

            # Perform Viterbi backtrace, finding the most likely tag type sequence through best paths to the beginning
            likeliest_tag_indexes = [-1] * len(tokens)
            likeliest_tag_indexes[-1] = end_backpointer
            for token_index in reversed(range(len(tokens)-1)):
                likeliest_tag_indexes[token_index] = backpointer[token_index+1][likeliest_tag_indexes[token_index+1]]

            # Formatting output
            likeliest_tags = [tag_types[index] for index in likeliest_tag_indexes]
            pos_tagged_line = ' '.join([tuple2str(tagged_token) for tagged_token in list(zip(tokens, likeliest_tags))])

            output.write(pos_tagged_line + "\n")


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
