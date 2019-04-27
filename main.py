"""
@author: Sriram Veturi
@title: Page Rank Algorithm
@date: 04/05/2019
"""

# Imports here
import os
import re
import nltk
import json
import argparse

# Global variables
ABSTRACTS = 'abstracts'
GOLD = 'gold'
NOUNS_ADJECTIVES = ['nn', 'nns', 'nnp', 'nnps', 'jj']
STOP_WORDS = nltk.corpus.stopwords.words('english')
ALPHA = 0.85

# Parse arguments
parser = argparse.ArgumentParser(description="Page Rank Implementation")
parser.add_argument(
                        '--data_path',
                        type=str,
                        help="Path where the data files are stored."
                   )
parser.add_argument(
                        '--w',
                        type=int,
                        help="Parameter 'w' for the implementation."
)
args = parser.parse_args()


def get_user_intput():
    """
    This function gets the user's input for data path and w parameter.
    :return: data_path, w
    """

    data_path = args.data_path
    w_parameter = args.w
    return data_path, w_parameter


def get_files_directory(directory):
    """
    This function gets all the files in the directory specified.
    :param directory: Directory where the files are stored
    :return: file_list: List of files in the directory
    """

    file_list = list()
    files = os.listdir(directory)
    for file in files:

        file_list.append(file)

    return file_list


def get_sub_dir_path(parent_dir):
    """
    This function should return the joined path of 'abstracts' and 'gold' with parent directory.
    :param parent_dir: Parent Directory
    :return: abstracts_path, gold_path
    """

    abstracts_path = os.path.join(parent_dir, ABSTRACTS)
    gold_path = os.path.join(parent_dir, GOLD)
    return abstracts_path, gold_path


def get_abstract_gold_files(main_path):
    """
    This function returns all the files in the 'abstracts' and 'gold' directory
    :param main_path: Parent directory
    :return: abstracts_files, gold_files
    """

    abstracts_path, gold_path = get_sub_dir_path(main_path)
    gold_files = get_files_directory(gold_path)
    abstracts_files = get_files_directory(abstracts_path)
    abstracts_files = [file for file in abstracts_files if file in gold_files]
    return abstracts_files, gold_files


def get_file_contents_dict(file_list, parent_dir):
    """
    This function creates a dictionary with filename and its contents.
    :param file_list: List of files
    :param parent_dir: Parent directory
    :return: file_contents_dict
    """

    file_contents_dict = dict()
    for file in file_list:

        file_dir_path = os.path.join(parent_dir, file)
        with open(file_dir_path, 'r', errors='ignore') as f:

            file_contents_dict[file] = f.read()

    return file_contents_dict


def preprocess_data(abstracts_dict):
    """
    This function does the preprocessing of the dictionary contents.
    :param abstracts_dict: abstracts folder files and contents dictionary
    :return: abstracts_dict, all_abstract_dict
    """

    porter_stemmer = nltk.stem.PorterStemmer()
    all_abstract_dict = dict()
    # Preprocess abstracts dictionary
    for file_name, contents in abstracts_dict.items():

        tokens = contents.split()
        tokens = [token.lower() for token in tokens]
        tokens = [re.sub('\d', '', token) for token in tokens]
        # regex to remove all special characters excpet '_' (underscore)
        tokens = [re.sub('[^A-Za-z0-9_]+', '', token) for token in tokens]
        words_without_annotations = [porter_stemmer.stem(word.split('_')[0]) for word in tokens]
        words_without_annotations[:] = [x for x in words_without_annotations if x != '']
        all_abstract_dict[file_name] = words_without_annotations
        # Keep only specified nouns and adjectives
        valid_tokens = list()
        for token in tokens:

            for noun_adj_type in NOUNS_ADJECTIVES:

                if noun_adj_type in token:

                    valid_tokens.append(token)

        tokens = [porter_stemmer.stem(token.split('_')[0]) for token in valid_tokens]
        tokens = [token for token in tokens if token not in STOP_WORDS]
        tokens[:] = [x for x in tokens if x != '']
        abstracts_dict[file_name] = tokens

    return abstracts_dict, all_abstract_dict


def create_words_graph(abstracts_dict_subset, abstracts_dict_superset, w):
    """
    This function should create the word graph from the dictionary
    :param abstracts_dict_subset: abstracts dictionary with only nouns and adjectives
    :param abstracts_dict_superset: original abstracts dictionary
    :param w: w parameter
    :return: abstracts_graph_dict, abstracts_dict_subset, abstracts_dict_superset
    """

    abstracts_graph_dict = dict()
    for file_name, contents in abstracts_dict_subset.items():

        graph_dict = dict()
        subset_words = set(contents)
        superset_words = abstracts_dict_superset[file_name]
        for subset_word in list(subset_words):

            # graph_dict[subset_word] = list()
            words_in_graph = list()
            for superset_word in list(subset_words):

                if subset_word != superset_word:

                    try:

                        if superset_word in superset_words:

                            superset_word_index = superset_words.index(superset_word)

                        if subset_word in superset_words:

                            subset_word_index = superset_words.index(subset_word)

                        if abs(superset_word_index - subset_word_index) < w:

                            if superset_word not in words_in_graph:

                                words_in_graph.append(superset_word)

                    except:

                        print("Do Nothing")

            graph_dict[subset_word] = words_in_graph

        abstracts_graph_dict[file_name] = graph_dict

    for file_name, contents in abstracts_graph_dict.items():

        for word in contents:

            for words_in_graph in contents[word]:

                weight1, weight2 = list(), list()
                weight = 0

                for superset_word in  abstracts_dict_superset[file_name]:

                    if superset_word == word:

                        if superset_word in abstracts_dict_superset[file_name]:

                            weight1.append(abstracts_dict_superset[file_name].index(superset_word))

                    if superset_word == words_in_graph:

                        if superset_word in abstracts_dict_superset[file_name]:

                            weight2.append(abstracts_dict_superset[file_name].index(superset_word))

                for x in weight1:

                    for y in weight2:

                        if (x - y) != 1 or (x - y) != -1:

                            continue

                        else:

                            weight += 1

                words_in_graph_index = contents[word].index(words_in_graph)
                contents[word][words_in_graph_index] = [words_in_graph, weight]

    return abstracts_graph_dict, abstracts_dict_subset, abstracts_dict_superset

def rank_pages(abstracts_graph_dict, abstracts_dict_subset, abstracts_dict_superset):
    """
    This function should rank the pages
    :param abstracts_graph_dict: abstracts word graph
    :param abstracts_dict_subset: abstracts dictionary with only nouns and adjectives
    :param abstracts_dict_superset: original abstracts dictionary
    :return: abstracts_dict_superset, abstracts_dict_subset, file_scores
    """

    file_scores = dict()
    for file_name, words in abstracts_dict_subset.items():

        word_scores = dict()
        unique_words = set(words)
        for word in unique_words:

            word_scores[word] = 1.0 / len(unique_words)

        file_scores[file_name] = word_scores

    for file_name, words in abstracts_dict_subset.items():

        previous_score = list()
        current_score = file_scores[file_name]
        current_score_keys = current_score.keys()
        first_current_score_keys = list(current_score_keys)[0]
        something = list(current_score.keys())[0]
        pi = current_score[first_current_score_keys]
        not_converged = True

        while not_converged:

            previous_score = current_score
            for word, individual_score in current_score.items():

                complex_term = 0.0
                for word_weight in abstracts_graph_dict[file_name][word]:

                    s = 0.0
                    for word_score_tuple in abstracts_graph_dict[file_name][word_weight[0]]:

                        s += word_score_tuple[1]

                    if s == 0.0:

                        continue

                    else:

                        complex_term += float((float(word_weight[1]) / float(s)) * float(current_score[word_weight[0]]))

                try:

                    current_score[word] = (float(ALPHA) * float(complex_term)) + (float(1 - ALPHA) * float(pi))

                except:

                    current_score[word] = 0.0

            if current_score == previous_score:

                not_converged = False

            else:

                continue

    return abstracts_dict_superset, abstracts_dict_subset, file_scores


def form_ngrams(l, words, n):
    """
    This function should form the ngrams (uni, bi, tri)
    :param l: list of ngram scores
    :param words: adjacent words
    :param n: 1/2/3
    :return: l
    """

    l += [words[x:x + n] for x in range(len(words)-n-1)]
    return l


def form_n_grams(adjacent_words):
    """
    This function should perform global ngrams calculations
    :param adjacent_words: word list
    :return: n_grams_scores
    """

    n_grams_scores = list()
    n_grams_scores = form_ngrams(n_grams_scores, adjacent_words, 1)
    n_grams_scores = form_ngrams(n_grams_scores, adjacent_words, 2)
    n_grams_scores = form_ngrams(n_grams_scores, adjacent_words, 3)
    return n_grams_scores


def get_individual_n_gram_scores(abstracts_dict_superset, abstracts_dict_subset, file_scores):
    """
    This function should get individual words ngram scores
    :param abstracts_dict_superset: original abstracts dictionary
    :param abstracts_dict_subset: abstracts dictionary with only nouns and adjectives
    :param file_scores: scores
    :return: n_grams_dict
    """

    n_grams_dict = dict()
    for file_name, words in abstracts_dict_superset.items():

        n_grams_list = form_n_grams(words)
        file_n_grams = list()
        for n_gram in n_grams_list:

            flag = False
            for x in n_gram:

                unique_subset_words = set(abstracts_dict_subset[file_name])
                if x in unique_subset_words:

                    continue

                else:

                    flag = True

            if flag is not True:

                if n_gram not in file_n_grams:

                    file_n_grams.append(n_gram)

        try:

            n_grams_dict[file_name] = file_n_grams

        except:

            print("DEAD END")
            try:

                n_grams_dict[file_name] = set(file_n_grams)

            except:

                print("DEAD END")

    return n_grams_dict


def n_grams_update_and_sort(n_grams_dict, file_scores):
    """
    This function should sort the ngrams socres list
    :param n_grams_dict: ngrams dictionary
    :param file_scores: scores
    :return: n_grams_dict_scores
    """

    n_grams_dict_scores = dict()
    for file_name, n_grams in n_grams_dict.items():

        individual_n_grams = dict()
        for n_gram in n_grams:

            score = 0
            for ind_n_gram in n_gram:

                score += file_scores[file_name][ind_n_gram]

            individual_n_grams[" ".join(n_gram)] = score

        n_grams_dict_scores[file_name] = individual_n_grams

    for file_name, n_gram_scores in n_grams_dict_scores.items():

        sorted_n_gram_scores = sorted(n_gram_scores.items(), key=lambda x: x[1], reverse=True)
        n_grams_dict_scores[file_name] = sorted_n_gram_scores

    return n_grams_dict_scores


def preprocess_gold_files(gold_dict):
    """
    This function should preprocess the gold files
    :param gold_dict: gold dictionary
    :return: gold_dict
    """

    porter_stemmer = nltk.stem.PorterStemmer()
    # Preprocess abstracts dictionary
    for file_name, contents in gold_dict.items():

        tokens = contents.split()
        tokens = [token.lower() for token in tokens]
        tokens = [re.sub('\d', '', token) for token in tokens]
        tokens = [re.sub('\n', '', token) for token in tokens]
        # regex to remove all special characters excpet '_' (underscore)
        tokens = [re.sub('[^A-Za-z0-9]+', '', token) for token in tokens]
        words_without_annotations = [porter_stemmer.stem(word) for word in tokens]
        words_without_annotations[:] = [x for x in words_without_annotations if x != '']
        words_without_annotations = " ".join(words_without_annotations)
        gold_dict[file_name] = words_without_annotations

    return gold_dict


def get_mmr(k, n_grams_dict, gold_dict):
    """
    This function should get the mmr calculations
    :param k: 1-10
    :param n_grams_dict: ngrams dictionary
    :param gold_dict: gold dictionary
    :return: global_mean_reciprocal_rank
    """

    mean_reciprocal_rank = 0
    for file_name, contents in gold_dict.items():

        rank_found = False
        mod_d = len(n_grams_dict[file_name][:k])
        for d in range(mod_d):

            n_gram_score_tuple = n_grams_dict[file_name][d]
            if n_gram_score_tuple[0] not in gold_dict[file_name]:

                continue

            else:

                rank = d
                rank_found = True

            if rank_found is True:

                break

        try:

            mean_reciprocal_rank += 1.0 / rank

        except:

            continue

    mod_capital_d = len(gold_dict)
    global_mean_reciprocal_rank = float(mean_reciprocal_rank) / float(mod_capital_d)
    return global_mean_reciprocal_rank


def get_final_mmr_scores(n_grams_dict, gold_dict):
    """
    This function records and prints the mmr scores
    :param n_grams_dict: ngrams dictionary
    :param gold_dict: gold dictionary
    :return: final_mmr_dict
    """

    final_mmr_dict = dict()
    for k in range(10):

        final_mmr = get_mmr(k+1, n_grams_dict, gold_dict)
        print("K = {0} --> MMR : {1}".format(k+1, final_mmr))
        final_mmr_dict[k+1] = final_mmr

    return final_mmr_dict


def get_files(file_path):
    """
    This function should get all the necessary files.
    :param file_path: Path to the files
    :return: abstracts_dict, gold_dict
    """

    abstracts_files, gold_files = get_abstract_gold_files(file_path)
    abstracts_path, gold_path = get_sub_dir_path(file_path)
    abstracts_dict = get_file_contents_dict(abstracts_files, abstracts_path)
    gold_dict = get_file_contents_dict(gold_files, gold_path)
    return abstracts_dict, gold_dict


def preprocess_files(abstracts_dict, gold_dict):
    """
    Driver function for preprocessing files.
    :param abstracts_dict: abstracts dicitonary
    :param gold_dict: gold dictionary
    :return: gold_dict, abstracts_dict_subset, abstracts_dict_superset
    """

    gold_dict = preprocess_gold_files(gold_dict)
    abstracts_dict_subset, abstracts_dict_superset = preprocess_data(abstracts_dict)
    return gold_dict, abstracts_dict_subset, abstracts_dict_superset


def page_rank_operations(abstracts_dict_subset, abstracts_dict_superset):
    """
    Driver Function for Page-Rank operations.
    :param abstracts_dict_subset: abstracts dictionary with only nouns and adjectives
    :param abstracts_dict_superset: original abstracts dictionary
    :return: abstracts_dict_superset, abstracts_dict_subset, file_scores
    """
    abstracts_graph_dict, abstracts_dict_subset, abstracts_dict_superset = create_words_graph(abstracts_dict_subset,
                                                                                              abstracts_dict_superset,
                                                                                              w)
    abstracts_dict_superset, abstracts_dict_subset, file_scores = rank_pages(abstracts_graph_dict,
                                                                             abstracts_dict_subset,
                                                                             abstracts_dict_superset)
    return abstracts_dict_superset, abstracts_dict_subset, file_scores


def n_grams_operations(abstracts_dict_superset, abstracts_dict_subset, file_scores):
    """
    Driver function for ngram operations.
    :param abstracts_dict_superset: original abstracts dictionary
    :param abstracts_dict_subset: abstracts dictionary with only nouns and adjectives
    :param file_scores: scores
    :return: n_grams_dict
    """

    n_grams_dict = get_individual_n_gram_scores(abstracts_dict_superset, abstracts_dict_subset, file_scores)
    n_grams_dict = n_grams_update_and_sort(n_grams_dict, file_scores)
    return  n_grams_dict


# Main Function starts here..
if __name__ == "__main__":

    data_path, w = get_user_intput()
    abstracts_dict, gold_dict = get_files(data_path)
    print("Files Found.\n")
    gold_dict, abstracts_dict_subset, abstracts_dict_superset = preprocess_files(abstracts_dict, gold_dict)
    print("Preprocessed Files.\n")
    abstracts_dict_superset, abstracts_dict_subset, file_scores = page_rank_operations(abstracts_dict_subset, abstracts_dict_superset)
    n_grams_dict = n_grams_operations(abstracts_dict_superset, abstracts_dict_subset, file_scores)
    print("Page Rank Operations Done.\n")
    print("MMR results:\n")
    final_mmr_dict = get_final_mmr_scores(n_grams_dict, gold_dict)
    json_file_name = 'mmr_map.json'
    with open(json_file_name, 'w') as fp:
        json.dump(final_mmr_dict, fp)
    print("\nOpen '{}' to find MMR results.".format(json_file_name))
