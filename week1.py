import re
from collections import Counter
import numpy as np
import pandas as pd

# UNQ_C1 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
# GRADED FUNCTION: process_data
def process_data(file_name):
    """
    Input:
        A file_name which is found in your current directory. You just have to read it in.
    Output:
        words: a list containing all the words in the corpus (text file you read) in lower case.
    """
    words = []  # return this variable correctly

    ### START CODE HERE ###
    bulk = []
    with open("data/shakespeare.txt") as f:
        bulk = f.readlines()

    lines = [s.lower() for s in bulk]

    for phrase in lines:
        #pattern = r'[a-zA-Z\'\-]+'
        pattern = r'\w+'
        phrase_words = re.findall(pattern, phrase)
        #print(phrase_words)
        words.extend(phrase_words)

    ### END CODE HERE ###
    return words

#DO NOT MODIFY THIS CELL
word_l = process_data('data/shakespeare.txt')
vocab = set(word_l)  # this will be your new vocabulary
print(f"The first ten words in the text are: \n{word_l[0:10]}")
print(f"There are {len(vocab)} unique words in the vocabulary.")


def get_count(word_l):
    '''
    Input:
        word_l: a set of words representing the corpus.
    Output:
        word_count_dict: The wordcount dictionary where key is the word and value is its frequency.
    '''

    word_count_dict = {}  # fill this with word counts
    ### START CODE HERE
    for word in word_l:
        if word in word_count_dict :
            word_count_dict [word] += 1
        else:
            word_count_dict [word] = 1

    ### END CODE HERE ###
    return word_count_dict

#DO NOT MODIFY THIS CELL
word_count_dict = get_count(word_l)
print(f"There are {len(word_count_dict)} key values pairs")
print(f"The count for the word 'thee' is {word_count_dict.get('thee',0)}")


# UNQ_C3 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
# GRADED FUNCTION: get_probs
def get_probs(word_count_dict):
    '''
    Input:
        word_count_dict: The wordcount dictionary where key is the word and value is its frequency.
    Output:
        probs: A dictionary where keys are the words and the values are the probability that a word will occur.
    '''
    probs = {}  # return this variable correctly

    ### START CODE HERE ###
    corpus_len = 0
    for num in word_count_dict.values():
        corpus_len += num

    for word in word_count_dict:
        probs[word] = word_count_dict[word]/corpus_len

    ### END CODE HERE ###
    return probs

#DO NOT MODIFY THIS CELL
probs = get_probs(word_count_dict)
print(f"Length of probs is {len(probs)}")
print(f"P('thee') is {probs['thee']:.4f}")


# UNQ_C4 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
# UNIT TEST COMMENT: Candidate for Table Driven Tests
# GRADED FUNCTION: deletes
def delete_letter(word, verbose=False):
    '''
    Input:
        word: the string/word for which you will generate all possible words
                in the vocabulary which have 1 missing character
    Output:
        delete_l: a list of all possible strings obtained by deleting 1 character from word
    '''

    delete_l = []
    split_l = []

    ### START CODE HERE ###
    # variant #1
    '''
    for i in range(len(word) + 1):
        print("{}-{}".format(word[:i], word[i:]))
        split_l.append([word[:i], word[i:]])
    '''
    # variant #2
    split_l = [(word[:i], word[i:]) for i in range(len(word) + 1)]

    splits = split_l

    # variant #1
    '''
    print('word : ', word)
    for L, R in splits:
        if R:
            print(L + R[1:], ' <-- delete ', R[0])
    '''
    # variant #2
    delete_l = [L + R[1:] for L, R in splits if R]

    ### END CODE HERE ###

    if verbose: print(f"input word {word}, \nsplit_l = {split_l}, \ndelete_l = {delete_l}")

    return delete_l

delete_word_l = delete_letter(word="cans",  verbose=True)

# test # 2
print(f"Number of outputs of delete_letter('at') is {len(delete_letter('at'))}")


# UNQ_C5 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
# UNIT TEST COMMENT: Candidate for Table Driven Tests
# GRADED FUNCTION: switches
def switch_letter(word, verbose=False):
    '''
    Input:
        word: input string
     Output:
        switches: a list of all possible strings with one adjacent charater switched
    '''

    switch_l = []
    split_l = []

    ### START CODE HERE ###
    split_l = [(word[:i], word[i:]) for i in range(len(word))]
    splits = split_l
    switch_l = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
    ### END CODE HERE ###

    if verbose: print(f"Input word = {word} \nsplit_l = {split_l} \nswitch_l = {switch_l}")

    return switch_l

switch_word_l = switch_letter(word="eta", verbose=True)

# test # 2
print(f"Number of outputs of switch_letter('at') is {len(switch_letter('at'))}")


# UNQ_C6 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
# UNIT TEST COMMENT: Candidate for Table Driven Tests
# GRADED FUNCTION: replaces
def replace_letter(word, verbose=False):
    '''
    Input:
        word: the input string/word
    Output:
        replaces: a list of all possible strings where we replaced one letter from the original word.
    '''

    letters = 'abcdefghijklmnopqrstuvwxyz'
    replace_l = []
    split_l = []

    ### START CODE HERE ###

    split_l = [(word[:i], word[i:]) for i in range(len(word))]
    splits = split_l
    replace_l = [L + c + R[1:] for L, R in splits if R for c in letters]
    replace_set = set(replace_l)
    replace_set.remove(word) if word else replace_set

    ### END CODE HERE ###

    # turn the set back into a list and sort it, for easier viewing
    replace_l = sorted(list(replace_set))

    if verbose: print(f"Input word = {word} \nsplit_l = {split_l} \nreplace_l {replace_l}")

    return replace_l

replace_l = replace_letter(word='can', verbose=True)


# UNQ_C7 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
# UNIT TEST COMMENT: Candidate for Table Driven Tests
# GRADED FUNCTION: inserts
def insert_letter(word, verbose=False):
    '''
    Input:
        word: the input string/word
    Output:
        inserts: a set of all possible strings with one new letter inserted at every offset
    '''
    letters = 'abcdefghijklmnopqrstuvwxyz'
    insert_l = []
    split_l = []

    ### START CODE HERE ###
    split_l = [(word[:i], word[i:]) for i in range(len(word))]
    splits = split_l
    splits.append((word, ''))
    insert_l = [L + c + R for L, R in splits if R or L for c in letters]

    ### END CODE HERE ###

    if verbose: print(f"Input word {word} \nsplit_l = {split_l} \ninsert_l = {insert_l}")

    return insert_l

insert_l = insert_letter('at', True)
print(f"Number of strings output by insert_letter('at') is {len(insert_l)}")

# test # 2
print(f"Number of outputs of insert_letter('at') is {len(insert_letter('at'))}")


# UNQ_C8 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
# UNIT TEST COMMENT: Candidate for Table Driven Tests
# GRADED FUNCTION: edit_one_letter
def edit_one_letter(word, allow_switches=True):
    """
    Input:
        word: the string/word for which we will generate all possible wordsthat are one edit away.
    Output:
        edit_one_set: a set of words with one possible edit. Please return a set. and not a list.
    """

    edit_one_set = set()

    ### START CODE HERE ###
    delete_l = delete_letter(word)
    insert_l = insert_letter(word)
    replace_l = replace_letter(word)
    switch_l = switch_letter(word) if allow_switches else []

    all = delete_l + insert_l + replace_l + switch_l

    edit_one_set = set(all)
    ### END CODE HERE ###

    return edit_one_set

tmp_word = "at"
tmp_edit_one_set = edit_one_letter(tmp_word)
# turn this into a list to sort it, in order to view it
tmp_edit_one_l = sorted(list(tmp_edit_one_set))

print(f"input word {tmp_word} \nedit_one_l \n{tmp_edit_one_l}\n")
print(f"The type of the returned object should be a set {type(tmp_edit_one_set)}")
print(f"Number of outputs from edit_one_letter('at') is {len(edit_one_letter('at'))}")


# UNQ_C9 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
# UNIT TEST COMMENT: Candidate for Table Driven Tests
# GRADED FUNCTION: edit_two_letters
def edit_two_letters(word, allow_switches=True):
    '''
    Input:
        word: the input string/word
    Output:
        edit_two_set: a set of strings with all possible two edits
    '''

    edit_two_set = set()

    ### START CODE HERE ###
    edit_one_list = list(edit_one_letter(word))
    edit_two_list = []

    for s in edit_one_list:
        edits = list(edit_one_letter(s))
        edit_two_list.extend(edits)

    edit_two_set = set(edit_two_list)

    ### END CODE HERE ###

    return edit_two_set

tmp_edit_two_set = edit_two_letters("a")
tmp_edit_two_l = sorted(list(tmp_edit_two_set))
print(f"Number of strings with edit distance of two: {len(tmp_edit_two_l)}")
print(f"First 10 strings {tmp_edit_two_l[:10]}")
print(f"Last 10 strings {tmp_edit_two_l[-10:]}")
print(f"The data type of the returned object should be a set {type(tmp_edit_two_set)}")
print(f"Number of strings that are 2 edit distances from 'at' is {len(edit_two_letters('at'))}")

# example of logical operation on lists or sets
print( [] and ["a","b"] )
print( [] or ["a","b"] )
#example of Short circuit behavior
val1 =  ["Most","Likely"] or ["Less","so"] or ["least","of","all"]  # selects first, does not evalute remainder
print(val1)
val2 =  [] or [] or ["least","of","all"] # continues evaluation until there is a non-empty list
print(val2)


# UNQ_C10 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
# UNIT TEST COMMENT: Candidate for Table Driven Tests
# GRADED FUNCTION: get_corrections
def get_corrections(word, probs, vocab, n=2, verbose=False):
    '''
    Input:
        word: a user entered string to check for suggestions
        probs: a dictionary that maps each word to its probability in the corpus
        vocab: a set containing all the vocabulary
        n: number of possible word corrections you want returned in the dictionary
    Output:
        n_best: a list of tuples with the most probable n corrected words and their probabilities.
    '''

    suggestions = []
    n_best = []

    ### START CODE HERE ###
    words_prob_dict = dict()

    # step 1
    # If the word is in the vocabulary, suggest the word
    if word in probs:
        suggestions = [word]
        n_best = [(word, probs[word])]
        return n_best

    # step 2, 3
    edit_one_set = edit_one_letter(word)
    edits = edit_one_set if len(edit_one_set) > 0 else edit_two_letters(word)

    if len(edits) == 0:
        edits = [word, ]

    for s in edits:
        words_prob_dict[s] = probs[s] if s in probs else 0

    counter_s = Counter(words_prob_dict)
    suggestions = [counter_s.most_common()[i][0] for i in range(n)]
    print(suggestions)

    for s in suggestions:
        tuple = (s, words_prob_dict[s])
        n_best.append(tuple)
    ### END CODE HERE ###

    if verbose: print("entered word = ", word, "\nsuggestions = ", suggestions)

    return n_best


# Test your implementation - feel free to try other words in my word
my_word = 'dys'
tmp_corrections = get_corrections(my_word, probs, vocab, 2, verbose=True) # keep verbose=True
for i, word_prob in enumerate(tmp_corrections):
    print(f"word {i}: {word_prob[0]}, probability {word_prob[1]:.6f}")

# CODE REVIEW COMMENT: using "tmp_corrections" insteads of "cors". "cors" is not defined
print(f"data type of corrections {type(tmp_corrections)}")


### PART IV
### Dynamic programming
# UNQ_C11 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
# GRADED FUNCTION: min_edit_distance
def min_edit_distance(source, target, ins_cost=1, del_cost=1, rep_cost=2):
    '''
    Input:
        source: a string corresponding to the string you are starting with
        target: a string corresponding to the string you want to end with
        ins_cost: an integer setting the insert cost
        del_cost: an integer setting the delete cost
        rep_cost: an integer setting the replace cost
    Output:
        D: a matrix of len(source)+1 by len(target)+1 containing minimum edit distances
        med: the minimum edit distance (med) required to convert the source string to the target
    '''
    # use deletion and insert cost as  1
    m = len(source)
    n = len(target)
    # initialize cost matrix with zeros and dimensions (m+1,n+1)
    D = np.zeros((m + 1, n + 1), dtype=int)

    ### START CODE HERE (Replace instances of 'None' with your code) ###

    # Fill in column 0, from row 1 to row m, both inclusive
    for row in range(0, m+1):  # Replace None with the proper range
        D[row, 0] = 0 if row==0 else D[row-1,0] + del_cost

    # Fill in row 0, for all columns from 1 to n, both inclusive
    for col in range(0, n+1):  # Replace None with the proper range
        D[0, col] = 0 if col == 0 else D[0, col-1] + ins_cost

    # Loop through row 1 to row m, both inclusive
    for row in range(1, m+1):

        # Loop through column 1 to column n, both inclusive
        for col in range(1, n+1):

            # Intialize r_cost to the 'replace' cost that is passed into this function
            r_cost = rep_cost

            # Check to see if source character at the previous row
            # matches the target character at the previous column,
            if source[row-1] == target[col-1]: #!!
                # Update the replacement cost to 0 if source and target are the same
                r_cost = 0

            # Update the cost at row, col based on previous entries in the cost matrix
            # Refer to the equation calculate for D[i,j] (the minimum of three calculated costs)
            D[row, col] = min(D[row-1, col] + ins_cost, D[row-1, col-1] + r_cost,
                              D[row, col-1] + del_cost)

    # Set the minimum edit distance with the cost found at row m, column n
    med = D[m, n]

    ### END CODE HERE ###
    return D, med

#DO NOT MODIFY THIS CELL
# testing your implementation
source =  'play'
target = 'stay'
matrix, min_edits = min_edit_distance(source, target)
print("minimum edits: ",min_edits, "\n")
idx = list('#' + source)
cols = list('#' + target)
df = pd.DataFrame(matrix, index=idx, columns= cols)
print(df)

#DO NOT MODIFY THIS CELL
# testing your implementation
source =  'eer'
target = 'near'
matrix, min_edits = min_edit_distance(source, target)
print("minimum edits: ",min_edits, "\n")
idx = list(source)
idx.insert(0, '#')
cols = list(target)
cols.insert(0, '#')
df = pd.DataFrame(matrix, index=idx, columns= cols)
print(df)