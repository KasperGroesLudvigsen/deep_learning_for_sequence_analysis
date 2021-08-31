# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 21:07:23 2021

@author: groes
"""
from os import listdir
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
import dataexploration as dx
import dataset as dataclass
import end_start_tokens as tokens

def get_data(params):
    """
    Wrapper function for all functions that are related to making the 
    training, validation and test datasets.
    """
    untokenized_training_sentences, untokenized_validation_sentences, \
        untokenized_test_sentences = get_sentences(params)
        
    train_loader, valid_loader, test_loader = make_data_loaders(
        params,
        untokenized_training_sentences,
        untokenized_validation_sentences,
        untokenized_test_sentences
        )
    
    return train_loader, valid_loader, test_loader

def make_data_samplers(dataset, test_size, validation_size):
    """
    Generates PyTorch SubsetRandomSamplers from a dataset

    Parameters
    ----------
    dataset : TensorDataset
        TensorDataset generated with assemble_dataset()

    Returns
    -------
    train_sampler : SubsetRandomSamplers
        DESCRIPTION.
    validation_sampler : SubsetRandomSamplers
        DESCRIPTION.
    test_sampler : SubsetRandomSamplers
        DESCRIPTION.
    
    shuffle : BOOL
        If true, indices will be shuffled in make_indices()

    """
    
    # Making indices to be used in train, validation and test samplers
    training_idx, validation_idx, test_idx = make_indices(
        dataset, test_size, validation_size
        )
    
    # Making samplers
    train_sampler = SubsetRandomSampler(training_idx)
    validation_sampler = SubsetRandomSampler(validation_idx)
    test_sampler = SubsetRandomSampler(test_idx)
    
    return train_sampler, validation_sampler, test_sampler

def make_indices(dataset, test_size, validation_size, make_validation=True):
    """
    Defining this as stand alone function in order to make it testable.
    It creates the indices to be used in creating the SubsetRandomSamplers
    in make_data_samplers(). The indices are shuffled if shuffle=True

    dataset : TensorDataset
        TensorDataset generated with assemble_dataset()

    """
    num_observations = len(dataset)
    indices = list(range(num_observations))
    #if shuffle:
    #    np.random.shuffle(indices)
    #    print("Shuffling indices")
    np.random.shuffle(indices)
    test_train_split = int(np.floor(test_size * num_observations))
    training_idx, test_idx = indices[test_train_split:], indices[:test_train_split]
    validation_idx = None
    
    if make_validation:
        num_training_data = len(training_idx)
        train_validation_split = int(np.floor(validation_size * num_training_data))
        training_idx, validation_idx = training_idx[train_validation_split:], training_idx[:train_validation_split]
    
    return training_idx, validation_idx, test_idx


def tokenize_sentence(sentence, dictionary, dict_keys, max_seq_length):
    ind = [dictionary[w] if w in dict_keys else dictionary[tokens.UNKNOWN] for w in sentence]
    if len(ind) >= max_seq_length:
        ind = ind[:max_seq_length]
    else:
        ind.extend([dictionary[tokens.UNKNOWN]]*(max_seq_length-len(ind)))
    ind.insert(0, dictionary[tokens.START_TOKEN])
    ind.append(dictionary[tokens.END_TOKEN])
    return ind
    





def make_y_vectors(data, comma):
    """ Makes target dataset where the value of an element is 0 if a word is 
    not followed by a comma, 1 if it is. 
    
    comma : STR or INT
        How is a comma represented? Is it a string like "," or is the data 
        argument tokenized so that a comma is represented by en integer?
    """
    
    # trimmed (i.e. sentences longer than threshold left out) and 0 for no subsequent comma and 1 if yes
    y_vector = []
    for sentence in data:
        temp = []
        for i in range(len(sentence)):
            if sentence[i] == comma:
                continue 
            try:
                if sentence[i+1] == comma:
                    temp.append(1)
                else:
                    temp.append(0)
            except:
                temp.append(0)
        y_vector.append(temp)
    return y_vector

def generate_all_paths_90(folder="korpus90/KDK-1990.scrambled", save=False, test=False):
    """ Used for generating all paths to txt files in korpus 90 """
    
    if test:
        folder = "korpus90_test/KDK-1990.scrambled"
    file_names = listdir(folder)
    
    all_paths = []
    for filename in file_names:
        complete_path = folder + "/" + filename 
        all_paths.append(complete_path)
        
    if save:
        print("Saving all paths in txt file called 'paths' ")
        with open('paths_korpus90.txt', 'w') as f:
            for item in all_paths:
                f.write("%s\n" % item)
    
    return all_paths
  
  
def generate_all_paths(folder_containing_sub_directories="korpus2021/KDK-2010.scrambled/", save=False):
    """
    Used to generate paths to txt files in Korpus 2010

    Parameters
    ----------
    folder_containing_sub_directories : TYPE, optional
        DESCRIPTION. The default is "korpus2021/KDK-2010.scrambled/" which is 
        the folder in the Korpus2010 directory that contains the korpus text.
        The folder contains a number of sub-directories that each contain a 
        number of txt files that each contain a number of sentences.
    save : TYPE, optional
        DESCRIPTION. The default is False. If true, a txt file with all paths
        is saved in the root of the working directory. 

    Returns
    -------
    all_paths : LIST
        A list of strings where each string represent a path to a txt file

    """
    
    sub_directories = listdir(folder_containing_sub_directories)
    paths_to_all_sub_directories = []
    all_paths = []
    
    # Generating paths to all sub directories in KDK-2010.scrambled
    for sub_directory in sub_directories:
        if sub_directory == ".DS_Store":
            continue
        path = folder_containing_sub_directories + sub_directory
        paths_to_all_sub_directories.append(path)
        
    # Generating all paths
    for path in paths_to_all_sub_directories:
        file_names = listdir(path)    
        #file_paths = utils.get_path_to_files(path)
        for file_name in file_names:
            if file_name == ".DS_Store":
                continue
            complete_path = path + "/" + file_name
            all_paths.append(complete_path)
    
    if save:
        print("Saving all paths in txt file called 'paths' ")
        with open('paths.txt', 'w') as f:
            for item in all_paths:
                f.write("%s\n" % item)
        
    return all_paths


def get_sentences(params):
    """
    Wrapper method for extract sentences. Extracts all sentences from raw text
    and perforsm some pre-processing steps on it. The output of this method
    can be used to create dataloaders. 
    """
    
    if params["token_type"] == "class":
        sentences, _, _ = extract_sentences_from_raw_txt(
            skip_no_comma=params["skip_no_comma"], korpus=params["corpus"])
        
    elif params["token_type"] == "lemma":
        _, sentences, _ = extract_sentences_from_raw_txt(
            skip_no_comma=params["skip_no_comma"], korpus=params["corpus"])
        
    else:
        raise ValueError("Unknown token type")
    
    # Show how how much of the dataset will be discarded due to interval 
    dx.percentage_in_range(sentences, params["interval"])
    
    # Creating list of sentences with length below threshold to be used in build_data_loader()
    # Removing sentences rather than capping sentences in order to retain meaningful
    # sequences of tokens
    sentences = [i for i in sentences if len(i) <= params["interval"][1] and \
                 len(i) >= params["interval"][0]] 
    
    # Splitting into train, validation and test and making data loaders
    train_idx, val_idx, test_idx = make_indices(sentences, params["test_size"],
                                                     params["validation_size"])
    untokenized_training_sentences = [sentences[idx] for idx in train_idx]
    untokenized_validation_sentences = [sentences[idx] for idx in val_idx]
    untokenized_test_sentences = [sentences[idx] for idx in test_idx]
    
    return untokenized_training_sentences, untokenized_validation_sentences, \
        untokenized_test_sentences 

def make_data_loaders(params,
                      untokenized_training_sentences,
                      untokenized_validation_sentences,
                      untokenized_test_sentences):
    """
    Make all three data laoders (test, valid, train)
    """
    
    print("Creating data loaders")
    
    if params["token_type"] == "class":
        allowed_symbols = [".", "-"] # some word class names contain a dash
    elif params["token_type"] == "lemma":
        allowed_symbols = ["."]
    else:
        raise ValueError("Unknown token type")
        
    train_loader = dataclass.build_data_loader(
        untokenized_sentences=untokenized_training_sentences,
        comma_representation=params["comma_representation"], 
        token_type=params["token_type"],
        allowed_symbols=allowed_symbols, 
        batch_size=params["batchsize"], 
        num_unique_words=params["num_unique_words"], 
        drop_last=True)
    
    valid_loader = dataclass.build_data_loader(
        untokenized_sentences=untokenized_validation_sentences,
        comma_representation=params["comma_representation"],
        token_type=params["token_type"],
        allowed_symbols=allowed_symbols,
        batch_size=params["batchsize"],
        num_unique_words=params["num_unique_words"],
        drop_last=True)

    test_loader = dataclass.PuncDataset(
        untokenized_test_sentences,
        comma_representation=params["comma_representation"],
        token_type=params["token_type"],
        allowed_symbols=allowed_symbols,
        num_unique_words=params["num_unique_words"],
        )

    return train_loader, valid_loader, test_loader

def extract_sentences_from_raw_txt(skip_no_comma, korpus):
    """
    Goes through all txt files in Korpus 2010 dataset and extracts the individual
    sentences from the dataset. 
    
    Parameters
    skip_no_comma : BOOL
        If true, sentences that do not contain commas are not added to the dataset
    
    korpus : STRING
        String specifying which korpus to use. Should be "1990" or "2010"
    
    Returns
    -------
    sentences_class : TYPE
        DESCRIPTION.
    sentences_lemma : TYPE
        DESCRIPTION.

    """
    print("Extracting sentences from raw data in korpus {}".format(korpus))
    
    if korpus == "1990":
        all_paths = generate_all_paths_90()
    elif korpus == "2010":
        all_paths = generate_all_paths()
    elif korpus == "test":
        all_paths = generate_all_paths_90(test=True)
    else:
        raise ValueError("Invalid korpus specified")
    
    # List to hold all ids. It should have the same length as sentences_class or 
    # sentences_lemma
    sentences_ids = []
    
    # List where each element is a list that contains the word classes of the
    # words in a sentence
    sentences_class = []
    
    # List where each element is a list that contains the word lemmas of the
    # words in a sentence
    sentences_lemma = []
    
    # holds all lemmas of a sentence as string instead of list
    #list_of_strings = [] 

    # Lists holding tokens of a sentence until added to sentences_lemma or 
    # sentences_class
    sentence_lemma = []
    sentence_class = []
    #sentence_as_string = ""
    # If any of these symbols are present in the sentence, it will not be 
    # included in final dataset
    check_for = ["#", "&", "$", "=", "¤", "{", "}", "[", "]", "/"]
    
    skipped_because_no_comma = 0
    
    for path in all_paths:
        with open(path, encoding="utf8") as f:
            txt = f.readlines()
            f.close
            
        # Looping over rows in txt. Each row represents a word in a sentence.
        # Row is a list with a number of elements that 
        # are associated with a word (original word, lemma, class etc.) in a sentence in
        # a txt file
        # each row looks something like this: 'af\taf\t_\taf\tT\tT-:----:--:----\n'
        for row in txt: 
            
            # Upon manual inspection of sentences, I've noticed some sentences
            # that do not make sense start with "iwill"
            if row[:5] == "iwill" or '\t'.join(row.split('\t')[3:4]) == "iwill":
                continue
            
            # When a start tag is encountered, get the sentence id and go to next row
            if row[:5] == "<s id":
                sentence_id = '"'.join(row.split('"')[1:2])
                continue
            
            # If the item is an end tag
            if row[:4] == '</s>':
                # Do not append the sentence if it does not contain a comma
                if skip_no_comma and "," not in sentence_lemma:
                    sentence_class = []
                    sentence_lemma = []
                    #sentence_as_string = ""
                    skipped_because_no_comma += 1
                    continue
                sentences_class.append(sentence_class)
                sentences_lemma.append(sentence_lemma)
                sentences_ids.append(sentence_id)
                #list_of_strings.append(sentence_as_string)
                sentence_class = []
                sentence_lemma = []
                #sentence_as_string = ""
                continue
            
            # first word in the row. In 'af\taf\t_\taf\tT\tT-:----:--:----\n' it would be 'af'
            first_item = '\t'.join(row.split('\t')[:1]) 
            
            # Skip if the word contains unwanted symbols
            if any(item in first_item for item in check_for):
                continue
            
            # Appending lemma of word and word class to the lists
            sentence_class.append('\t'.join(row.split('\t')[-1:])[:2])
            lemma = '\t'.join(row.split('\t')[-3:-2])
            sentence_lemma.append(lemma)
            #sentence_as_string += lemma + " "
            
            # Adding a comma to the lists if row shows that the word is followed by one
            if is_word_followed_by_comma(row): 
                sentence_class.append(",")
                sentence_lemma.append(",")

    if skip_no_comma:
        print("{} sentences were skipped because they did not contain commas".format(skipped_because_no_comma))
    
    return sentences_class, sentences_lemma, sentences_ids#, list_of_strings

def is_word_followed_by_comma(row):
    """ Check if the word is followed by a comma. 'row' refers to a row in the 
    raw txt files and the row contains the word in a text piece, and variations 
    of it including its lemma and its word class """
    if '\t'.join(row.split('\t')[-4:-2])[0] == ",":
        return True
    return False


    
         
def load_sentence_files():
    with open("sentences_class", encoding="utf8") as f:
        sentences_class = f.read().splitlines()
    with open("sentences_lemma") as f:
        sentences_lemma = f.read().splitlines()
    return sentences_class, sentences_lemma

def count_commas(sentences_class):
    counter = 0
    for sentence in sentences_class:
        if "," in sentence:
            counter += 1
    print(counter)
    
def count_unique_tokens(sentences):
    unique_tokens = []
    counter = 0
    for sentence in sentences:
        for word in sentence:
            counter += 1
            if word not in unique_tokens:
                unique_tokens.append(word)
        if counter % 10000 == 0:
            print(counter)
    print("Number of unique tokens is: {}".format(len(unique_tokens)))

def save_sentences(sentences, filename):
    with open(filename, 'w') as f:
        for item in sentences:
            f.write("%s\n" % item)
    print("JOB'S DONE!")


def get_sentences_with_symbols(sentences):
    counter = 0
    check_for = ["#", "&", "$", "=", "¤"]
    inspect = []
    for sentence in sentences:
        counter += 1
        if counter % 100000 == 0:
            print(counter)
        for word in sentence:
            if any(item in word for item in check_for):
                inspect.append(sentence)
                continue
    return inspect

"""
def make_data_loaders(dataset, train_sampler, validation_sampler, test_sampler, batchsize):
    """
"""
    Generates the dataloaders over which to loop when training and testing the
    models. Use:
        
        for i, (label, target) in enumerate(dataloader):
            ...

    Parameters
    ----------
    dataset : TensorDataset
        TensorDataset generated with assemble_dataset().
        
    train_sampler : SubsetRandomSamplers
        SubsetRandomSamplers object made with make_data_samplers().
        
    validation_sampler : SubsetRandomSamplers
        SubsetRandomSamplers object made with make_data_samplers().
        
    test_sampler : SubsetRandomSamplers
        SubsetRandomSamplers object made with make_data_samplers().
        
    batchsize : INT

    Returns
    -------
    train_loader : DataLoader
        DESCRIPTION.
    validation_loader : DataLoader
        DESCRIPTION.
    test_loader : DataLoader
        DESCRIPTION.
"""
"""
    train_loader = DataLoader(dataset=dataset,
                              batch_size=batchsize,
                              sampler=train_sampler)
    
    validation_loader = DataLoader(dataset=dataset,
                                   batch_size=batchsize,
                                   sampler=validation_sampler)
    
    test_loader = DataLoader(dataset=dataset,
                             batch_size=batchsize,
                             sampler=test_sampler)

    return train_loader, validation_loader, test_loader"""