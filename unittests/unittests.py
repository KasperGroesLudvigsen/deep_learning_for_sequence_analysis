# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 11:20:18 2021

@author: groes
"""
import preprocessing as prep
import os 
from collections import OrderedDict


def unittest_is_word_followed_by_comma():
    path_parent = os.path.dirname(os.getcwd())
    os.chdir(path_parent)
    path = "korpus90/KDK-1990.scrambled/0000.txt"
    with open(path, encoding="utf8") as f:
        txt = f.readlines()
        f.close
    row = txt[13]
    row2 = txt[2]
    assert prep.is_word_followed_by_comma(row)
    assert not prep.is_word_followed_by_comma(row2)
    
unittest_is_word_followed_by_comma()

def unittest_make_y_vectors():
    data = [["hello", ",", "my", "name", "is", "kasper"],
            ["my", "name", "jeff"],
            ["hello"], 
            ["test", "test", "test", ",", "test", "test"]]
    
    y = prep.make_y_vectors(data, ",")
    
    assert y[0][0] == 1
    for i in y[0][1:]:
        assert i == 0
    for i in y[1]:
        assert i == 0
    assert y[3][2] == 1
    assert len(y[2]) == 1
    assert len(y[0]) == len(data[0])
    assert len(y[3]) == len(data[3])
    
    data = [[1,2,3,4,0,1], [1,0,1,1,1,0]]
    y_test = prep.make_y_vectors(data, 0)
    assert y_test[0][3] == 1 # third index should be 1, which represents that a comma is in the 4 index position of the input
    assert y_test[0][0] == 0
    assert y_test[0][1] == 0
    assert y_test[0][2] == 0
    assert y_test[1][0] == 1
    assert y_test[1][4] == 1
    assert y_test[1][0] == 1
    assert y_test[1][1] == 0
    assert y_test[1][2] == 0
    assert y_test[1][3] == 0
       
unittest_make_y_vectors()


def unittest_tokenize_sentence():
    sentence = ["VF", "NP", "T-", "PI", ",", "NC", "T-", "NW"]
    sentence_for_dict = sentence.copy()
    sentence_for_dict.append("<UNK>")
    sentence_for_dict.append("<S>")
    sentence_for_dict.append("</S>")
    
    ordered_dict = OrderedDict()
    for idx, w in enumerate(sorted(sentence_for_dict)):
        ordered_dict[w] = idx
        
    dict_keys = ordered_dict.keys()
    max_seq_len = 10
    tokenized = prep.tokenize_sentence(sentence, ordered_dict, dict_keys, max_seq_len)
    
    assert len(tokenized) == max_seq_len + 2 # because of the addition of start and end tokens
    assert tokenized[0] == 2
    assert tokenized[1] == 10
    assert tokenized[-2] == 3 # 3 = unkown
    assert tokenized[-1] == 1
    assert 1 in tokenized # 1 = end token
    assert 2 in tokenized # 2 = end token
    
    max_seq_len = 15
    tokenized = prep.tokenize_sentence(sentence, ordered_dict, dict_keys, max_seq_len)
    assert tokenized[-2] == 3
    assert tokenized[-3] == 3
    assert tokenized[-4] == 3
    assert tokenized[-5] == 3
    assert tokenized[-6] == 3
    assert tokenized[-7] == 3
    assert tokenized[-8] == 3
    
    max_seq_len = 5
    tokenized = prep.tokenize_sentence(sentence, ordered_dict, dict_keys, max_seq_len)
    
    assert len(tokenized) == max_seq_len + 2
    assert 0 in tokenized
    assert 3 not in tokenized 
    assert 1 in tokenized
    assert 2 in tokenized
    
    max_seq_len = 4
    tokenized = prep.tokenize_sentence(sentence, ordered_dict, dict_keys, max_seq_len)
    assert 0 not in tokenized
    assert 1 in tokenized
    assert 2 in tokenized
    
    sentence = ["VF", "NP", "T-", "PI", ",", "NC", "T-", "NW", "test"]
    tokenized = prep.tokenize_sentence(sentence, ordered_dict, dict_keys, 9)
    assert len(tokenized) == 11
    assert tokenized[-2] == 3
    
unittest_tokenize_sentence()

def unittest_make_indices():
    datasize = 10000
    testsize = 0.2
    val_size = 0.25
    data = list(range(datasize))
    train_idx, val_idx, test_idx = prep.make_indices(data, testsize, val_size)
    
    assert len(train_idx) == 6000
    assert len(val_idx) == 2000
    assert len(test_idx) == 2000
    assert len(train_idx) + len(test_idx) + len(val_idx) == datasize
    
    for i in train_idx:
        assert i not in val_idx
        assert i not in test_idx
    for i in val_idx:
        assert i not in test_idx

unittest_make_indices()




















