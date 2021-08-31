# -*- coding: utf-8 -*-
"""
Code adapted from Xu, Xie & Yao (2016)
https://github.com/kaituoxu/X-Punctuator/blob/fd8215defcb3508c517c182da853c367c576be8d/src/data/dataset.py#L21 
"""

import torch.utils.data as data
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SequentialSampler
from collections import OrderedDict
import end_start_tokens as tokens
import pickle

class PuncDataset(data.Dataset):
    def __init__(self, untokenized_sentences, comma_representation, token_type,
                 allowed_symbols, num_unique_words, sort=True):
        """
        untokenized_sentences:
            should be the raw sentences below threshold before
            any other preprocessing
        
        comma_representation:
            How is a comma represented in sentences? By "," or 0 ? 
            
        
        """
        self.untokenized_sentences = untokenized_sentences
        if sort:
            self.untokenized_sentences.sort(key=lambda x: len(x), reverse=True)

        
        # I need to remove commas and make target sequences in one go
        self.target_sequences = []
        X = []
        for sentence in self.untokenized_sentences:
            X.append([token for token in sentence if token != comma_representation])
            temp = []
            for i in range(len(sentence)):
                if sentence[i] == comma_representation:
                    continue 
                try:
                    if sentence[i+1] == comma_representation:
                        temp.append(1)
                    else:
                        temp.append(0)
                except:
                    temp.append(0)
            self.target_sequences.append(temp)
        
        # Creating vocabulary and tokenizing sentences
        if token_type == "class":
            vocabulary = create_vocabulary_class(self.untokenized_sentences,
                                                           allowed_symbols)
        else:
            #vocabulary = create_vocabulary_lemma(self.untokenized_sentences,
            #                                               allowed_symbols)
            if num_unique_words == 50000:
                with open("most_frequent_words_50k.pkl", "rb") as fp: 
                    vocabulary = pickle.load(fp)
            if num_unique_words == 8000:
                with open("most_frequent_words_8k.pkl", "rb") as fp: 
                    vocabulary = pickle.load(fp)
            if num_unique_words == 100000:
                with open("most_frequent_words_100k.pkl", "rb") as fp: 
                    vocabulary = pickle.load(fp)        
            if num_unique_words == 200000:
                with open("most_frequent_words_200k.pkl", "rb") as fp: 
                    vocabulary = pickle.load(fp)
                    
                    
            vocabulary.extend([tokens.END_TOKEN, tokens.START_TOKEN])
                    
        vocabulary.append('<UNK>')
        vocabulary = set(vocabulary)
        print("Size of vocabulary: {}".format(len(vocabulary)))
        self.vocabulary = sorted(vocabulary)
        self.dictionary = OrderedDict()
        for idx, w in enumerate(self.vocabulary):
            self.dictionary[w] = idx
        self.dict_keys = self.dictionary.keys()
        
        self.tokenized_sentences = []
        for sentence in X: #untokenized_sentences:
            tokenized = [self.dictionary[w] if w in self.dict_keys else 
                         self.dictionary[tokens.UNKNOWN] for w in sentence]
            self.tokenized_sentences.append(tokenized)
        
    def __len__(self):
        return len(self.tokenized_sentences)
    
    def __getitem__(self, index):
        input = torch.LongTensor(self.tokenized_sentences[index])
        label = torch.LongTensor(self.target_sequences[index])
        # adding this to be able to inspect the untokenized version of sentences with incorrect predictions
        untokenized = self.untokenized_sentences[index]
        return input, label, untokenized
        
    
class RandomBucketSampler(object):
    """Yields of mini-batch of indices, sequential within the batch, random between batches.
    
    I.e. it works like bucket, but it also supports random between batches.
    Helpful for minimizing padding while retaining randomness with variable length inputs.
    
    Args:
        data_source (Dataset): dataset to sample from.
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``
    """
    
    def __init__(self, data_source, batch_size, drop_last):
        self.sampler = SequentialSampler(data_source) # impl sequential within the batch
        self.batch_size = batch_size
        self.drop_last = drop_last
        # random_batches is a list of lists where each inner-most list is a series of indices
        self.random_batches = self._make_batches()
        
    def _make_batches(self):
        indices = [i for i in self.sampler]
        batches = [indices[i:i+self.batch_size]
                   for i in range(0, len(indices), self.batch_size)]
        if self.drop_last and len(self.sampler) % self.batch_size > 0:
            random_indices = torch.randperm(len(batches)-1).tolist() + [len(batches)-1]
        else:
            random_indices = torch.randperm(len(batches)).tolist()
        return [batches[i] for i in random_indices]
    
    def __iter__(self):
        for batch in self.random_batches: 
            yield batch
            
    def __len__(self):
        return len(self.random_batches) # return the number of batches
            
class TextAudioCollate(object):
    """Another way to implement collate_fn passed to DataLoader.
    Use class but not function because this is easier to pass some parameters.
    """
    def __init__(self):
        pass
    
    def __call__(self, batch, PAD=tokens.IGNORE_ID):
        """Process one mini-batch samples, such as sorting and padding.
        Args:
            batch: a list of (text sequence, audio feature sequence)
        Returns:
            input_padded_seqs
            label_padded_seqs
            lengths
        """
        # Sort a list by sequence length (descending order) to use in pack_padded_sequence
        batch.sort(key=lambda x: len(x[0]), reverse=True)
        # Separate inputs and labels
        input_seqs, label_seqs, _ = zip(*batch)
        # padding
        lengths = [len(seq) for seq in input_seqs]
        input_padded_seqs = torch.zeros(len(input_seqs), max(lengths)).long()
        label_padded_seqs = torch.zeros(len(input_seqs), max(lengths)).fill_(PAD).long()
        for i, (input, label) in enumerate(zip(input_seqs, label_seqs)):
            end = lengths[i]
            input_padded_seqs[i, :end] = input[:end]
            label_padded_seqs[i, :end] = label[:end]
        return input_padded_seqs, torch.IntTensor(lengths), label_padded_seqs

def build_data_loader(untokenized_sentences, comma_representation, token_type,
                      allowed_symbols, batch_size, num_unique_words, drop_last=False,
                      num_workers=0):
    dataset = PuncDataset(untokenized_sentences, comma_representation,
                          token_type, allowed_symbols, num_unique_words)
    
    batch_sampler = RandomBucketSampler(data_source=dataset,
                                        batch_size=batch_size,
                                        drop_last=drop_last)
    collate_fn = TextAudioCollate()
    data_loader = DataLoader(dataset, batch_sampler=batch_sampler,
                             collate_fn=collate_fn, num_workers=num_workers)
    return data_loader
    
        


def create_vocabulary_class(data, allowed_symbols):
    vocabulary = []
    for sentence in data:
        s = [w for w in sentence if w.isalnum() or w in allowed_symbols or "-" in w]
        s.insert(0, tokens.START_TOKEN)
        s.append(tokens.END_TOKEN)
        vocabulary.extend(s)
    return vocabulary


def create_vocabulary_lemma(data, allowed_symbols):
    vocabulary = []
    for sentence in data:
        s = [w for w in sentence if w.isalnum() or w in allowed_symbols]
        s.insert(0, tokens.START_TOKEN)
        s.append(tokens.END_TOKEN)
        vocabulary.extend(s)
    return vocabulary