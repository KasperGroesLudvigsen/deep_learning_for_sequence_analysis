# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 10:58:16 2021

@author: groes
"""
import pandas as pd



## Getting some descriptive stats
def get_lengths(dataset):
    """ Produce list containing the lengths of all the elements in the dataset
    """
    lengths = []
    for sentence in dataset:
        lengths.append(len(sentence))
    
    return lengths



def get_descriptive_stats(dataset):
    """ Returns df with descriptive stats of dataset"""
    lengths = get_lengths(dataset)
    df_lengths = pd.DataFrame(lengths)
    return df_lengths.describe()
    
def get_df_lengths(dataset):
    # dataset is e.g. sentences_lemma_90_incl
    lengths = get_lengths(dataset)
    return pd.DataFrame(lengths)
    
def percentage_above_threshold(dataset, threshold):
    df_lengths = get_df_lengths(dataset)
    above_threshold = df_lengths[df_lengths[0] > threshold]
    percentage_above_threshold = (len(above_threshold) / len(df_lengths)) * 100
    print("% of sentences removed because they exceed the {} threshold: {} %".format(
        threshold, percentage_above_threshold))
    
def percentage_in_range(dataset, interval):
    """ interval (tuple) : first element should be lower bound, second element 
    upper bound """
    df_lengths = get_df_lengths(dataset)
    above_threshold = df_lengths[df_lengths[0] > interval[1]]
    below_threshold = df_lengths[df_lengths[0] < interval[0]]
    total_length = len(above_threshold) + len(below_threshold)
    percentage_in_range = total_length / len(df_lengths) * 100
    print("% of sentences removed because they exceed the {} range: {} %".format(
        interval, percentage_in_range))

"""




################### EXPERIMENTS ########################
# How long is the longest sentence
longest = 0
idx = None
for i in sentences_lemma_90:
    if len(i) > longest:
        longest = len(i)
        idx = sentences_lemma_90.index(i)
        

# Measuring how many tokens are in korpus 90 after preprocessing, = 25,318,858
length = 0
for i in sentences_lemma_90:
    length += len(i)

# Checking that there are indeed no sentences without comma
counter = 0
for sentence in sentences_lemma_commas:
    if "," not in sentence:
        counter += 1

# Checking how many sentences are left with weird symbols. Its only 11
inspect_sentences_lemma_commas = prep.get_sentences_with_symbols(sentences_lemma_commas)

inspect = get_sentences_with_symbols(sentences_lemma)        
    

save_sentences(sentences_class, "sentences_class.txt")
save_sentences(sentences_lemma, "sentences_lemma.txt")

for i in unique_set:
    print(i)
    break

list_of_unique_lemmas = list(unique_set)
list_of_unique_lemmas.sort()

#sentences_class, sentences_lemma = extract_sentences_from_raw_txt()
           
            
a_set = set(sentences_lemma[0])
len(a_set)        
len(sentences_lemma[0])

# Saving sentence files
#utils.save_file("sentences_class.txt", sentences_class)
#utils.save_file("sentences_lemma.txt", sentences_lemma)

check = ["#", "&"]

test = ["test", "hej", "&amp;Mrs"]
for i in test:
    if "&" in i:
        print(i)

any(item in test for item in check)

    
#if i in check:
    #    print(i)

  
    if row[0] != "<":
        idx += 1
        sentence_class.append('\t'.join(row.split('\t')[-1:])[:2])
        sentence_lemma.append('\t'.join(row.split('\t')[-3:-2]))
        
            sentence_comma_indices.append(idx)
            
    else:
        #sentences.append(sentence)
        if len(sentence_comma_indices) > 0:
            sentence_class.append(sentence_comma_indices)
            sentence_lemma.append(sentence_comma_indices)
        sentence_class = []
        sentence_lemma = []
        sentence_comma_indices = []
    if len(sentence_class) > 0:
        sentences_class.append(sentence_class)
        sentences_lemma.append(sentence_lemma)
        
        
        
row = txt[1]

txt[1]

s = txt[25]
'\t'.join(s.split('\t')[-4:-2])[0]


s.split('\t')


idx = -1
dots = 0
for i in sentences_lemma_90_incl_trimmed:
    if "." in i:
        dots += 1
        
dots_c = 0
for i in sentences_class_90_incl_trimmed:
    if "LC" in i:
        dots_c += 1
        
        
sentences_class_commas, sentences_lemma_commas, sentences_ids_commas = prep.extract_sentences_from_raw_txt(skip_no_comma=True)

sentences_class_90, sentences_lemma_90, sentences_ids_90 = prep.extract_sentences_from_raw_txt(skip_no_comma=True, korpus="1990")

sentences_class_90_incl, sentences_lemma_90_incl, sentences_ids_90_incl \
    = prep.extract_sentences_from_raw_txt(skip_no_comma=False, korpus="1990")

lengths = get_lengths(sentences_lemma_90_incl)
avg_length = sum(lengths) / len(lengths)
print("Average length of elements in dataset is {}".format(avg_length))

df_lengths = pd.DataFrame(lengths)
df_lengths_descriptive = df_lengths.describe()
print(df_lengths_descriptive)
       
threshold = 45
percentage_above_threshold(df_lengths, threshold)


# What's the longest sequence in sentence_lemma_90_incl
longest = 0

total_length = 0
sentences = 0
for sentence in sentences_lemma_90_incl:
    sentences += 1
    length = len(sentence)
    total_length += length
    if length > longest:
        longest = length

avg_length = total_length/len(sentences_lemma_90_incl)



### Taking a look at a sentence with dot
idx = -1
for sentence in sentences_lemma_90_incl:
    idx += 1
    if "." in sentence:
        print(sentence)
        break
sentences_ids_90_incl[idx]



# Checking 
for i in sentences_as_strings_90:
    if 0 < len(i):
        print(i)

# Experimenting
dots = []
for row in txt:
    if row[:5] == "<s id":
        continue
    if row[:4] == '</s>':
        continue
    next_row = txt[txt.index(row)+1]
    if '\t'.join(row.split('\t')[-4:-2])[0] == "." and next_row[:4] != '</s>':
        dots.append(txt.index(row))

# There are dots in 1,700,000 sentences. Getting indices of which
idx = -1
dots = 0
dot_indices = []
for sentence in sentences_lemma_90_incl:
    idx += 1
    if "." in sentence:
        dot_indices.append(idx)
        dots += 1

# Checking sentences that are shorter than 3 elements
short_sentences = []
for sentence in sentences_lemma_90_incl:
    if len(sentence) < 4:
        short_sentences.append(sentence)
empty_elements = []
for sentence in sentences_lemma_90_incl:
    if len(sentence) == 0:
        empty_elements.append(sentence)
# len(empty_elements) = 1,710,117
        
        
sentences_with_dots = [sentences_lemma_90_incl[i] for i in dot_indices]

# There's something wrong here. Look at the partial_sentences. It seems they are not split as intended
partial_sentences = []
indices = []
idx = -1
for sentence in sentences_with_dots[:1000]:
    idx += 1
    if sentence[-1] != ".":
        indices.append(idx)
        # sentence is a list where each element is a word. Converting to one string
        sentence_as_string = " ".join(sentence)
        
        split_sentence = sentence_as_string.split(".")
        
        for i in split_sentence:
            lst = i.split(" ")
            if lst[-1] == "":
                lst = lst[:-1]
            if lst[0] == "":
                lst = lst[1:]
            partial_sentences.append(lst)
        
        #partial_sentences.extend(split_sentence)
        #previous_index = 0
        #for word in sentence:
        #    if word == ".":
        #        idx = sentence.index(word)
        #        partial_sentence = sentence[previous_index:idx] # does not include the dot itself
        #        partial_sentences.append(partial_sentence)
        #        previous_index = idx+1 # +1 in order to not include the dot



def split_by_dots(sentence):
    partial_sentence = sentence.split(".")
    return partial_sentence

split_sentence = split_by_dots("hej mit navn. Det er jim")
test = ["kasper kan"]  
test.extend(split_sentence)
test = ["hej", "mit", "navn", "det", "er"]
test2 = [0,1,2]
test[test2]

# Take a look at some sentences without commas
indices_no_comma = []
for sentence in sentences_lemma_90_incl[7:]:
    if "," not in sentence:

        idx = sentences_lemma_90_incl.index(sentence)
        print(idx)
        print(sentence)
        break
        #indices_no_comma.append(idx)

def print_no_comma_sentence(start_idx):
    for sentence in sentences_lemma_90_incl[start_idx:]:
        if "," not in sentence:

            idx = sentences_lemma_90_incl.index(sentence)
            print(idx)
            print(sentence)
            return idx

idx = print_no_comma_sentence(idx+1)

# How long is the longest sentence
longest = 0
idx = None
for i in sentences_lemma_90:
    if len(i) > longest:
        longest = len(i)
        idx = sentences_lemma_90.index(i)
        

# Measuring how many tokens are in korpus 90 after preprocessing, = 25,318,858
length = 0
for i in sentences_lemma_90:
    length += len(i)

# Checking that there are indeed no sentences without comma
counter = 0
for sentence in sentences_lemma_commas:
    if "," not in sentence:
        counter += 1

    
## Seems to not be working as expected
longest_seq_b4_comma = 0
idx = None
for sentence in sentences_lemma_90:
    words_before_comma = 0
    for token in sentence:
        print(token)
        if token == ",":
            if words_before_comma > longest_seq_b4_comma:
                longest_seq_b4_comma = words_before_comma
                idx = sentences_lemma_90.index(sentence)
            longest_seq_b4_comma = 0
            continue
        words_before_comma += 1

        
    break

# THis works
test = sentences_lemma_90[0]

longest_seq_b4_comma = 0
words_before_comma = 0
for i in test:
    if i == ",":
        print(i)
        if words_before_comma > longest_seq_b4_comma:
            longest_seq_b4_comma = words_before_comma
        words_before_comma = 0
        continue
    words_before_comma += 1
    break

# Finding length of longest sequence of tokens before a comma
longest_seq_b4_comma = 0
idx = 0
for sentence in sentences_lemma_90:
    
    words_before_comma = 0
    for i in sentence:
        if i == ",":
            #print(i)
            if words_before_comma > longest_seq_b4_comma:
                longest_seq_b4_comma = words_before_comma
                idx = sentences_lemma_90.index(sentence)
            words_before_comma = 0
            continue
        words_before_comma += 1
    
# Finding indices of sentences with more than x tokens before a comma
threshold = 50
idx = 0
indices = []
for sentence in sentences_lemma_90:
    
    words_before_comma = 0
    for i in sentence:
        if i == ",":
            #print(i)
            if words_before_comma > threshold:
                idx = sentences_lemma_90.index(sentence)
                indices.append(idx)
            words_before_comma = 0
            continue
        words_before_comma += 1

    
testint = 0
testint2 = 2
testint3 = testint
testint += 1
"""