import numpy as np 
import pandas as pd
import random
import sys
import re


class model:
    def __init__(self,vocab_size : int, vector_size: int,encoded_data :int):
        self.w1 = np.random.rand(vocab_size,vector_size)
        self.w2 = np.random.rand(vector_size,vocab_size)
        
        
    def softmax(self,x):
        return np.exp(x) / np.sum(np.exp(x), axis=0)
    
    def forward(self,encoded_data):
        if encoded_data.shape[0] != self.w1.shape[0]:
            raise ValueError("The input shape is not correct")
        self.z1 = np.dot(encoded_data,self.w1)
        self.z2 = np.dot(self.z1,self.w2)
        self.z3 = self.softmax(self.z2)
        return self.z3
    
    def backward(self,encoded_data,outputs,lr):
        self.dz2 = outputs - encoded_data
        self.dw2 = np.dot(self.z1.T,self.dz2)
        self.dz1 = np.dot(self.dz2,self.w2.T)
        self.dw1 = np.dot(encoded_data.T,self.dz1)
        self.w1 -= lr * self.dw1
        self.w2 -= lr * self.dw2 
        
    def train_for_a_epoch(self,encoded_data,lr):
        outputs = self.forward(encoded_data)
        self.backward(encoded_data,outputs,lr)
     


# def encode_data(data,vocab_size):
#     encoded_data = np.zeros((vocab_size,1))
#     encoded_data[data] = 1
#     return encoded_data




def separate_into_sentences(entire_text, sentence_separator = "।"):
    sentences = entire_text.split(sentence_separator)
    return sentences

def separate_into_words(sentences_arr, word_separator = " "):
    words = []
    for sentence in sentences_arr:
        words.append(sentence.split(word_separator))
    return words

def get_word_vocab(words_arr):
    vocab = []
    for words in words_arr:
        for word in words:
            if word not in vocab:
                vocab.append(word)
    return vocab


#######################################

def remove_all_characters_except_vocab(text,character_vocab):
    new_text = ""
    for char in text:
        if char in character_vocab:
            new_text += char
    return new_text        


def filter_text(text):
    text = re.sub("\(.*?\)","",text)
    return text

def remove_consecutive_spaces(text):
    return re.sub(" +"," ",text)

#######################################


def prepare_dataset(filepath, character_vocab = None):
    ''' 
        Character vocab is a list of characters that are allowed in the dataset.
        None means no data is filtered.
    '''
    text = filter_text(open(filepath,encoding="utf-8").read())
    
    if character_vocab:
        text = remove_all_characters_except_vocab(text,character_vocab)
    text = remove_consecutive_spaces(text)
    sentences = separate_into_sentences(text)
    words = separate_into_words(sentences)
    
    return(words)
    
def filter_dataset(entire_text):
    new_text = []
    for sentence in entire_text:
        new_sent = []
        for word in sentence:
            if word != "":
                new_sent.append(word)
        if len(new_sent) > 0:
            new_text.append(new_sent)
    return new_text
            
    

def _get_values_for_parameter(parameter_name,arguments):
    if parameter_name not in arguments:
        raise ValueError("Parameter {} not provided".format(parameter_name))
    index = arguments.index(parameter_name)
    if index == len(arguments) - 1:
        raise ValueError("No value provided for parameter {}".format(parameter_name))
    if arguments[index + 1].startswith("-"):
        raise ValueError("No value provided for parameter {}".format(parameter_name))
    return arguments[index + 1]


if __name__ == "__main__":
    arguments = sys.argv[1:]
    
    # inp = _get_values_for_parameter("-i",arguments)
    # out = _get_values_for_parameter("-o",arguments)
    # print(inp,out)
    
    inp = "data.txt"
    
    entire_text = prepare_dataset(inp)
    entire_text = filter_dataset(entire_text)
    print(len(entire_text))
    
    