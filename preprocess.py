import numpy as np
import sys
import json
import re 
import pickle


def one_hot_encoding_of_the_dataset(training_data,words_vocab):
    encoded_data = np.zeros((len(training_data),len(words_vocab)),dtype="bool")
    for i, (d1,d2) in enumerate(training_data):
        if d1 in words_vocab and d2 in words_vocab:
            encoded_data[i,words_vocab.index(d1)] = 1
            encoded_data[i,words_vocab.index(d2)] = 1
    return encoded_data

def prepare_training_dataset(entire_text, words_vocab):
    words_vocab = list(words_vocab)
    training_data = []
    for sentence in entire_text:
        if len(sentence) < 3:
            continue
        for i in range(len(sentence)-2):
            w1 = sentence[i]
            w2 = sentence[i+1]
            w3 = sentence[i+2]
            if w1 in words_vocab:
                w1 = words_vocab.index(w1)
                if w2 in words_vocab:
                    w2 = words_vocab.index(w2)
                    training_data.append([w1,w2])
        
                if w3 in words_vocab:
                    w3 = words_vocab.index(w3) 
                    training_data.append([w1,w3])
        if sentence[-2] in words_vocab and sentence[-1] in words_vocab:
            w1 = sentence[-2]
            w2 = sentence[-1]
            if w1 in word_vocab and w2 in word_vocab:
                w1 = words_vocab.index(w1)
                w2 = words_vocab.index(w2)
                training_data.append([w1,w2])

            # training_data.append([sentence[-2],sentence[-1]])   
    return training_data



def separate_into_sentences(entire_text, sentence_separator = "।"):
    sentences = entire_text.split(sentence_separator)
    return sentences

def separate_into_words(sentences_arr, word_separator = " "):
    words = []
    for sentence in sentences_arr:
        words.append(sentence.split(word_separator))
    return words

def get_word_vocab(words_arr):
    vocab = {}
    for words in words_arr:
        for word in words:
            if word not in vocab:
                vocab[word] = 1
            else:
                vocab[word] += 1
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
        Character vocab is a list or dictionery of characters that are allowed in the dataset.
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
            
def save_vocab_file(vocab,filepath):
    with open(filepath,"w",encoding="utf-8") as f:
        for word in vocab:
            f.write(word + "\n")

def save_dataset(dataset,filepath):
    with open(filepath,"wb") as f:
        pickle.dump(dataset,f)
  

def get_values_for_parameter(parameter_name,arguments):
    if parameter_name not in arguments:
        return None
    index = arguments.index(parameter_name)
    if index == len(arguments) - 1:
        return "__default__"
    if arguments[index + 1].startswith("-"):
        return "__default__"
    return arguments[index + 1]


if __name__ == "__main__":
    arguments = sys.argv[1:]
    
    input_file = get_values_for_parameter("-i",arguments)
    if input_file == None or input_file == "__default__":
        print(" No input file provided. Using default data.txt")
        input_file = "data.txt"
        
    output_file = get_values_for_parameter("-o",arguments)
    if output_file == None or output_file == "__default__":
        print(" No output file provided. Using default word_vocab.txt")
        output_file = "word_vocab.txt"
        
    vocab_file = get_values_for_parameter("-v",arguments)
    if vocab_file == "__default__":
        print(" No vocab file provided. Using default character_vocab.json")
        vocab = json.load(open("character_vocab.json","r",encoding="utf-8"))
    elif vocab_file == None:
        print(" No vocab being used to filter text ")
        vocab = None
    else:
        vocab = json.load(open(vocab_file,"r",encoding="utf-8"))
    
    
    
    
    entire_text = prepare_dataset(input_file,vocab)
    entire_text = filter_dataset(entire_text)
    word_vocab = get_word_vocab(entire_text).items()
    print(len(word_vocab))
    # print(len(word_vocab))
    word_vocab = {word for word,count in word_vocab if count > 500}
    print(len(word_vocab))
    
    save_vocab_file(word_vocab,output_file)
    
    dataset = prepare_training_dataset(entire_text,word_vocab)
    save_dataset(dataset,"dataset.pkl")
    
    
    
    # encoded_dataset = one_hot_encoding_of_the_dataset(dataset,list(word_vocab))
       
    # joblib.dump(encoded_dataset,"encoded_dataset.joblib")
        
    # print(word_vocab)
    # print(len(dataset))
    