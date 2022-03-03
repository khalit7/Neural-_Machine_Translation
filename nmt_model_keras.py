from keras.layers import Embedding,LSTM,Dropout,Dense,Layer
from keras import Model,Input
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam
import keras.backend as K
import collections
import numpy as np
import time
from nltk.translate.bleu_score import corpus_bleu


class LanguageDict():
  def __init__(self, sents):
    word_counter = collections.Counter(tok.lower() for sent in sents for tok in sent)

    self.vocab = []
    self.vocab.append('<pad>') #zero paddings
    self.vocab.append('<unk>')
    # add only words that appear at least 10 times in the corpus
    self.vocab.extend([t for t,c in word_counter.items() if c > 10])

    self.word2ids = {w:id for id, w in enumerate(self.vocab)}
    self.UNK = self.word2ids['<unk>']
    self.PAD = self.word2ids['<pad>']



def load_dataset(source_path,target_path, max_num_examples=30000):
  ''' This helper method reads from the source and target files to load max_num_examples 
  sentences, split them into train, development and testing and return relevant data.
  Inputs:
    source_path (string): the full path to the source data, SOURCE_PATH
    target_path (string): the full path to the target data, TARGET_PATH
  Returns:
    train_data (list): a list of 3 elements: source_words, target words, target word labels
    dev_data (list): a list of 2 elements - source words, target word labels
    test_data (list): a list of 2 elements - source words, target word labels
    source_dict (LanguageDict): a LanguageDict object for the source language, Vietnamese.
    target_dict (LanguageDict): a LanguageDict object for the target language, English.
  ''' 
  # source_lines/target lines are list of strings such that each string is a sentence in the
  # corresponding file. len(source/target_lines) <= max_num_examples
  source_lines = open(source_path).readlines()
  target_lines = open(target_path).readlines()
  assert len(source_lines) == len(target_lines)
  if max_num_examples > 0:
    max_num_examples = min(len(source_lines), max_num_examples)
    source_lines = source_lines[:max_num_examples]
    target_lines = target_lines[:max_num_examples]

  # strip trailing/leading whitespaces and tokenize each sentence. 
  source_sents = [[tok.lower() for tok in sent.strip().split(' ')] for sent in source_lines]
  target_sents = [[tok.lower() for tok in sent.strip().split(' ')] for sent in target_lines]
    # for the target sentences, add <start> and <end> tokens to each sentence 
  for sent in target_sents:
    sent.append('<end>')
    sent.insert(0,'<start>')

  # create the LanguageDict objects for each file
  source_lang_dict = LanguageDict(source_sents)
  target_lang_dict = LanguageDict(target_sents)


  # for the source sentences.
  # we'll use this to split into train/dev/test 
  unit = len(source_sents)//10
  # get the sents-as-ids for each sentence
  source_words = [[source_lang_dict.word2ids.get(tok,source_lang_dict.UNK) for tok in sent] for sent in source_sents]
  # 8 parts (80%) of the sentences go to the training data. pad upto maximum sentence length
  source_words_train = pad_sequences(source_words[:8*unit],padding='post')
  # 1 parts (10%) of the sentences go to the dev data. pad upto maximum sentence length
  source_words_dev = pad_sequences(source_words[8*unit:9*unit],padding='post')
  # 1 parts (10%) of the sentences go to the test data. pad upto maximum sentence length
  source_words_test = pad_sequences(source_words[9*unit:],padding='post')


  eos = target_lang_dict.word2ids['<end>']
  # for each sentence, get the word index for the tokens from <start> to up to but not including <end>,
  target_words = [[target_lang_dict.word2ids.get(tok,target_lang_dict.UNK) for tok in sent[:-1]] for sent in target_sents]
  # select the training set and pad the sentences
  target_words_train = pad_sequences(target_words[:8*unit],padding='post')
  # the label for each target word is the next word after it
  target_words_train_labels = [sent[1:]+[eos] for sent in target_words[:8*unit]]
  # pad the labels. Dim = [num_sents, max_sent_lenght]
  target_words_train_labels = pad_sequences(target_words_train_labels,padding='post')
  # expand dimensions Dim = [num_sents, max_sent_lenght, 1]. 
  target_words_train_labels = np.expand_dims(target_words_train_labels,axis=2)

  # get the labels for the dev and test data. No need for inputs here. no need to expand dimensions
  target_words_dev_labels = pad_sequences([sent[1:] + [eos] for sent in target_words[8 * unit:9 * unit]], padding='post')
  target_words_test_labels = pad_sequences([sent[1:] + [eos] for sent in target_words[9 * unit:]], padding='post')

  # we have our data.
  train_data = [source_words_train,target_words_train,target_words_train_labels]
  dev_data = [source_words_dev,target_words_dev_labels]
  test_data = [source_words_test,target_words_test_labels]

  return train_data,dev_data,test_data,source_lang_dict,target_lang_dict