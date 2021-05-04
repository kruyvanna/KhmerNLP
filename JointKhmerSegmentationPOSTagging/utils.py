import torch
import numpy as np
from sklearn.utils import shuffle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from torch.utils.data import Dataset, DataLoader

class Tokenizer(object):
  def __init__(self, trainfilepath, testfilepath=None):
    self.train_filepath = trainfilepath
    #self.test_filepath = testfilepath
    self.chDict = dict()
    self.posDict = dict()
    self.REVISED_POS  = {
      'M'    : 'NN' ,
      'RPN'  : 'NN' ,
      'CUR'  : 'SYM',
      'DBL'  : 'SYM',
      'ETC'  : 'SYM',
      'KAN'  : 'SYM',
      'UH'   : 'PA' ,
      'VB_JJ': 'VB' ,
      'VCOM' : 'VB'
    }
    
    self.train_data = self.process_data(self.train_filepath, 'train')
    #self.test_data = self.process_data(self.test_filepath, 'test')
    self.id2ch = {j:i for i,j in self.chDict.items()}
    self.id2pos = {j:i for i,j in self.posDict.items()}
  
  def encode(self, sequence, seq_type='ch'):
    idDict = self.chDict if seq_type == 'ch' else self.posDict
    ids = [idDict[i] if i in idDict else idDict['<UNK>'] for i in sequence]
    return ids

  def decode(self, tag_seq, character_seq):
    tmp, seq, pos = [], [], []
    for i,p in enumerate(tag_seq):
      if p == self.posDict['NS']:
        tmp.append(self.id2ch[character_seq[i]])
      else:
        pos.append(self.id2pos[p])
        if len(tmp) > 0:
          seq.append(''.join(tmp))
          tmp = []
        tmp.append(self.id2ch[character_seq[i]])
    if len(tmp) > 0:
      seq.append(''.join(tmp))
    return seq, pos
  
  def update_dictionary(self, word, pos):
    if pos not in self.posDict:    # update part-of-speech dictionary
      self.posDict[pos] = len(self.posDict)+1
    for c in list(word):
      if c not in self.chDict:     # update character dictionary
        self.chDict[c] = len(self.chDict)+1

  def process_data(self, filepath, data_type):
    sentences_word, sentences_pos = [], []
    word_tmp, pos_tmp = [], []

    with open(filepath, "r") as fin:
      for line in fin:
        items = line.split(" ")
        for word_pos in items:
          word, pos = word_pos.split('/')
          pos = pos.replace('\n','')
          if pos in self.REVISED_POS:
            pos = self.REVISED_POS[pos]
        
          if data_type == 'train':
            self.update_dictionary(word, pos)
          
          word_tmp += list(word)
          pos_tmp += [pos]
          pos_tmp += ['NS' for i in range(len(list(word))-1)]

        sentences_word.append(word_tmp)
        sentences_pos.append(pos_tmp)
        word_tmp = []; pos_tmp = []
      
    if data_type == 'train':
      self.posDict['NS'] =  len(self.posDict)+1
      self.chDict['<UNK>'] = len(self.chDict)+1
      self.chDict['<PAD>'] = 0
      self.posDict['<PAD>'] = 0
    
    return sentences_word, sentences_pos

class khPOSdataset(Dataset):
  def __init__(self, word_s, pos_s):
    self.word = word_s
    self.pos = pos_s

  def __len__(self):
    return len(self.pos)
  
  def __getitem__(self, idx):
    word_s = self.word[idx]
    pos_s = self.pos[idx]
    return (torch.tensor(word_s, dtype=torch.int64),
            torch.tensor(pos_s, dtype=torch.int64)
    )

class DataLoader(object):
    def __init__(self, dataset,
                 batch_size=100,
                 shuffle=False,
                 batch_first=False,
                 device='cpu',
                 random_state=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.batch_first = batch_first
        self.device = device

        if random_state is None:
            random_state = np.random.RandomState(123)

        self.random_state = random_state
        self._idx = 0
        self._reset()

    def __len__(self):
        N = len(self.dataset)
        b = self.batch_size
        return N // b + bool(N % b)

    def __iter__(self):
        return self

    def __next__(self):
        if self._idx >= len(self.dataset):
            self._reset()
            raise StopIteration()

        x, t = zip(*self.dataset[self._idx:(self._idx + self.batch_size)])
        x = pad_sequences(x, padding='post')
        t = pad_sequences(t, padding='post')

        x = torch.LongTensor(x)
        t = torch.LongTensor(t)

        if not self.batch_first:
            x = x.t()
            t = t.t()

        self._idx += self.batch_size

        return x.to(self.device), t.to(self.device)

    def _reset(self):
        if self.shuffle:
            self.dataset = shuffle(self.dataset,random_state=self.random_state)
        self._idx = 0
