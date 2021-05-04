# What will we do here?

We are going to implement a method of Khmer word segmentation and Part-Of-Speech(POS) tagging introduced in a paper entitled 
```Joint Khmer Word Segmentation and Part-of-Speech Tagging Using Deep Learning```[[1]](https://arxiv.org/abs/2103.16801).

## Overview

- Task: Joint word segmentation and POS tagging
  - word segmentation: break a sentence into list of words
  - POS tagging: labelling part-of-speech to each word

- Why this task?
  - Computers do not understand our language!
  - Want to encode our language (syntactically, semantically) which can work with Computer!
  - Word is the most fundamental element to understand language (meaning)
  - So, need word segmentation process!!!
  - Part-of-speech helps in further understanding!!!
  - These tasks strongly effect other Natural Language Processing tasks (Machine Translation, Speech recognition...)


- Problem: where to seperate words in a sentence? What is the part-of-speech of those words?
  - In English, easily choose **white space** as a symbol of word seperation.
  - How about **Khmer**? →　No IDEA ???

- What people did to solve this/these task(s)?
  - In general, people solve these tasks seperately
  - Manually (human)
  - Conditional Random Field
  - Deep Learning: RNN, LSTM, etc.

- And what is introduced in [1] ?
  - Use Deep Learning approach
    - Bi-directional Long Short-Term Memory (BiLSTM)
  - Use character level input
  - Jointly predict where to seperate words and their pos at the same time
  - Read the [paper](https://arxiv.org/abs/2103.16801) for detail!!!

## Dataset
- We use khPOS dataset from https://github.com/ye-kyaw-thu/khPOS
- Training data: ```khPOS/corpus-draft-ver-1.0/data/after-replace/train.all2```
- Testing data: ```khPOS/corpus-draft-ver-1.0/data/CLOSE-TEST, OPEN-TEST```

