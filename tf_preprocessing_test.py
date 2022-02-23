import json
from tensorflow.keras.preprocessing.text import Tokenizer, tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

data = [
  "I am working on an NLP problem.".split(' '),
  "I have downloaded premade embedding weights to use for an embedding layer.".split(' '),
  " I want to tokenize it using the same indices as my premade embedding layer.".split(' '),
]
tokenizer = Tokenizer(num_words=4000)
tokenizer.fit_on_texts(data)

texts_to_sequences = tokenizer.texts_to_sequences(data)

# 모델저장
tokenizer_json = tokenizer.to_json()
with open('tokenizer.json', 'w', encoding='utf-8') as f:
    f.write(json.dumps(tokenizer_json, ensure_ascii=False))

# 모델불러오기
with open('tokenizer.json') as f:
    data = json.load(f)
    tokenizer = tokenizer_from_json(data)

print(tokenizer.index_word)
'''
{
  1: 'i', 2: 'embedding', 3: 'an', 
  4: 'premade', 5: 'to', 6: 'layer.', 
  7: 'am', 8: 'working', 9: 'on', 
  10: 'nlp', 11: 'problem.', 12: 'have', 
  13: 'downloaded', 14: 'weights', 15: 'use',
  16: 'for', 17: '', 18: 'want',
  19: 'tokenize', 20: 'it', 21: 'using', 
  22: 'the', 23: 'same', 24: 'indices', 25: 'as', 26: 'my'
}
'''
print(tokenizer.word_index)
'''
{
  'i': 1, 'embedding': 2, 'an': 3, 
  'premade': 4, 'to': 5, 'layer.': 6, 
  'am': 7, 'working': 8, 'on': 9, 
  'nlp': 10, 'problem.': 11, 'have': 12, 
  'downloaded': 13, 'weights': 14, 'use': 15, 
  'for': 16, '': 17, 'want': 18, 
  'tokenize': 19, 'it': 20, 'using': 21, 
  'the': 22, 'same': 23, 'indices': 24, 'as': 25, 'my': 26
}
'''
print(texts_to_sequences)
'''
[
  [1, 7, 8, 9, 3, 10, 11], 
  [1, 12, 13, 4, 2, 14, 5, 15, 16, 3, 2, 6],
  [17, 1, 18, 5, 19, 20, 21, 22, 23, 24, 25, 26, 4, 2, 6]
]
'''

# maxlen만큼 빈 공간은 0으로 채운다
pad = pad_sequences(texts_to_sequences, padding='post', maxlen=70)
print(pad)
'''
[[ 1  7  8  9  3 10 11  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
   0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
   0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]

 [ 1 12 13  4  2 14  5 15 16  3  2  6  0  0  0  0  0  0  0  0  0  0  0  0
   0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
   0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]

 [17  1 18  5 19 20 21 22 23 24 25 26  4  2  6  0  0  0  0  0  0  0  0  0
   0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
   0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]]
'''

# 원-핫 인코딩
tag_size = len(tokenizer.word_index) + 1
category = to_categorical(pad, num_classes=tag_size)
print(category)

'''
[[[0. 1. 0. ... 0. 0. 0.]
  [0. 0. 0. ... 0. 0. 0.]
  [0. 0. 0. ... 0. 0. 0.]
  ...
  [1. 0. 0. ... 0. 0. 0.]
  [1. 0. 0. ... 0. 0. 0.]
  [1. 0. 0. ... 0. 0. 0.]]


 [[0. 1. 0. ... 0. 0. 0.]
  [0. 0. 0. ... 0. 0. 0.]
  [0. 0. 0. ... 0. 0. 0.]
  ...
  [1. 0. 0. ... 0. 0. 0.]
  [1. 0. 0. ... 0. 0. 0.]
  [1. 0. 0. ... 0. 0. 0.]]


 [[0. 0. 0. ... 0. 0. 0.]
  [0. 1. 0. ... 0. 0. 0.]
  [0. 0. 0. ... 0. 0. 0.]
  ...
  [1. 0. 0. ... 0. 0. 0.]
  [1. 0. 0. ... 0. 0. 0.]
  [1. 0. 0. ... 0. 0. 0.]]]
'''


# 원핫 인코딩 샘플
y_data = [2, 2, 2, 1, 1, 2, 0, 0, 0, 1, 0, 2]
y_data = to_categorical(y_data)
print(y_data) 
'''
[[0. 0. 1.]
 [0. 0. 1.]
 [0. 0. 1.]
 [0. 1. 0.]
 [0. 1. 0.]
 [0. 0. 1.]
 [1. 0. 0.]
 [1. 0. 0.]
 [1. 0. 0.]
 [0. 1. 0.]
 [1. 0. 0.]
 [0. 0. 1.]]
'''