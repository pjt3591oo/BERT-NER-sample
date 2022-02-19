'''
샘플 데이터 다운로드
wget https://github.com/naver/nlp-challenge/raw/master/missions/ner/data/train/train_data
'''
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from transformers import *
import tensorflow as tf
from tqdm import tqdm
import pandas as pd
import json, os, re
import numpy as np
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report

tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
# print(tokenizer.tokenize("대한민국 만세."))

def get_data() :
  train = pd.read_csv("train_data_bert.data", names=['src', 'tar'], sep="\t")
  train = train.reset_index()

  train['src'] = train['src'].str.replace("．", ".", regex=False)

  train['src'] = train['src'].astype(str)
  train['tar'] = train['tar'].astype(str)
  train['src'] = train['src'].str.replace(r'[^ㄱ-ㅣ가-힣0-9a-zA-Z.]+', "", regex=True)

  data = [list(x) for x in train[['index', 'src', 'tar']].to_numpy()]

  label = train['tar'].unique().tolist()
  label_dict = {word:i for i, word in enumerate(label)}
  label_dict.update({"[PAD]":len(label_dict)})
  index_to_ner = {i:j for j, i in label_dict.items()}

  tups = []
  temp_tup = []
  temp_tup.append(data[0][1:])
  sentences = []
  targets = []
  
  for i, j, k in data:
    
    if i != 1:
      temp_tup.append([j,label_dict[k]])
    if i == 1:
      if len(temp_tup) != 0:
        tups.append(temp_tup)
        temp_tup = []
        temp_tup.append([j,label_dict[k]])
  tups.pop(0)

  sentences = []
  targets = []
  for tup in tups:
    sentence = []
    target = []
    sentence.append("[CLS]")
    target.append(label_dict['-'])
    for i, j in tup:
      sentence.append(i)
      target.append(j)
    sentence.append("[SEP]")
    target.append(label_dict['-'])
    sentences.append(sentence)
    targets.append(target)

  return sentences, targets, label_dict, index_to_ner

def tokenize_and_preserve_labels(sentence, text_labels):
  tokenized_sentence = []
  labels = []

  for word, label in zip(sentence, text_labels):

    tokenized_word = tokenizer.tokenize(word)
    n_subwords = len(tokenized_word)

    tokenized_sentence.extend(tokenized_word)
    labels.extend([label] * n_subwords)

  return tokenized_sentence, labels

def ner_inference(test_sentence, nr_model, index_to_ner):
  max_len = 88
  tokenized_sentence = np.array([tokenizer.encode(test_sentence, max_length=max_len, truncation=True, padding='max_length')])
  tokenized_mask = np.array([[int(x!=1) for x in tokenized_sentence[0].tolist()]])
  ans = nr_model.predict([tokenized_sentence, tokenized_mask])
  ans = np.argmax(ans, axis=2)

  tokens = tokenizer.convert_ids_to_tokens(tokenized_sentence[0])
  new_tokens, new_labels = [], []
  for token, label_idx in zip(tokens, ans[0]):
    if (token.startswith("##")):
      new_labels.append(index_to_ner[label_idx])
      new_tokens.append(token[2:])
    elif (token=='[CLS]'):
      pass
    elif (token=='[SEP]'):
      pass
    elif (token=='[PAD]'):
      pass
    elif (token != '[CLS]' or token != '[SEP]'):
      new_tokens.append(token)
      new_labels.append(index_to_ner[label_idx])

  for token, label in zip(new_tokens, new_labels):
      print("{}\t{}".format(label, token))

def create_model():
  max_len = 88
  SEQ_LEN = max_len
  model = TFBertModel.from_pretrained("bert-base-multilingual-cased", from_pt=True, num_labels=len(label_dict), output_attentions = False,
    output_hidden_states = False)

  token_inputs = tf.keras.layers.Input((SEQ_LEN,), dtype=tf.int32, name='input_word_ids') # 토큰 인풋
  mask_inputs = tf.keras.layers.Input((SEQ_LEN,), dtype=tf.int32, name='input_masks') # 마스크 인풋

  bert_outputs = model([token_inputs, mask_inputs])
  bert_outputs = bert_outputs[0] # shape : (Batch_size, max_len, 30(개체의 총 개수))
  nr = tf.keras.layers.Dense(30, activation='softmax')(bert_outputs) # shape : (Batch_size, max_len, 30)
  
  nr_model = tf.keras.Model([token_inputs, mask_inputs], nr)
  
  nr_model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.00002), loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
      metrics=['sparse_categorical_accuracy'])
  nr_model.summary()


  return nr_model

sentences, targets, label_dict, index_to_ner = get_data()

print(label_dict)
print(index_to_ner)

# bert 인풋 만들기
tokenized_texts_and_labels = [tokenize_and_preserve_labels(sent, labs) for sent, labs in zip(sentences, targets)]

tokenized_texts = [token_label_pair[0] for token_label_pair in tokenized_texts_and_labels]
labels = [token_label_pair[1] for token_label_pair in tokenized_texts_and_labels]

max_len = 88
bs = 32

input_ids = pad_sequences(
  [tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],
  maxlen=max_len, 
  dtype = "int", 
  value=tokenizer.convert_tokens_to_ids("[PAD]"), 
  truncating="post", 
  padding="post"
)

tags = pad_sequences(
  [lab for lab in labels], 
  maxlen=max_len, 
  value=label_dict["[PAD]"], 
  padding='post',
  dtype='int', 
  truncating='post'
)

attention_masks = np.array([[int(i != tokenizer.convert_tokens_to_ids("[PAD]")) for i in ii] for ii in input_ids])

tr_inputs, val_inputs, tr_tags, val_tags = train_test_split(input_ids, tags, random_state=2018, test_size=0.1)
tr_masks, val_masks, _, _ = train_test_split(attention_masks, input_ids, random_state=2018, test_size=0.1)

# # 모델 만들기
nr_model = create_model()
print('before fit')
nr_model.fit([tr_inputs, tr_masks], tr_tags, validation_data=([val_inputs, val_masks], val_tags), epochs=3, shuffle=False, batch_size=bs)

try :
  nr_model.save('bert-ner.h5')
  print('모덜 저장 완료')
except:
  pass

# # 실제 데이터 테스트
ner_inference("문재인 대통령은 1953년 1월 24일 경상남도 거제시에서 아버지 문용형과 어머니 강한옥 사이에서 둘째(장남)로 태어났다.", nr_model, index_to_ner)

# # 정확도 테스트
# y_predicted = nr_model.predict([val_inputs, val_masks])
# f_label = [i for i, j in label_dict.items()]
# val_tags_l = [index_to_ner[x] for x in np.ravel(val_tags).astype(int).tolist()]
# y_predicted_l = [index_to_ner[x] for x in np.ravel(np.argmax(y_predicted, axis=2)).astype(int).tolist()]
# f_label.remove("[PAD]")
