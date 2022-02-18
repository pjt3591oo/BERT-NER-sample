import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import urllib.request

urllib.request.urlretrieve(
  "https://raw.githubusercontent.com/ukairia777/tensorflow-nlp-tutorial/main/12.%20Sequence%20Labeling/dataset/train.txt", 
  filename="train_data_lstm"
)

urllib.request.urlretrieve(
  "https://github.com/naver/nlp-challenge/raw/master/missions/ner/data/train/train_data", 
  filename="train_data_bert"
)