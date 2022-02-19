import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import urllib.request
import requests as rq
import json

urllib.request.urlretrieve(
  "https://raw.githubusercontent.com/ukairia777/tensorflow-nlp-tutorial/main/12.%20Sequence%20Labeling/dataset/train.txt", 
  filename="./data/train_data_lstm.data"
)

urllib.request.urlretrieve(
  "https://github.com/naver/nlp-challenge/raw/master/missions/ner/data/train/train_data", 
  filename="./data/train_data_bert.data"
)

res = rq.get('https://search.shopping.naver.com/api/review?nvMid=24392533524&topicCode=design&reviewType=ALL&sort=QUALITY&isNeedAggregation=N&isApplyFilter=Y&page=1&pageSize=20')

with open("./data/naver_comments.json", "w") as json_file:
    json.dump(res.json(), json_file)