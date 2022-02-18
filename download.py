import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import re
import urllib.request

urllib.request.urlretrieve(
  "https://raw.githubusercontent.com/ukairia777/tensorflow-nlp-tutorial/main/12.%20Sequence%20Labeling/dataset/train.txt", 
  filename="train.txt"
)