import json
from bs4 import BeautifulSoup

def parse(content, topics):
  soup = BeautifulSoup(content.replace('<br/>', ''), 'html.parser')
  rs = soup.find_all('em')

  tag = []
  cnt = 0

  for topic, text in zip(topics, rs):
    temp = []
    for idx, word in enumerate(text.text.split(' ')):
      temp.append((word, '%s-%s'%(idx == 0 and 'B' or 'I', topic['topicCode'])))
    tag.append(temp)
    text.replaceWith(' $$$$$$$$$$$$$_%d '%cnt)
    cnt += 1

  recovery = soup.text
  words = recovery.split(' ')

  rst = []

  for word in words:
    if word.startswith('$$$$$$$$$$$$$_'):
      idx = int(word.split('_')[1].split(' ')[0])
      rst += tag[idx]
    else:
      rst.append((word, 'O'))

  rst = [pack for pack in rst if pack[0]]
  
  # rst = [('놓고', 'I-definition'), ('보기', 'I-definition'), ('딱', 'I-definition'), ('좋습니다', 'I-definition'), ('.', 'O'), ('구입하길', 'O'), ('잘했습니다.', 'O')]
  return [item[0] for item in rst], [item[1] for item in rst]

if __name__ == '__main__':
  idx = 3

  reviews = json.load(open("./data/naver_comments.json"))['reviews']

  topics = reviews[idx]['topics']
  content = reviews[idx]['content']

  words, tag = parse(content, topics)

  print(words)
  print(tag)