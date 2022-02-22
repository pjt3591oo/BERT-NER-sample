import requests as rq
from bs4 import BeautifulSoup
from urllib.parse import urlsplit, parse_qsl

def get_product_id():
  url = 'https://search.shopping.naver.com/search/category/100000029?catId=50000807&origQuery&pagingIndex=3&pagingSize=40&productSet=total&query&sort=rel&timestamp=&viewType=list'
  res = rq.get(url)
  soup = BeautifulSoup(res.text, 'html.parser')
  links = soup.select('a')
  
  product_ids = []

  for link in links:
    if link.get('href').find('https://cr.shopping.naver.com/adcr.nhn') == 0:
      data = dict(parse_qsl(link.get('href')))
      print(link.text)
      product_ids.append(data.get('nvMid'))
  
  return product_ids

products = get_product_id()
for product in products:
  print(product)
