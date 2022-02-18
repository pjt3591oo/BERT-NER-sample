import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# from Korpora import Korpora
# Korpora.fetch("naver_changwon_ner")

from Korpora import NaverChangwonNERKorpus


corpus = NaverChangwonNERKorpus()
print(corpus.train[0].text)
print(corpus.train[0].words)
print(corpus.train[0].tags)

print(corpus.get_all_tags())