en_text = "A Dog Run back corner near spare bedrooms"

# spaCy 사용하기

# spaCy 불러오기
import spacy
# 영어 로드하기
spacy_en = spacy.load("en_core_web_sm")

# 토큰화
def tokenize(en_text):
    return [tok.text for tok in spacy_en.tokenizer(en_text)]

# 출력
print(tokenize(en_text))

# NLTK 사용하기

# NLTK 불러오기
import nltk
# punkt 다운로드
nltk.download('punkt')

# space 단위와 구두점을 기준으로 토큰화 하는 함수
from nltk.tokenize import word_tokenize
print(word_tokenize(en_text))

# 띄어쓰기로 토큰화
print(en_text.split())

# 한국어 띄어쓰기 토큰화
kor_text = "사과의 놀라운 효능이라는 글을 봤어. 그래서 오늘 사과를 먹으려고 했는데 사과가 썩어서 슈퍼에 가서 사과랑 오렌지 사왔어"

# 문자열을 나눠 리스트로 저장한 후 출력
print(kor_text.split())

# 형태소 토큰화
# https://colab.research.google.com/drive/18ZJfJTvT6jIPolotbgQsRTuUxxydOir_?usp=sharing

# 문자 토큰화
print(list(en_text))
