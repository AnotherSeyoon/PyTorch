# vocabulary 파일 세팅
# 라이브러리 불러오기
import urllib.request
import pandas as pd
from konlpy.tag import Mecab
from nltk import FreqDist, tokenize
import numpy as np
import matplotlib.pyplot as plt

# 데이터 받아오기
urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings.txt", filename = "ratings.txt")
# 데이터프레임에 저장
data = pd.read_table('ratings.txt')

# 임의로 100개만 저장
sample_data = data[:100]
# 한글과 공백을 제외하고 모두 제거
sample_data['document'] = sample_data['document'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","")

# 불용어 정의
stopwords = ['의','가','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','한','하다']

# 변수에 함수 넣기
tokenizer = Mecab()

# 토큰화 및 불용어가 제거된 데이터
tokenized = []

for sentence in sample_data['document']:
    temp = tokenizer.morphs(sentence) # 토큰화
    temp = [word for word in temp if not word in stopwords] # 불용어 제거
    tokenized.append(temp) # 리스트에 삽입

# 사용된 단어(토큰)의 사용빈도 데이터를 변수에 저장
vocab = FreqDist(np.hstack(tokenized))

vocab_size = 500

# 상위 vocab_size개의 단어만 보존
vocab = vocab.most_common(vocab_size)

# Enumerate 시작
# 인데그 0과 1을 제외한 나머지 단어들을 2부터 501까지 인덱스 부여
word_to_index = {word[0] : index + 2 for index, word in enumerate(vocab)}
word_to_index['pad'] = 1
word_to_index['unk'] = 0

encoded = []
for line in tokenized: # 입력 데이터에서 1줄씩 문장을 읽음
    temp = []
    for w in line: # 각 줄에서 1개씩 글자를 읽음
        try:
            temp.append(word_to_index[w]) # 글자를 해당되는 정수로 변환
        except KeyError:
            temp.append(word_to_index['unk']) # unk의 인덱스로 변환

    encoded.append(temp) # 리스트에 삽입

# 상위 10개 출력
print(encoded[:10])