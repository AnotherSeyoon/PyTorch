import urllib.request
import pandas as pd

urllib.request.urlretrieve("https://raw.githubusercontent.com/LawrenceDuan/IMDb-Review-Analysis/master/IMDb_Reviews.csv", filename="IMDb_Reviews.csv")

df = pd.read_csv('IMDb_Reviews.csv', encoding='latin1')
print(df.head())

print('전체 샘플의 개수 : {}'.format(len(df)))

train_df = df[:25000]
test_df = df[25000:]

