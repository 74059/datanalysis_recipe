# import modules
import ast
import copy

import numpy as np
import pandas as pd

from gensim.models import Word2Vec

# 리뷰데이터 복사본
review_sy = copy.deepcopy(pd.read_csv('C:/Users/LG/PycharmProjects/datanalysis_recipe/visualization/p2_data_preprocessing/data/recipe_review_preprocessing_1202.csv'))  # 상대경로로 수정

# func(str to lst)
def func(obj):
    List = []
    for i in ast.literal_eval(obj):
        List.append(i['name'])
    return List

# 리뷰데이터에서 학습데이터 추출, input_txt에 저장
input_txt = []

for idx, row in review_sy.iterrows():
    try: 
        ini_list = row['내용_전처리']
        res = ast.literal_eval(ini_list)  # ast.literal_eval takes a string and return the python evaluated object
        if row['내용_전처리'] == 'nan':
            pass
        else:
            input_txt.append(res)
        
    # print(idx, res)
    except Exception:
        # print(Exception)
        # print("row 내용_전처리:", row['내용_전처리'])
        continue

# 모델학습(word2vec 기본 파라미터)
model = Word2Vec(sentences = input_txt, vector_size=200, window=5, min_count=1, workers=4, sg=1)

# 모델확인
print(model.wv.vectors.shape)  # 메트릭스 크기
print(model.wv.most_similar("감자"))  # 명사
print(model.wv.most_similar("맛있"))  # 형용사

# 불필요한 메모리 unload
model.init_sims(replace=True)

# 모델저장
model_name = 'review_w2v'
model.save(model_name)