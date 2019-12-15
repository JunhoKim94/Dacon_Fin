import numpy as np
import pandas as pd
from torch import nn,tensor
from sklearn.preprocessing import StandardScaler,RobustScaler,MinMaxScaler,MaxAbsScaler
import konlpy


"""
pandas에서 DataFrame에 적용되는 함수들
sum() 함수 이외에도 pandas에서 DataFrame에 적용되는 함수는 다음의 것들이 있다.
count 전체 성분의 (NaN이 아닌) 값의 갯수를 계산
min, max 전체 성분의 최솟, 최댓값을 계산
argmin, argmax 전체 성분의 최솟값, 최댓값이 위치한 (정수)인덱스를 반환
idxmin, idxmax 전체 인덱스 중 최솟값, 최댓값을 반환
quantile 전체 성분의 특정 사분위수에 해당하는 값을 반환 (0~1 사이)
sum 전체 성분의 합을 계산
mean 전체 성분의 평균을 계산
median 전체 성분의 중간값을 반환
mad 전체 성분의 평균값으로부터의 절대 편차(absolute deviation)의 평균을 계산
std, var 전체 성분의 표준편차, 분산을 계산
cumsum 맨 첫 번째 성분부터 각 성분까지의 누적합을 계산 (0에서부터 계속 더해짐)
cumprod 맨 첫번째 성분부터 각 성분까지의 누적곱을 계산 (1에서부터 계속 곱해짐)
"""
#train_data = np.loadtxt("./data/train.csv", delimiter = ",")
#train_data = pd.read_csv("./data/train.csv")
#iloc과 loc 사용해서 추출
#iloc --> 인덱스로 접근 가능 loc --> 키워드 접근도 가능


def tokenizing(batch_size ,tokenizer = Kkma()):
    
    tokenizer = tokenizer
    elements = dict()

    for iteration in range(batch_size):
        temp = tokenizer.morphs(batch_x.iloc[iteration])
        for item in temp:
            if item not in elements.keys():
                num = len(elements)
                elements[item] = num
                
    return elements