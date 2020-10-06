## 파일 설명
- train.py: 데이터셋 # Matrix Factorization 모델 훈련을 위한 코드
- explicit_mf.py: Matrix Factorization 모듈

```python
python train.py # 훈련을 시작하고 결과 파일을 작성합니다.
```
## 개발 환경
- Ubuntu Desktop 16.04
- Python 3.6 

## 데이터셋 분석

- MovieLense Dataset: 유저가 특정 영화에 대하여 0<rating <=5 인 값을 매긴 데이터입니다.
- 예시 <br> <br>
![데이터셋 예시](/images/dataset.png)
- 분포 <br> <br>
![데이터셋 분포](/images/dataset_dist.png)

## 사용 알고리즘 설명
(1) Matrix Factorization with Gradient Decent
- train set에 userId가 있는 test set에 대하여서 Matrix Factorization 을 적용합니다. 최적화 방법으로는 Gradient Descent를 사용하여 손실 함수의 Gradient x learning_rate 만큼 Latent variable의 값을 업데이트하여 손실 함수가 최소화될 수 있도록 하였습니다.


(2) 목적 함수(손실 함수) <br>
![목적 함수](/images/loss_func.png)

- Bias term을 사용하여 특정 유저나 특정 영화가 가질 수 있는 Bias를 완화합니다. 이 때 유저 u의 영화 i에 대한 Bias는 b(u,i) = b(u) + b(i) + total_average 로 나타내어 Bias가 전체 평균과 얼마나 차이가 나는지 표현할 수 있습니다. 또한 Latent User Variabe 인 P와 Latent Movie Variable인 Q의 norm, 그리고 Bias term의 norm을 손실함수에 더해주어서 모델이 overfit하여 복잡하게 되는 것을 방지 할 수 있습니다. 이때 lambda 값을 사용하여 이 Regularizer가 얼마나 모델에 영향을 끼치는 지 조절할 수 있습니다. lambda 값이 크면 모델이 단순해져서 variance가 줄어드는 대신 error가 증가할 수 있고, lambda 값이 작으면 모델이 복잡해져 variance가 늘어나는 대신 error는 감소할 수 있습니다.

(3) Train에 없는 test 데이터의 예측 방법 1: Inferring ratings using Collaborative Filtering based on nearest neighborhood method
- train set에 userId가 없는 test set에 대하여서는 가장 유사한, 유클리디안 거리(eucledian distance) 또는 해밍 거리(hamming distance)가 가장 가까운 train set 내 user의 prediction 결과를 원래 user의 prediction으로 유추합니다. <br> 
-> 시도하였으나, 함수의 속도가 느려 (4) 방법을 사용하여 train에 없는 test data를 예측하였습니다.

(4) Train에 없는 test 데이터의 예측 방법 2: Latent User Matrix (user x latent_dim)의 평균 벡터를, Latent Movie Matrix ( movie x latent_dim) 의 평균 벡터를 train에 없는 test 데이터를 예측할 때 사용합니다. 이때, Bias term 역시 평균 값으로 대체하여 사용합니다.



## hyper parameters
|hyper-param| value|
|--|--|
|regularizer lambda|0.01|
|epoch num #1|20|
|epoch num #2| 5|
|learning rate|0.01|
|latent dimension|40|


## 훈련 결과

훈련 결과에 따른 RMSE의 추이를 그래프로 나타내면 다음과 같습니다. <br>

| Epoch| Train RMSE | Test RMSE| 
|--|--|--|
|1|0.941|1.173|
|2|0.866|1.124|
|..|..|..|
|20|0.773|1.014|


![훈련 결과](/images/curve.png)




## Netflix prize 논문 요약
### 추천 시스템
(1) Content filtering
- User Profiling: 나이, 성별, 관심사, 지역 등 사용자의 특성을 사용한 추천
- Item Profiling: 영화의 장르, 영화의 주연배우, 영화의 감독 등 영화의 특성을 사용한 추천

(2) Collaborative Filtering
- 사용자와 아이템의 관계를 사용자-아이템 행렬을 구하여 분석합니다.
- 방법: neighborhood method, latent factor models
- 장점: 일반적으로 Content filtering보다 정확도가 높습니다.
- 단점: matrix factorization의 경우, train set에 없는 test set 데이터에 대한 추천을 할 수 없습니다.(cold start problem)

