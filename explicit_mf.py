#%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np

class ExplicitMatrixFactorization():
    def __init__(self, rating_tr, rating_ts, latent_dim, learning_rate, reg_lambda, epoch_num):
        self._rating_matrix_tr = rating_tr
        self._rating_matrix_ts = rating_ts
        self._user_num_tr = rating_tr.shape[0]
        self._item_num_tr = rating_tr.shape[1]
        self._user_num_ts = rating_ts.shape[0]
        self._item_num_ts = rating_ts.shape[1]
        self._latent_dim = latent_dim
        self._learning_rate = learning_rate
        self._reg_lambda = reg_lambda
        self._epochs = epoch_num
        self._P = np.random.normal(size=(self._user_num_tr, self._latent_dim))
        self._P = self._P.astype(np.float16, copy=False)
        self._Q = np.random.normal(size=(self._item_num_tr, self._latent_dim))
        self._Q = self._Q.astype(np.float16, copy=False)

        self._P_bias = np.zeros(self._user_num_tr, dtype=np.float16)
        self._Q_bias = np.zeros(self._item_num_tr, dtype=np.float16)

        self._bias = np.mean(self._rating_matrix_tr[np.where(self._rating_matrix_tr != 0)])
        self._train_record = []
        self._test_record = []


    def train_and_eval(self):
        # 모델을 학습힙니다. epoch만 큼 반복하여 Gradient Descent 로 손실 값의 그라디언트를 계산하여 Latent Variable을 업데이트합니다. 
        for epoch in range(self._epochs):
            print("Epoch: "+str(epoch+1))
            for i in range(self._user_num_tr):
                if i % 1000 == 0:
                    print(i)
                for j in range(self._item_num_tr):
                    if self._rating_matrix_tr[i][j] > 0:
                        self.gradient_descent(i, j, self._rating_matrix_tr[i][j])
            cost_tr = self.cost_tr()
            cost_ts = self.cost_ts()
            self._train_record.append(cost_tr)
            self._test_record.append(cost_ts)

            # test 의 RMSE를 출력합니다.
            print("test cost: ")
            print(self._test_record)
            # train 의 RMSE를 출력합니다. 
            print("train cost: ")
            print(self._train_record)


    def cost_ts(self):
        # test set의 RMSE를 계산합니다.
        nonzero_x, nonzero_y = self._rating_matrix_ts.nonzero()
        predicted = self.get_prediction_matrix_ts()
        cost = 0
        for x, y in zip(nonzero_x, nonzero_y):
            delta = self._rating_matrix_ts[x][y] - predicted[x][y]
            cost += pow(delta, 2)
        return np.sqrt(cost / len(nonzero_x))


    def cost_tr(self):
        # train set의 RMSE 를 계산합니다.
        nonzero_x, nonzero_y = self._rating_matrix_tr.nonzero()
        predicted = self.get_prediction_matrix()
        cost = 0
        for x, y in zip(nonzero_x, nonzero_y):
            delta = self._rating_matrix_tr[x][y] - predicted[x][y]
            cost += pow(delta, 2)
        return np.sqrt(cost / len(nonzero_x))


    def get_gradient(self, error, i, j):
        # 손실 값과 Latent Variable의 행렬 인덱스로 그라디언트를 계산합니다.
        delta_p = (error * self._Q[j, :]) - (self._reg_lambda * self._P[i, :])
        delta_q = (error * self._P[i, :]) - (self._reg_lambda * self._Q[j, :])
        return delta_p, delta_q


    def gradient_descent(self, i, j, rating):
        # 그라디언트를 구하고 learning rate 만큼 곱하여 Latent Variable에 더해주는 gradient descent를 실시합니다.
        prediction = self.get_single_prediction_by_index(i, j)
        error = rating - prediction

        self._P_bias[i] += self._learning_rate * (error - self._reg_lambda * self._P_bias[i])
        self._Q_bias[j] += self._learning_rate * (error - self._reg_lambda * self._Q_bias[j])

        delta_p, delta_q = self.get_gradient(error, i, j)
        self._P[i, :] += self._learning_rate * delta_p
        self._Q[j, :] += self._learning_rate * delta_q


    def get_single_prediction_by_index(self, i, j):
        # 인덱스 i,j 값에 해당하는 prediction을 계산합니다.
        return self._bias + self._P_bias[i] + self._Q_bias[j] + self._P[i, :].dot(self._Q[j, :].T)


    def get_prediction_matrix(self):
        # 최종 예측 결과 행렬을 계산합니다.
        return self._bias + self._P_bias[:, np.newaxis] + self._Q_bias[np.newaxis:, ] + self._P.dot(self._Q.T)


    def extend_latent(self, latent_matrix, ratings_tr, ratings_ts):
        # train set에 없는 test set을 예측하기 위해 trainset의 latent variable의 평균을 구하여 test set의 값으로 대체합니다.
        user_num, movie_num = ratings_tr.shape
        user_num_two , movie_num_two = ratings_ts.shape
        avg = np.average(latent_matrix,axis=0)
        delta = user_num_two - user_num
        add_avg = np.tile(avg,[delta, 1])

        return np.concatenate([latent_matrix, add_avg], axis=0)


    def extend_latent_bias(self, bias, ratings_tr, ratings_ts):
       # train set에 없는 test set을 예측하기 위해 trainset의 latent bias variable의 평균을 구하여 test set의 값으로 대체합니다. 
        user_num, movie_num = ratings_tr.shape
        user_num_two, movie_num_two = ratings_ts.shape
        avg = np.average(bias, axis=0)
        delta = user_num_two - user_num
        add_avg = np.tile([avg], [delta])
        return np.concatenate([bias, add_avg], axis=0)        

    def get_prediction_matrix_ts(self):
        # test set의 최종 결과 행렬을 계산합니다.
        self._P_two = self.extend_latent(self._P, self._rating_matrix_tr, self._rating_matrix_ts)
        self._Q_two = self.extend_latent(self._Q, self._rating_matrix_tr, self._rating_matrix_ts)
        self._b_P_two = self.extend_latent_bias(self._P_bias,self._rating_matrix_tr, self._rating_matrix_ts )
        self._b_Q_two = self.extend_latent_bias(self._Q_bias,self._rating_matrix_tr, self._rating_matrix_ts )

        return self._bias + self._b_P_two[:, np.newaxis] + self._b_Q_two[np.newaxis:, ] + self._P_two.dot(self._Q_two.T)

    def predict(self, user_ids_ts, movie_ids_ts, ratings_ts, user_dict_ts, movie_dict_ts):
        # test dataset의 예측 값을 계산하여 리스트의 현태로 리턴합니다.
        rating_predict = self.get_prediction_matrix_ts()
        prediction = []

        for i, user_id in enumerate(user_ids_ts):
            user_idx = user_dict_ts[user_id]
            movie_id = movie_ids_ts[i]
            movie_idx = movie_dict_ts[movie_id]
            prediction.append(rating_predict[user_idx][movie_idx])

        return prediction

    def plot_learning_curve(self):
        # 학습 커브를 그래프로 그립니다.
        ts = self._test_record
        tr = self._train_record

        linewidth = 3
        plt.plot(ts, label = 'Test', linewidth = linewidth)
        plt.plot(tr, label = 'Train', linewidth = linewidth)
        plt.xlabel('iterations')
        plt.ylabel('RMSE')
        plt.legend(loc = 'best')        


