import csv
import numpy as np
from explicit_mf import ExplicitMatrixFactorization

# train 파일에서  user_id, movie_id, rating list를 불러옵니다.
def read_train_file():
    user_ids = []
    movie_ids = []
    ratings = []
    timestamps = []
    with open("/home/hajung/movie/ml-20m/train_rating.tsv", 'r') as f:
        reader = csv.reader(f, delimiter ="\t")
        next(reader)
        for i, row in enumerate(reader):
            if i % 10000 == 0 :
                print(i)
            timestamp = row[3]
            user_ids.append(row[0])
            movie_ids.append(row[1])
            ratings.append(float(row[2]))
            timestamps.append(timestamp)

    return user_ids, movie_ids, ratings, timestamps

# test 파일에서  user_id, movie_id, rating list를 불러옵니다.
def read_test_file():
    user_ids = []
    movie_ids = []
    ratings = []
    timestamps = []
    with open("/home/hajung/movie/ml-20m/test_rating.tsv", 'r') as f:
        reader = csv.reader(f, delimiter ="\t")
        next(reader)
        for i, row in enumerate(reader):
            if i % 10000 == 0 :
                print(i)
            timestamp = row[3]
            user_ids.append(row[0])
            movie_ids.append(row[1])
            ratings.append(float(row[2]))
            timestamps.append(timestamp)
    return user_ids, movie_ids, ratings, timestamps

# user_id, movie_id 중 중복되지 않은 값을 추려서 1부터 user_id 개수 (혹은 movie_id) 까지의 idx 번호를 부여하는 dict을 만듭니다.
def get_user_movie_dict(user_ids, movie_ids):
    cnt = 0
    user_dict = dict()
    for user_id in user_ids:
        if not user_id in user_dict:
            user_dict[user_id] = cnt
            cnt += 1
    print("Total unique user num " + str(len(user_dict)))

    cnt = 0
    movie_dict = dict()
    for movie_id in movie_ids:
        if not movie_id in movie_dict:
            movie_dict[movie_id] = cnt
            cnt += 1

    print("Total unique movie num "+str(len(movie_dict)))

    return user_dict, movie_dict


# user_id x movie_id 인 행렬을 만들어서 해당 user, movie칸에 rating을 채웁니다.
def make_rating(user_dict, movie_dict, user_ids, movie_ids, ratings):
    n_users = len(user_dict)
    n_items = len(movie_dict)

    ratings_matrix = np.zeros((n_users, n_items),dtype =np.float16)
    for i, user_id in enumerate(user_ids):
        movie_id = movie_ids[i]
        rating = ratings[i]
        ratings_matrix[user_dict[user_id]][movie_dict[movie_id]] = rating

    return ratings_matrix


# user_id x movie_id 인 훈련 rating 행렬의 row중 1 x movie_id 인 test rating 벡터와 유클리디안 거리가 가장 가까운 user_id의 index 값을 리턴합니다.
# 이 함수를 사용해보니 느린 관계로 이번 실험에는 사용하지 않았습니다.
def get_min_distance_train_idx(train_rating, test_rating):
    min_distance = 9999999
    min_idx = -1
    for i, rating in enumerate(train_rating):
        delta = test_rating.shape[-1] - rating.shape[-1]
        rating = np.concatenate([rating, np.zeros([delta],dtype=np.float16)])
        dist = np.linalg.norm(test_rating - rating)
        if dist < min_distance:
            min_distance = dist
            min_idx = i
    if min_idx == -1:
        print("ERROR: could not find a min distance user idx")
    return min_idx


# 같은 user_id 값을 가진 test set의 user index 와 train set의 user index를 매핑합니다.
def init_match_with_train_test_user(train, test):
    match = dict()

    for user_id, user_idx in test.items():
        if user_id in train:
            match[user_idx] = train[user_id]
    return match


# train set에 없지만 test set에 있는 user 또는 movie의 idx 값의 리스트를 반환합니다.
def get_untrained_test_id(user_list_test, user_list_train, user_dict):
    exclued_ids = []
    for user_id in user_list_test:
        if not user_id in user_list_train:
            exclued_ids.append(user_id)
    exclude_idx = []
    for user_id in exclued_ids:
        exclude_idx.append(user_dict[user_id])
    return exclude_idx



# train set에 없지만 test set에는 있는 user id 에 대한 rating 이 가장 유사한 train set 의 user id를 매치해 주는 디셔너리를 반환합니다.
# train set에 없는 test set에 대한 예측을 유추하기 위해 작성한 함수이나, 속도 문제로 사용하지 않았습니다. 
def get_test_train_idx_match(user_dict_train, user_dict_test, test_rating, train_rating, excluded_user_idx):
    match = init_match_with_train_test_user(user_dict_train, user_dict_test)
    for user_idx in excluded_user_idx:
        rating = test_rating[user_idx]
        closest_user_idx_train = get_min_distance_train_idx(train_rating, rating)
        match[user_idx] = closest_user_idx_train
    return TestTrainMatchRecordList2match



# train set에서 만든 user_dict, movie_dict에 test set에 새로운 데이터가 나타나면 기존 dict에 추가하는 방식으로 test set의 user_dict와 movie_dict를 만들었습니다.
def get_user_movie_dict_ts(user_ids_ts, movie_ids_ts, user_dict_tr,
                           movie_dict_tr):
    cnt = len(user_dict_tr)
    # class UserIdIdxRecord = [Int2user_id, Int2user_idx]
    user_dict = user_dict_tr.copy()
    for user_id in user_ids_ts:
        if not user_id in user_dict:
            user_dict[user_id] = cnt
            cnt += 1
    print("Total unique user num " + str(len(user_dict)))

    cnt = len(movie_dict_tr)
    movie_dict = movie_dict_tr.copy()
    for movie_id in movie_ids_ts:
        if not movie_id in movie_dict:
            movie_dict[movie_id] = cnt
            cnt += 1

    print("Total unique movie num "+str(len(movie_dict)))

    return user_dict, movie_dict


# test set의 prediction 결과를 파일에 작성합니다.
def write_file(predicted, user_ids_ts, movie_ids_ts, timestamps_ts):
    with open("B_results_DS1.csv", 'w',newline="") as f:
        writer = csv.writer(f, delimiter=",")
        writer.writerow(["userId", "movieId", 'predicted rating', 'timestamp'])
        for i, user_id in enumerate(user_ids_ts):
            writer.writerow([user_id, movie_ids_ts[i], predicted[i], timestamps_ts[i]])



if __name__ == "__main__":
    # 파일을 읽습니다.
    user_ids_tr, movie_ids_tr, ratings_tr, timestamps_tr = read_train_file()
    user_ids_ts, movie_ids_ts, ratings_ts, timestamps_ts = read_test_file()
    print("Done Reading file")

    # user, movie 인덱스 딕셔너리를 만듭니다.
    user_dict_tr, movie_dict_tr = get_user_movie_dict(user_ids_tr, movie_ids_tr)
    user_dict_ts, movie_dict_ts = get_user_movie_dict_ts(user_ids_ts, movie_ids_ts, user_dict_tr, movie_dict_tr)
    print("Made Dictionary")    

    # (user x movie) 크기의 rating 행렬을 만듭니다. test의 rating 행렬은 train의 rating 행렬에 새로운 user, movie를 행과 열에 확장하여 만듭니다.
    ratings_matrix_tr = make_rating(user_dict_tr, movie_dict_tr, user_ids_tr, movie_ids_tr, ratings_tr)
    ratings_matrix_ts = make_rating(user_dict_ts, movie_dict_ts, user_ids_ts, movie_ids_ts, ratings_ts)
    print("Made Rating Matrix")
    print("train rating matrix shape: ")
    print(ratings_matrix_tr.shape)
    print("test rating matrix shape: ")
    print(ratings_matrix_ts.shape)

    # Matrix Factorization 모델 을 이용하여 학습과 평가를 시작합니다.
    model = ExplicitMatrixFactorization(ratings_matrix_tr, ratings_matrix_ts, latent_dim=40, learning_rate=0.01, reg_lambda=0.01, epoch_num=30)
    print("Starting Training.....")
    model.train_and_eval()

    # 학습을 마쳤으면 test data에 대한 rating 값을 predict하고 파일에 저장합니다.
    predicted = model.predict(user_ids_ts, movie_ids_ts, ratings_ts, user_dict_ts, movie_dict_ts)
    write_file(predicted, user_ids_ts, movie_ids_ts, timestamps_ts)
    print("Wrote prediction in file")
