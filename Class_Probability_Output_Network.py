import numpy as np
import csv
from scipy.stats import beta
import matplotlib.pyplot as plt
import openpyxl
import random
from numpy.linalg import inv
import sys
from math import exp
import pandas as pd
import sklearn.metrics as metrics
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis



# 0~1 로 normalization 하는 함수
def MinMaxScaler(data):
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    return numerator / (denominator + 1e-7)


# 그냥 예시를 들기위한 y_hat 값을 불러오는 함수입니다.
def read_yhat() :
    f = open('output.csv', 'r')
    y_hat = []
    csvReader = csv.reader(f, delimiter='\n')
    for i in csvReader:
        temp = []
        temp.append(i)
        y_hat.append(float(temp[0][0]))
    f.close()
    return y_hat

# array 를 list로 바꿔주는 함수
def ArrayToList(array) :
    list = []
    for i in range(array.shape[0]) :
        temp = []
        for k in range(array.shape[1]) :
            temp.append(array[i][k])
        list.append(temp)
    return list


def ArrayToList_1d(array) :
    list = []
    for i in range(array.shape[0]) :
        list.append(array[i])
    return list


def FrameToList(frame) :
    list = []
    for i in range(frame.shape[0]) :
        temp = []
        for j in range(frame.shape[1]) :
            temp.append(frame.iloc[i,j])
        list.append(temp)
    return list


# 1차원 리스트 평균
def Mean_1d(list) :
    temp = 0
    for i in range(len(list)) :
        temp += list[i]
    average = temp / len(list)
    return average

# 1차원 리스트 분산
def Variance_1d(list, mean) :
    temp = 0
    for i in range(len(list)) :
        temp += np.square(list[i]-mean)
    return temp/len(list)

# 2차원 리스트 mean var
def Mean(list) :
    average = []
    for i in range(len(list[0])) :
        temp = 0
        for k in range(len(list)) :
            temp += list[k][i]
        temp_average = temp / len(list)
        average.append(temp_average)
    return average


def Variance(list, mean) :
    var = []
    for i in range(len(list[0])):
        temp = 0
        for k in range(len(list)):
            temp += np.square(list[k][i] - mean[i])
        var.append(temp/len(list))
    return var


#  A, B 값
def AandB(mean, var) :
    a = mean * ( ((mean*(1-mean))/var) - 1)
    b = (1-mean) * ( ((mean*(1-mean))/var) - 1)
    return a, b


# PDF값을 그리기 위한 함수
def GetPDF(y_hat_box) :
    X_a = []
    X_b = []
    for i in range(len(y_hat_box)) :
        temp_a, temp_b = GetAandB(y_hat_box[i])
        X_a.append(temp_a)
        X_b.append(temp_b)

    cdf_x = np.linspace(0, 1, 1001)

    pdf_line = []
    for i in range(len(X_a)) :
        pdf_line.append(beta.pdf(cdf_x, X_a[i], X_b[i]))

    # 측정된 pdf 값을 그리는 plot
    for i in range(len(pdf_line)) :
        plt.plot(cdf_x, pdf_line[i]);

    # plt.ylim(0, 40)
    # plt.show()

    # y_hat에 대한 histogram
    # hist_list = []
    # [hist_list.append(y_hat_box[i]) for i in range(len(y_hat_box))]
    # plt.hist(hist_list)
    # plt.xlim(0,1)
    # plt.show()

    return pdf_line


# def GetCDF(y_hat, PD_LEN, NONPD_LEN) :
#     P_y, N_y = PyNy(y_hat, PD_LEN)
#     for i in range(len(N_y)) :
#         N_y[i] = 1-N_y[i]
#
#     cdf_min = 1
#     for i in range(1, 999, 1):
#         temp = abs(P_y[i] - N_y[i])
#         if cdf_min >= temp:
#             cdf_min = temp
#             cdf_idx = i
#
#     for i in range(len(y_hat)):
#         if y_hat[i] >= cdf_idx * 0.001:
#             y_hat[i] = 1
#         else:
#             y_hat[i] = -1
#
#     return y_hat



def GetAandB(y_hat) :
    y_hat_mean = Mean_1d(y_hat)
    y_hat_var = Variance_1d(y_hat, y_hat_mean)
    a, b = AandB(y_hat_mean, y_hat_var)
    return a, b

def PyNy(y_hat, PD_LEN) :
    PD_y_hat = y_hat[:PD_LEN]
    PD_y_hat_mean = Mean_1d(PD_y_hat)
    PD_y_hat_var = Variance_1d(PD_y_hat, PD_y_hat_mean)
    PD_a, PD_b = AandB(PD_y_hat_mean, PD_y_hat_var)

    nonPD_y_hat = y_hat[PD_LEN:]
    nonPD_y_hat_mean = Mean_1d(nonPD_y_hat)
    nonPD_y_hat_var = Variance_1d(nonPD_y_hat, nonPD_y_hat_mean)
    N_a, N_b = AandB(nonPD_y_hat_mean, nonPD_y_hat_var)

    cdf_x = np.linspace(0, 1, 1001)

    P_y = beta.cdf(cdf_x, PD_a, PD_b)
    N_y = beta.cdf(cdf_x, N_a, N_b)

    for i in range(len(N_y)) :
        N_y[i] = 1-N_y[i]

    # 측정된 cdf 값을 그리는 plt
    # plt.plot(cdf_x, P_y);
    # plt.plot(cdf_x, N_y);
    #
    # plt.ylim(0, 2)
    # plt.show()

    # y_hat 값에 대한 histogram 을 그리는 plt
    # if PD_a > 0:
    #     plt.hist([y_hat[:PD_LEN], y_hat[PD_LEN:]])
    #     plt.show()

    return P_y, N_y


# 측정된 cdf_line(P_y)을 이용하여 y_hat에 따른 cdf 값 반환
def GetCDF(y_hat_box, sig = 0) :
    y_hat = y_hat_box[0] + y_hat_box[1]
    P_y, N_y = PyNy(y_hat, len(y_hat_box[0]))

    cdf_x = np.linspace(0, 1, 1001)
    plt.plot(cdf_x, P_y);
    plt.plot(cdf_x, N_y);

    plt.ylim(0, 2)
    plt.show()

    cdf_min = 1
    for i in range(1, 999, 1):
        temp = abs(P_y[i] - N_y[i])
        if cdf_min >= temp:
            cdf_min = temp
            cdf_idx = i

    cdf_idx = cdf_idx/1000

    y_hat_cdf = []
    for i in range(len(y_hat)) :
        y_hat_cdf.append(P_y[int(round(y_hat[i],3)*1000)])

    if sig == 0 :
        return y_hat_cdf, P_y
    else :
        return y_hat_cdf, P_y, cdf_idx


def ClassifyByCDF(y_hat, cdf_idx) :
    for i in range(len(y_hat)) :
        if y_hat[i] > cdf_idx :
            y_hat[i] = 1
        else :
            y_hat[i] = -1
    return y_hat


def GetCDF_line(y_hat, P_y) :

    y_hat_cdf = []
    for i in range(len(y_hat)) :
        y_hat_cdf.append(P_y[int(round(y_hat[i],3)*1000)])

    return y_hat_cdf

# 데이터 간 거리 구하는 함수
def Dist(x1, x2, E):
    dist = 0
    for i in range(0, E):
        dist += np.square(x1[i] - x2[i])

    return np.sqrt(dist)

# 각 데이터 별로 가장 가까운 커널과 연결시켜주는 함수, stand = kernel
# stand list의 0번째 커널과 가장 가까운 애들이 return 되는 temp 라는 리스트의 0번째에 위치하게 됩니다.
def Friend(list, stand) :
    temp = []
    friend_idx = []
    for i in range(len(stand)):
        temp.append([])
        friend_idx.append([])
    for i in range(len(list)):
        dist = []
        for j in range(len(stand)):
            dist.append(Dist(list[i], stand[j], len(list[i])))
        minimum = min(dist)
        idx = dist.index(minimum)
        temp[idx].append(list[i])
        friend_idx[idx].append(i)
    # temp는 friend의 실제 값 저장
    return temp, friend_idx


# kernel 값을 이용하여 multiple linear regression 함.
def kernel_regression(One_kernel, Zero_kernel, one_another, C) :
    One_fr, One_friend_idx = Friend(one_another[0] ,One_kernel)
    Zero_fr, Zero_friend_idx = Friend(one_another[1] ,Zero_kernel)
    all_fr = One_fr + Zero_fr
    all_data = one_another[0] + one_another[1]

    all_tilde = []
    # 커널에 달라붙은 데이터를 이용(all_fr)하여 tilde 값을 구다.
    for i in range(len(all_fr)):
        all_tilde.append(kernel_C(all_data, i, all_fr, C))

    true = []
    for i in range(len(one_another[0])):
        true.append(1)
    for i in range(len(one_another[1])) :
        true.append(-1)

    # 여기서부턴 MLR 수식입니다.
    true = np.array(true)
    all_tilde = np.transpose(np.array(all_tilde))

    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    clf = LinearDiscriminantAnalysis()
    clf.fit(all_tilde, true)

    B = ArrayToList_1d(clf.coef_[0])
    all_tilde = ArrayToList(all_tilde)

    y_hat = []
    for i in range(len(all_tilde)):
        temp = 0
        for j in range(len(B)):
            temp += B[j] * all_tilde[i][j]
        y_hat.append(temp)

    return y_hat, true, B


# stand = kernel 입니다.
# 커널에 달라붙은 데이터의 var 및 mean 값을 이용하여 데이터의 값을 구한다.
def kernel_C(list, idx, all_friend, C) :
    friend = all_friend[idx]

    try :
        mean = Mean(friend)
        var = Variance(friend, mean)
    except :
        print('friend err')
        print(friend)

    output = []
    for i in range(len(list)) :
        temp = 0
        for j in range(len(mean)) :
            if var[j] == 0 : # 혹시 커널에 달라붙은 데이터가 1개밖에 없을때 variance 를 epsilon으로 처리
                var[j] = sys.float_info.epsilon
            temp += np.square(list[i][j]-mean[j])/var[j]

        temp = -temp/2
        temp = exp(C * temp)
        output.append(temp)

    return output


# train 에서 나온 B 값을 이용하여 test의 regression 값을 구합니다.
def kernel_regression_B(One_kernel, Zero_kernel, train_one_another, test_one_another, B, C) :
    One_fr, One_friend_idx = Friend(train_one_another[0], One_kernel)
    Zero_fr, Train_Zero_friend_idx = Friend(train_one_another[1], Zero_kernel)

    all_fr = One_fr + Zero_fr
    all_data = test_one_another[0] + test_one_another[1]

    all_tilde = []
    # 커널에 달라붙은 데이터를 이용(all_fr)하여 test data에 대한 tilde 값을 구합니다.
    for i in range(len(all_fr)):
        all_tilde.append(kernel_C(all_data, i, all_fr, C))

    true = []
    for i in range(len(test_one_another[0])):
        true.append(1)
    for i in range(len(test_one_another[1])) :
        true.append(-1)

    # 여기서부턴 MLR 수식입니다.
    all_tilde = np.transpose(np.array(all_tilde))
    all_tilde = ArrayToList(all_tilde)

    y_hat = []
    for i in range(len(all_tilde)):
        temp = 0
        for j in range(len(B)):
            temp += B[j] * all_tilde[i][j]
        y_hat.append(temp)

    return y_hat, true


def new_kernel_regression_B(One_kernel, Zero_kernel, train_one_another, test_data, B, C) :
    One_fr, One_friend_idx = Friend(train_one_another[0], One_kernel)
    Zero_fr, Train_Zero_friend_idx = Friend(train_one_another[1], Zero_kernel)

    all_fr = One_fr + Zero_fr

    all_tilde = []
    # 커널에 달라붙은 데이터를 이용(all_fr)하여 test data에 대한 tilde 값을 구합니다.
    for i in range(len(all_fr)):
        all_tilde.append(kernel_C(test_data, i, all_fr, C))


    # 여기서부턴 MLR 수식입니다.
    all_tilde = np.transpose(np.array(all_tilde))
    all_tilde = ArrayToList(all_tilde)

    y_hat = []
    for i in range(len(all_tilde)):
        temp = 0
        for j in range(len(B)):
            temp += B[j] * all_tilde[i][j]
        y_hat.append(temp)

    return y_hat


def DataBox(dataX, dataY, D) :
    box = []
    for i in range(D) :
        box.append([])
    for i in range(len(dataX)) :
        box[dataY[i]].append(dataX[i])
    return box


# xbox = friend 같은개념
# y_hat을 friend 리스트의 개수만큼 나눠주는 함수
def Sameform(y_hat, Xbox) :
    Hbox = []
    for i in range(len(Xbox)) :
        Hbox.append([])
    pre_temp = 0
    for i in range(len(Xbox)) :
        next_temp = len(Xbox[i]) + pre_temp
        Hbox[i] = y_hat[pre_temp:next_temp]
        pre_temp = next_temp
    return Hbox


def kernel_pred(data, pdf_line) :
    pred_data = []
    for i in range(len(data)) :
        idx = int(round(data[i] * 1000))
        temp = []
        for j in range(len(pdf_line)) :
            temp.append(pdf_line[j][idx])
        maximum = max(temp)
        temp_idx = temp.index(maximum)
        pred_data.append(temp_idx)
    return pred_data


# y 값 별로 한 리스트에 모아주는 함수
def clustering_onehot(x_data, y_data) :
    zero, onehot_type = [], []
    [zero.append(0) for i in range(len(y_data[0]))]
    for i in range(len(y_data[0])) :
        temp = zero + []
        temp[i] = 1
        onehot_type.append(temp)

    friend = []
    [friend.append([]) for i in range(len(y_data[0]))]
    for i in range(len(y_data)) :
        for j in range(len(onehot_type)) :
            if y_data[i] == onehot_type[j] :
                friend[j].append(x_data[i])

    return friend


# regression 된 데이터에 대하여 cdf maximum 값 기준으로 최종 class 반환
def Onehot(temp_train_pred) :
    train_pred = []
    for i in range(len(temp_train_pred)):
        maximum = max(temp_train_pred[i])
        idx = temp_train_pred[i].index(maximum)
        temp = []
        for k in range(len(temp_train_pred[i])) :
            if k == idx :
                temp.append(1)
            else :
                temp.append(0)
        train_pred.append(temp)
    return train_pred

# 평가 함수 틀린애들의 idx 반환
def EvaluateOnehot(y_hat, true) :
    err_idx = []
    for i in range(len(y_hat)) :
        if y_hat[i] != true[i] :
            err_idx.append(i)
    return err_idx



# 클래스 1, 클래스 0 인애들 나누기
def OneAnother(friend, idx) :
    one = friend[idx] + []
    another = []
    for i in range(len(friend)) :
        if i != idx :
            another += (friend[i] + [])
    one_another = [one, another]

    return one_another


def MakeKernel(KERNEL_LEN, FEATURE_LEN) :
    One_kernel = []
    Zero_kernel = []

    for i in range(KERNEL_LEN):
        temp = []
        for j in range(FEATURE_LEN):
            temp.append(random.uniform(0, 1))
        One_kernel.append(temp)

    for i in range(KERNEL_LEN):
        temp = []
        for j in range(FEATURE_LEN):
            temp.append(random.uniform(0, 1))
        Zero_kernel.append(temp)

    return One_kernel, Zero_kernel


# 커널 centroid 잡는 함수.
# 한 class에 대한 centroid,
def FirstCentroid(list, kernel) :
    friend, friend_idx = Friend(list, kernel)

    # 새로운 커널
    new_kernel = []
    for i in range(len(kernel)) :
        a = np.array(friend[i])

        if len(a) == 0 :
            new_kernel.append([])
        else :
            a = np.mean(a, axis=0)
            temp_a = []
            for k in range(len(a)) :
                temp_a.append(a[k])
            new_kernel.append(temp_a)

    # 달라붙은 data가 하나도 없는 커널 제거
    for i in range(len(new_kernel)-1, -1, -1) :
        if len(friend[i]) == 0 :
            del new_kernel[i]
            del friend[i]
            del friend_idx[i]

    return new_kernel, friend, friend_idx

# 커널의 이동이 멈출때 까지 centroid 하는 함수.
def SecondCentroid(One_data, Zero_data, One_kernel, Zero_kernel) :
    new_One_kernel = One_kernel + []
    new_Zero_kernel = Zero_kernel + []

    while (1):
        pre_One_kernel = new_One_kernel + []
        pre_Zero_kernel = new_Zero_kernel + []
        new_One_kernel, One_friend, One_friend_idx = FirstCentroid(One_data, pre_One_kernel)
        new_Zero_kernel, Zero_friend, Zero_friend_idx = FirstCentroid(Zero_data, pre_Zero_kernel)

        # 이전 커널과 centroid 했던 커널들의 위치가 똑같다면 = 커널이 더이상 이동을 하지 않는다면
        if pre_One_kernel + pre_Zero_kernel == new_One_kernel + new_Zero_kernel :
            # centroid를 멈춘다.
            break;

    return new_One_kernel, new_Zero_kernel, One_friend, Zero_friend


# 커널별 에러 측정 함수
def EvaluateError(list, One_kernel, Zero_kernel, ONE_LEN):
    # temp 는 각 커널에 달라붙은 애들의 index
    # err_idx 는 err난 애들의 index
    # kerner_err 은 각 커널별 에러횟수
    all_kernel = One_kernel + Zero_kernel
    temp = []
    kernel_err = []
    err_idx = []

    for i in range(len(all_kernel)):
        temp.append([])
        kernel_err.append(0)

    for i in range(len(list)):
        dist = []
        # 각 커널과의 거리를 구한 후
        for j in range(len(all_kernel)):
            dist.append(Dist(list[i], all_kernel[j], len(list[i])))

        # 거리가 가장 가까운 커널을 선정하여
        minimum = min(dist)
        idx = dist.index(minimum)
        temp[idx].append(i)

        # 그 커널이 1속성이면 1로 예측, 0 속성이면 0으로 예측 후
        # 커널과 실제 class 가 같으면 통과, 다르면 그 커널에 에러 1개 추가
        if i < ONE_LEN:
            if idx >= len(One_kernel):
                kernel_err[idx] += 1
                err_idx.append(i)
        if i >= ONE_LEN:
            if idx < len(One_kernel):
                kernel_err[idx] += 1
                err_idx.append(i)

    return temp, kernel_err, err_idx



# 각 kernel의 error 측정 후 (cdf가 아닌 각 데이터별로 가장 가까운 커널의 속성을 기준으로 예측하여 err 측정)
# (err가 가장 큰 커널 & err가 err_rate 이상) 이면 그 커널 주변에 그 커널과 반대 속성을 가진 커널 생성
def KernelError(One_kernel, Zero_kernel, One_data, Zero_data) :
    err_rate = 0.3

    all_data = One_data + Zero_data
    all_kernel = One_kernel + Zero_kernel
    ONE_LEN = len(One_data)

    # temp 는 각 커널별 달라붙은 애들의 index
    # err_idx 는 err난 애들의 index
    temp, kernel_err, err_idx = EvaluateError(all_data, One_kernel, Zero_kernel, len(One_data))
    # friend = Friend(all_data, all_kernel)

    # err 가 가장 큰 커널을 구하고 그 커널과 반대인 커널을 그 커널 구간 내부에 생성, 단 err가 가장 큰 커널의 에러율이 0.5 미만이면 커널 생성 중단
    maximum = max(kernel_err)
    idx = kernel_err.index(maximum)

    temp_One_kernel = One_kernel + []
    temp_Zero_kernel = Zero_kernel + []

    sig = 0     # 커널 생성 시도 횟수

    # 커널 에러율이 err_rate 이상이면 커널 생성시작
    if (kernel_err[idx]/len(temp[idx])) >= err_rate :
        # 에러가 높은 커널이 0 클래스의 커널이라면, 그 근처에 1 클래스 커널 생성
        if idx >= len(One_kernel) :
            while(1) :
                # 커널 근처에 커널을 새로 생선한다는게 그 커널과, 그 커널과 가장 가까운 커널 사이의 절반 거리 안에 있는 범위 내에 커널을 생성
                dist = []
                for i in range(len(all_kernel)) :
                    if i == idx :
                        dist.append(999999999999)
                    else :
                        dist.append(Dist(all_kernel[idx], all_kernel[i], len(all_data[0]))) # len(all_data[0])은 feature의 길이임
                minimum = min(dist)
                # 원에 내적하는 사각형으로 거리계산 후 커널생성
                r = minimum/2/np.sqrt(2)

                ker_temp = [] # 앞으로 추가할 커널(임시)
                for i in range(len(all_data[0])) :
                    ker_temp.append(random.uniform(all_kernel[idx][i]-r, all_kernel[idx][i]+r))                 # 커널 근처에 랜덤값으로 커널 생성

                temp_One_kernel.append(ker_temp)    # 임시커널을 추가하고
                new_One_kernel, new_Zero_kernel, One_friend, Zero_friend = SecondCentroid(One_data, Zero_data, temp_One_kernel, temp_Zero_kernel)  # centroid를 다시잡고 끝

                # 새로운 커널 생성으로 이전 커널에 영향을 미쳤다면(이전 커널 개수와 같더라도)  while 문 탈출
                if new_One_kernel != One_kernel :
                    finish = 0
                    break;
                # 그렇지 않으면
                else :
                    sig += 1  # 커널 생성 시도횟수가 1 증가

                # 만약 커널 생성 시도 횟수가 10회를 넘어간다면(10회 이상 시도했으나 더이상 커널을 생성할 수 없다면) while 문 탈출
                if sig > 70 :
                    finish = 1
                    break;

        # 에러가 높은 커널이 1 클래스의 커널이라면, 그 근처에 0 클래스 커널 생성
        else :
            while (1):
                # 커널 근처에 커널을 새로 생선한다는게 그 커널과, 그 커널과 가장 가까운 커널 사이의 절반 거리 안에 있는 범위 내에 커널을 생성
                dist = []
                for i in range(len(all_kernel)):
                    if i == idx:
                        dist.append(999999999999)
                    else:
                        dist.append(Dist(all_kernel[idx], all_kernel[i], len(all_data[0])))  # len(all_data[0])은 feature의 길이임
                minimum = min(dist)
                # 원에 내적하는 사각형으로 거리계산 후 커널생성
                r = minimum / 2 / np.sqrt(2)

                ker_temp = []  # 앞으로 추가할 커널(임시)
                for i in range(len(all_data[0])):
                    ker_temp.append(random.uniform(all_kernel[idx][i] - r, all_kernel[idx][i] + r))  # 커널 근처에 랜덤값으로 커널 생성

                temp_Zero_kernel.append(ker_temp)  # 임시커널을 추가하고
                new_One_kernel, new_Zero_kernel, One_friend, Zero_friend = SecondCentroid(One_data, Zero_data, temp_One_kernel, temp_Zero_kernel)  # centroid를 다시잡고 끝

                # 새로운 커널 생성이 이전 커널에 영향을 미쳤다면(이전 커널 개수와 같더라도) while 문 탈출
                if new_Zero_kernel != Zero_kernel:
                    finish = 0
                    break;
                # 그렇지 않으면
                else:
                    sig += 1  # 커널 생성 시도횟수가 1 증가

                # 만약 커널 생성 시도 횟수가 10회를 넘어간다면(10회 이상 시도했으나 더이상 커널을 생성할 수 없다면) while 문 탈출
                if sig > 70:
                    finish = 1
                    break;

        return new_One_kernel, new_Zero_kernel, finish

    # 가장 에러가 높은 커널의 에러가 err_rate 미만일 땐 커널 생성없이 return
    else :
        finish = 1
        return One_kernel, Zero_kernel, finish


def RandomKernel(One_kernel, Zero_kernel, One_data, Zero_data) :
    temp_One_kernel = One_kernel + []
    temp_Zero_kernel = Zero_kernel + []

    a, b = MakeKernel(1, len(One_data[0]))
    temp_One_kernel.append(a[0])
    temp_Zero_kernel.append(b[0])

    new_One_kernel, new_Zero_kernel, One_friend, Zero_friend = SecondCentroid(One_data, Zero_data, temp_One_kernel, temp_Zero_kernel)

    return new_One_kernel, new_Zero_kernel

def DeleteEmptyKernel(One_kernel, Zero_kernel, One_data, Zero_data) :
    temp_One_kernel = One_kernel + []
    temp_Zero_kernel = Zero_kernel + []
    one_friend, one_friend_idx = Friend(One_data, temp_One_kernel)
    zero_friend, zero_friend_idx = Friend(Zero_data, temp_Zero_kernel)

    for i in range(len(temp_One_kernel) - 1, -1, -1):
        if len(one_friend[i]) == 0:
            del temp_One_kernel[i]

    for i in range(len(temp_Zero_kernel) - 1, -1, -1):
        if len(zero_friend[i]) == 0:
            del temp_Zero_kernel[i]

    return temp_One_kernel, temp_Zero_kernel


def TrainLayer(One_data, Zero_data) :
    # 각 class 별로 초기 커널 num개씩 랜덤생성 (달라붙은 데이터가 없는 커널은 자동으로 사라짐)
    num = 10
    One_kernel, Zero_kernel = MakeKernel(num, len(One_data[0]))

    print('Start training')
    # add = 0
    while(1) :
        One_kernel, Zero_kernel, finish = KernelError(One_kernel, Zero_kernel, One_data, Zero_data) # 커널별 에러를 측정하고 에러가 큰 커널 근처에 반대 속성의 커널 생성
        # One_kernel, Zero_kernel = RandomKernel(One_kernel, Zero_kernel, One_data, Zero_data)
        One_kernel, Zero_kernel = DeleteEmptyKernel(One_kernel, Zero_kernel, One_data, Zero_data) # 달라붙은 데이터가 없는 커널 제거
        kernel_len = [len(One_kernel), len(Zero_kernel)]
        print(kernel_len)
        if finish == 1 :
            break;
        # add += 1
        # if add > 2 :        # finish = 1 => 더이상 커널 생성 불가능
        #     break;

    return One_kernel, Zero_kernel


def TwoClassLayer(train_one_another, test_one_another, k, x_train, x_test) :
    One_kernel, Zero_kernel = TrainLayer(train_one_another[0], train_one_another[1])  # 커널 생성하는 함수
    One_kernel, Zero_kernel = DeleteEmptyKernel(One_kernel, Zero_kernel, train_one_another[0], train_one_another[1])  # 달라붙은 데이터가 없는 커널 제거
    temp_y_hat, true, B = kernel_regression(One_kernel, Zero_kernel, train_one_another, k)  # kernel 을 이용한 regression
    y_hat = ArrayToList_1d(MinMaxScaler(temp_y_hat))  # y_hat 값을 0~1 사이로
    y_hat_box = Sameform(y_hat, train_one_another)  # y_hat_box 형식 => y_hat_box[n] 에는 n번째 커널에 속해있는 모든 y_hat 값이 들어있습니다.
    y_hat_cdf, cdf_line, cdf_idx = GetCDF(y_hat_box, 1)  # y_hat에 따른 cdf 값, cdf_line 반환
    plt.hist([y_hat_box[0], y_hat_box[1]])
    plt.show()

    y_hat = ClassifyByCDF(y_hat, cdf_idx)
    err_idx = EvaluateOnehot(y_hat, true)  # err 측정
    tr_precision = metrics.precision_score(y_hat, true)
    tr_recall = metrics.recall_score(y_hat, true)
    tr_f1_score = metrics.f1_score(y_hat, true)
    print(('train accuracy : %f')%(1 - (len(err_idx) / len(y_hat))))
    print(('train precision : %f') % (tr_precision))
    print(('train recall : %f') % (tr_recall))
    print(('train f1_score : %f') % (tr_f1_score))
    y_hat=list(y_hat)
    y=np.where(y_hat==1)
    print(y[0])
    

    test_temp_y_hat, test_true = kernel_regression_B(One_kernel, Zero_kernel, train_one_another, test_one_another, B, k)
    temp_y_hat_ = test_temp_y_hat + temp_y_hat
    temp_y_hat_ = ArrayToList_1d(MinMaxScaler(temp_y_hat_)) # train, test 한꺼번에 normalization 해야함
    temp_y_hat_ = temp_y_hat_[:len(test_temp_y_hat)]
    test_y_hat = ClassifyByCDF(temp_y_hat_, cdf_idx)
    test_err_idx = EvaluateOnehot(test_y_hat, test_true)  # err 측정
    test_precision = metrics.precision_score(test_y_hat, test_true)
    test_recall = metrics.recall_score(test_y_hat, test_true)
    test_f1_score = metrics.f1_score(test_y_hat, test_true)
    print(('test accuracy : %f') % (1 - (len(test_err_idx) / len(test_y_hat))))
    print(('test precision : %f') % (test_precision))
    print(('test recall : %f') % (test_recall))
    print(('test f1_score : %f') % (test_f1_score))

    return


def Layer(friend, y_train) :
    accuracy = []
    all_one_another = []
    all_one_kernel = []
    all_zero_kernel = []
    all_B = []
    all_cdf = []
    all_y_hat = []
    k_err = []
    k = 1
    cdf_box = []
    for idx in range(len(friend)):
        # 원하는 클래스 1값, 다른클래스 0값 으로 만들어 주는 함수, 1 클래스가 one_another[0], 0 클래스가 one_another[1]
        one_another = OneAnother(friend, idx)
        One_kernel, Zero_kernel = TrainLayer(one_another[0], one_another[1])  # 커널 생성하는 함수
        One_kernel, Zero_kernel = DeleteEmptyKernel(One_kernel, Zero_kernel, one_another[0],
                                                       one_another[1])  # 달라붙은 데이터가 없는 커널 제거
        temp_y_hat, true, B = kernel_regression(One_kernel, Zero_kernel, one_another, k)  # kernel 을 이용한 regression
        y_hat = ArrayToList_1d(MinMaxScaler(temp_y_hat))  # y_hat 값을 0~1 사이로
        y_hat_box = Sameform(y_hat, one_another)  # y_hat_box 형식 => y_hat_box[n] 에는 n번째 커널에 속해있는 모든 y_hat 값이 들어있습니다.
        y_hat_cdf, cdf_line = GetCDF(y_hat_box)  # y_hat에 따른 cdf 값, cdf_line 반환
        # plt.hist([y_hat_box[0], y_hat_box[1]])
        # plt.show()

        cdf_box.append(y_hat_cdf)  # 최종 cdf값 저장

        # 얘네들은 test 할때에 필요한 데이터들 저장하는 리스트들입니다.
        all_y_hat.append(y_hat)
        all_cdf.append(cdf_line)
        all_one_another.append(one_another)
        all_one_kernel.append(One_kernel)
        all_zero_kernel.append(Zero_kernel)
        all_B.append(B)

    temp_train_pred = ArrayToList(np.transpose(np.array(cdf_box)))  # cdf_box를 transpose하고
    train_pred = Onehot(temp_train_pred)  # 최종 class를 onehot 으로 결정 # 예를들면 [ 0.34, 0.5, 0.4, 0.77, 0.9 ] 같이 되어있으면 이중에 max 값인 0.9 를 1로 하고 나머지를 0 으로 처리 => [ 0, 0, 0, 0, 1 ]
    err_idx = EvaluateOnehot(train_pred, y_train)  # err 측정
    accuracy.append(1 - len(err_idx) / len(train_pred))
    print('accuracy')
    print(accuracy)

    return temp_train_pred


def Train_TwoClassLayer(train_one_another, k) :
    One_kernel, Zero_kernel = TrainLayer(train_one_another[0], train_one_another[1])  # 커널 생성하는 함수
    One_kernel, Zero_kernel = DeleteEmptyKernel(One_kernel, Zero_kernel, train_one_another[0], train_one_another[1])  # 달라붙은 데이터가 없는 커널 제거
    temp_y_hat, true, B = kernel_regression(One_kernel, Zero_kernel, train_one_another, k)  # kernel 을 이용한 regression
    y_hat = ArrayToList_1d(MinMaxScaler(temp_y_hat))  # y_hat 값을 0~1 사이로
    y_hat_box = Sameform(y_hat, train_one_another)  # y_hat_box 형식 => y_hat_box[n] 에는 n번째 커널에 속해있는 모든 y_hat 값이 들어있습니다.
    y_hat_cdf, cdf_line, cdf_idx = GetCDF(y_hat_box, 1)  # y_hat에 따른 cdf 값, cdf_line 반환
    plt.hist([y_hat_box[0], y_hat_box[1]])
    plt.show()

    y_hat = ClassifyByCDF(y_hat, cdf_idx)
    err_idx = EvaluateOnehot(y_hat, true)  # err 측정
    tr_precision = metrics.precision_score(y_hat, true)
    tr_recall = metrics.recall_score(y_hat, true)
    tr_f1_score = metrics.f1_score(y_hat, true)
    print(('train accuracy : %f')%(1 - (len(err_idx) / len(y_hat))))
    print(('train precision : %f') % (tr_precision))
    print(('train recall : %f') % (tr_recall))
    print(('train f1_score : %f') % (tr_f1_score))

    return cdf_idx, temp_y_hat, One_kernel, Zero_kernel, B

def Test_TwoClassLayer(cdf_idx, temp_y_hat, One_kernel, Zero_kernel, train_one_another, test_data, B, k) :
    test_temp_y_hat = new_kernel_regression_B(One_kernel, Zero_kernel, train_one_another, test_data, B, k)
    temp_y_hat_ = test_temp_y_hat + temp_y_hat
    temp_y_hat_ = ArrayToList_1d(MinMaxScaler(temp_y_hat_)) # train, test 한꺼번에 normalization 해야함
    temp_y_hat_ = temp_y_hat_[:len(test_temp_y_hat)]
    test_y_hat = ClassifyByCDF(temp_y_hat_, cdf_idx)

    return  test_y_hat

def ChangeOnehot(idx, dim) :
    temp = []
    for i in range(dim) :
        temp.append(0)
    temp[idx] = 1
    return temp
