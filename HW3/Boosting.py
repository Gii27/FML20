import pandas as pd
import math
import numpy as np

filename = "abalone.data"


def data_arrange(filename):
    df = pd.read_csv(filename, sep=",", header=0)

    df.loc[df['Sex'] == "M", "Sex"] = 1
    df.loc[df['Sex'] == "F", "Sex"] = -1
    df.loc[df['Sex'] == "I", "Sex"] = 0

    df.loc[df["Rings"] > 9, "Rings"] = -1
    df.loc[df["Rings"] != -1, "Rings"] = 1

    # print(df)
    train_range = 3133
    test = df.loc[train_range:].reset_index(drop=True)
    # print(test)
    df = df.loc[:train_range - 1]
    print(df)

    target = df["Rings"]
    print(test)
    df = df.iloc[:, :-1]

    stumps = df.mean()
    print(stumps)


    for c in df.columns:
        df.loc[df[c] <= stumps[c], c] = 'a'
        df.loc[df[c] != 'a', c] = -1
        df.loc[df[c] == 'a', c] = 1             # df[i, j] = h_j(x_i)

    # print(df)

    for c in df.columns:
        df[c] = df[c] * target                  # df[i, j] = y_i * h_j(x_i)

    return df, stumps, test


# def get_epsilon(data, distributions):
#     epsilons = []
#     for i in range(len(stumps)):
#         prob = 0
#         for j in range(len(data.index)):
#             if data.iloc[j, i] >= stumps[i]:
#                 predict = 1
#             else:
#                 predict = -1
#             if predict != data.iloc[j, -1]:
#                 prob = prob + distributions[j]
#         epsilons.append(prob)
#     return epsilons


def ada_boost(df, T):                       # df[i, j] = y_ih_j(x_i)
    m = len(df.index)
    D = pd.Series([1/m] * m)
    alpha = [0] * len(df.columns)
    ans = []
    for t in range(1, T + 1):
        # print(D)
        error = (df - 1) / (-2)             # error[i, j] = 1 if h_j(x_i) error, else 0
        error = error.mul(D, axis=0)
        print(error.sum().min())
        epsilons = error.sum().tolist()
        # print(epsilons)

        k = 0
        e = epsilons[0]
        for i in range(len(epsilons)):
            if epsilons[i] < e:
                k = i
                e = epsilons[i]
        # print(k)
        # print(e)
        alpha_t = math.log((1 - e) / e)/2
        Z_t = 2 * math.sqrt(e * (1 - e))
        print(t)
        print(alpha)
        temp_D = D.tolist()
        for i in range(m):
            temp_D[i] = (temp_D[i] * math.exp(-alpha_t * df.iloc[i, k])) / Z_t
        D = pd.Series(temp_D)
        alpha[k] += alpha_t
        if t % 10 == 0:
            ans.append(alpha.copy())
    return ans


def logistic_boost(df, T):
    m = len(df.index)
    D = pd.Series([1 / m] * m)
    alpha = [0] * len(df.columns)
    ans = []

    for t in range(1, T + 1):
        print(t)
        # print(D)
        error = (df - 1) / (-2)                 # error[i, j] = 1 if h_j(x_i) error, else 0
        error = error.mul(D, axis=0)
        print(error.sum().min())
        epsilons = error.sum().tolist()

        k = 0
        e = epsilons[0]
        for i in range(len(epsilons)):
            if epsilons[i] < e:
                k = i
                e = epsilons[i]
        alpha_t = math.log((1 - e) / e) / 2
        alpha[k] = alpha[k] + alpha_t

        Z_t = 0
        temp_D = D.tolist()
        for i in range(m):
            g = (df.loc[i].values * alpha).sum()      # sum of alpha_j h_j(x_i)y_i
            temp_D[i] = 1 / (1 + math.exp(g)) / math.log(2)
            Z_t = Z_t + temp_D[i]
        temp_D = [x / Z_t for x in temp_D]
        D = pd.Series(temp_D)
        print(alpha)
        if t % 10 == 0:
            ans.append(alpha.copy())
    return ans


def boosting_test(test, stumps, alpha):
    test_target = test.iloc[:, -1]
    test_data = test.iloc[:, :-1]

    # print(test_target)
    # print(test_data)

    df = test_data.copy()

    for i in range(len(test_data.columns)):
        c = test_data.columns[i]
        df.loc[df[c] <= stumps[c], c] = 'a'
        df.loc[df[c] != 'a', c] = -alpha[i]
        df.loc[df[c] == 'a', c] = alpha[i]
    df["res"] = df.sum(axis=1)
    # print(df)

    temp = df["res"] * test_target
    # print(temp)
    rate = temp.loc[temp > 0].count() / temp.count()
    return rate


# data, stumps = get_stumps(filename)
# print('s')
# alpha, stumps =  ada_boost(data, stumps, 100)
data, stumps, test = data_arrange(filename)

alpha = ada_boost(data, 100)
correct_rates = []
for a in alpha:
    correct_rates.append(boosting_test(test, stumps, a))
print(correct_rates)

beta = logistic_boost(data, 100)
correct_rates = []
for b in beta:
    correct_rates.append(boosting_test(test, stumps, b))
print(correct_rates)
