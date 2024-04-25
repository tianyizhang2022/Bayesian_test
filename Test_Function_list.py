import numpy as np
import math


def Test_Function( Function_name, x,dimension):
    """

    :param Function_name: 獲得関数
    :param x: 探索点
    :param dimension: 次元数
    :return: 獲得関数値
    """
    dim = dimension
    if Function_name == "Rastrigin":
        return Test_rastrigin(x,dimension=dim)
    if Function_name == "sphere":
        return Test_sphere(x)
    if Function_name == "ackley":
        return Test_Ackley(x)
    if Function_name == "Schwefel":
        return Test_Schwefel(x)
    if Function_name == "Rosenbrock":
        return Test_Rosenbrock(x)
    if Function_name == "Boothfunction":
        return Test_Boothfunction(x)
    if Function_name == "Griewangk":
        return Test_Griewangk(x)


def getY_onedimension(x):  # 改成二元方程 y=x1+x2的形式
    y = 0
    if x > 0 and x < math.pi:
        y = 2 * math.sin(x)
    if x > math.pi and x < 2 * math.pi:
        y = -math.sin(x)
    if x > 2 * math.pi:
        y = 3 * math.sin(x)
    return y


def Test_rastrigin(x,dimension):
    A = 10
    sum = dimension * A
    for i in x:
        sum = sum + (i ** 2 - A * np.cos(2 * np.pi * i))
    return sum


def Test_sphere(x):
    y = 0
    for i in x:
        y = y + i * i
    return y


def Test_Ackley(x):
    sigma1 = 0
    sigma2 = 0
    t1 = 20
    for i in x:
        sigma1 = (i ** 2) + sigma1
    t2 = -20 * np.exp(-0.2 * np.sqrt((1/len(x))*sigma1))
    t3 = np.e
    for i in x:
        sigma2 = np.cos(2 * np.pi * i) + sigma2
    t4 = -np.exp((1/len(x)) * sigma2)
    y = t1 + t2 + t3 + t4
    return y


def Test_Schwefel(x):
    s = 0
    for i in x:
        s = i * math.sin(math.sqrt(abs(i))) + s
    y = 418.9829 * len(x) - s
    return y


def Test_Rosenbrock(x):
    s = 0
    for i in range(len(x) - 1):
        s += 100 * math.pow(x[i + 1] - x[i] * x[i], 2) + math.pow(x[i] - 1, 2)
    return s

def Test_Griewangk(x):
    n = len(x)
    sum_term = sum([(xi ** 2) / 4000 for xi in x])
    product_term = math.prod([math.cos(xi / math.sqrt(i + 1)) for i, xi in enumerate(x)])
    return sum_term - product_term + 1

def Test_Boothfunction(x):
    x0 = x[0]
    x1 = x[1]
    y = math.pow((x0 + 2 * x1 - 7), 2) + math.pow((2 * x0 + x1 - 5), 2)
    return y


def getY_twodimension2(x):  # 测试函数1
    x1 = x[0]
    x2 = x[1]
    y = 0.5 + ((math.sin((x1 ** 2 + x2 ** 2) ** 0.5)) ** 2 - 0.5) / (1 + 0.001 * (x1 ** 2 + x2 ** 2)) ** 2
    return y