import numpy as np
import random as rd
def constrant_initialization(Function_name,trans_dimension):
    """
                   自分で決めるテスト関数初期探索点の初期化

                   Args:
                       Function_name (str): テスト関数
                       dimension (int): 次元数
                       n(int):初期探索点の数

                   Returns:
                       lower (np.array[float]): 目標テスト関数の設計変数の下界値
                       upper (np.array[float]): 目標テスト関数の設計変数の上界値
　　　　　　　　　　　　　  l1,l3: 初期探索点のlist
                   """
    lower_limit = [] #目標テスト関数の設計変数の下界値
    upper_limit = [] #目標テスト関数の設計変数の上界値
    l3 =[]
    if Function_name == "sphere":
        lower_limit = trans_dimension * [-5]
        upper_limit = trans_dimension * [5]
        l3 = [[-4, -4,-4,-4,-4], [-4, 4,-4,4,-4], [4, -4,4,-4,4], [4, 4,4,4,4]]  # !!!!!!!!!!!!!!!次元数を変える時、その初期探索点を変えるべき

    if Function_name == "Rastrigin":
        lower_limit = trans_dimension * [-5.12]
        upper_limit = trans_dimension * [5.12]
        l3 = [[-4]*trans_dimension, [4]*trans_dimension, [4.5]*trans_dimension, [-4.5]*trans_dimension,[4.9]*trans_dimension]  # !!!!!!!!!!!!!!!次元数を変える時、その初期探索点を変えるべき


    if Function_name == "ackley":  # 这个函数只有2维
        lower_limit = trans_dimension * [-5]
        upper_limit = trans_dimension * [5]
        l3 = [[-5, -5], [-5, 5], [5, -5], [5, 5]]  # !!!!!!!!!!!!!!!次元数を変える時、その初期探索点を変えるべき


    if Function_name == "Schwefel":
        lower_limit = trans_dimension * [-500]
        upper_limit = trans_dimension * [500]
        l3 = [[-299, -299,-299,-299,-299], [-299, 299,-299,299,-299], [299, -299,299,-299,299], [299, 299,299,299,299],
              [499,499,-499,-499,-399],[-499,-499,-499,-499,-499],[-499,-499,499,499,499],[-499,499,-499,499,-499],[499,-499,499,-499,499],[499,499,499,499,499]]  # !!!!!!!!!!!!!!!改变维度时需要修改

    if Function_name == "Rosenbrock":  # 取值范围负无穷到正无穷
        lower_limit = trans_dimension * [-10]
        upper_limit = trans_dimension * [10]
        l3 = [[-5]*trans_dimension, [5]*trans_dimension, [8]*trans_dimension, [-8]*trans_dimension,[-9]*trans_dimension]  # !!!!!!!!!!!!!!!改变维度时需要修改

    if Function_name == "Boothfunction":  # 这个函数只有2维
        lower_limit = trans_dimension * [-10]
        upper_limit = trans_dimension * [10]
        l3 = [[-9, -9], [-9, 9], [9, -9], [9, 9]]  # !!!!!!!!!!!!!!!次元数を変える時、その初期探索点を変えるべき

    if Function_name == "Griewangk":
        lower_limit = trans_dimension * [-20]
        upper_limit = trans_dimension * [20]
        l3 = [[19]*trans_dimension, [-17]*trans_dimension, [17]*trans_dimension, [15]*trans_dimension,[-15]*trans_dimension]  # !!!!!!!!!!!!!!!次元数を変える時、その初期探索点を変えるべき

    l1 = np.asarray(l3)

    return lower_limit,upper_limit,l1,l3

def random_initialization(Function_name,trans_dimension,num):
    """
                       ランダムでテスト関数初期探索点の初期化

                       Args:
                           Function_name (str): テスト関数
                           dimension (int): 次元数
                           num(int) : 初期探索点の数

                       Returns:
                           lower (np.array[float]): 目標テスト関数の設計変数の下界値
                           upper (np.array[float]): 目標テスト関数の設計変数の上界値
    　　　　　　　　　　　　　  l1,l3: 初期探索点のlist
                       """
    lower_limit = []
    upper_limit = []
    l3 = []

    if Function_name == "sphere":
        lower_limit = trans_dimension * [-5]
        upper_limit = trans_dimension * [5]
        for i in range(num):
            l3.append(randfloat(trans_dimension,lower_limit[0],upper_limit[0]))# !!!!!!!!!!!!!!!改变维度时需要修改

    if Function_name == "rastrigin":
        lower_limit = trans_dimension * [-5.12]
        upper_limit = trans_dimension * [5.12]
        for i in range(num):
            l3.append(randfloat(trans_dimension, lower_limit[0], upper_limit[0]))  # !!!!!!!!!!!!!!!改变维度时需要修改


    if Function_name == "ackley":  # 这个函数只有2维
        lower_limit = trans_dimension * [-5]
        upper_limit = trans_dimension * [5]
        for i in range(num):
            l3.append(randfloat(trans_dimension, lower_limit[0], upper_limit[0]))  # !!!!!!!!!!!!!!!改变维度时需要修改


    if Function_name == "Schwefel":
        lower_limit = trans_dimension * [-500]
        upper_limit = trans_dimension * [500]
        for i in range(num):
            l3.append(randfloat(trans_dimension, lower_limit[0], upper_limit[0]))  # !!!!!!!!!!!!!!!改变维度时需要修改


    if Function_name == "Rosenbrock":  # 取值范围负无穷到正无穷
        lower_limit = trans_dimension * [-10]
        upper_limit = trans_dimension * [10]
        for i in range(num):
            l3.append(randfloat(trans_dimension, lower_limit[0], upper_limit[0]))  # !!!!!!!!!!!!!!!改变维度时需要修改


    if Function_name == "Boothfunction":  # 这个函数只有2维
        lower_limit = trans_dimension * [-10]
        upper_limit = trans_dimension * [10]
        for i in range(num):
            l3.append(randfloat(trans_dimension, lower_limit[0], upper_limit[0]))  # !!!!!!!!!!!!!!!改变维度时需要修改

    l1 = np.asarray(l3)

    return lower_limit, upper_limit, l1, l3

def lhs_initialization(Function_name,trans_dimension,num):
    """
                           lhsでテスト関数初期探索点の初期化

                           Args:
                               Function_name (str): テスト関数
                               dimension (int): 次元数
                               num(int) : 初期探索点の数

                           Returns:
                               lower (np.array[float]): 目標テスト関数の設計変数の下界値
                               upper (np.array[float]): 目標テスト関数の設計変数の上界値
        　　　　　　　　　　　　　  l1,l3: 初期探索点のlist
                           """
    lower_limit = []
    upper_limit = []
    l3 =[]
    if Function_name == "sphere":

        lower_limit = trans_dimension * [-5]
        upper_limit = trans_dimension * [5]
        l3 = lhs(num,trans_dimension,lower_limit,upper_limit)


    if Function_name == "rastrigin":

        lower_limit = trans_dimension * [-5.12]
        upper_limit = trans_dimension * [5.12]
        l3 = lhs(num,trans_dimension,lower_limit,upper_limit)

    if Function_name == "ackley":  # 这个函数只有2维
        lower_limit = trans_dimension * [-5]
        upper_limit = trans_dimension * [5]
        l3 = lhs(num,trans_dimension,lower_limit,upper_limit)

    if Function_name == "Schwefel":
        lower_limit = trans_dimension * [-500]
        upper_limit = trans_dimension * [500]
        l3 = lhs(num,trans_dimension,lower_limit,upper_limit)

    if Function_name == "Rosenbrock":  # 取值范围负无穷到正无穷
        lower_limit = trans_dimension * [-10]
        upper_limit = trans_dimension * [10]
        l3 = lhs(num,trans_dimension,lower_limit,upper_limit)

    if Function_name == "Boothfunction":  # 这个函数只有2维
        lower_limit = trans_dimension * [-10]
        upper_limit = trans_dimension * [10]
        l3 = lhs(num,trans_dimension,lower_limit,upper_limit)
    l3 = l3.tolist()
    l1 = np.asarray(l3)
    return lower_limit, upper_limit, l1, l3


def lhs(num, dimension, lower, upper):
    """
     LHSで初期解を生成

            Args:
                num (int): 初期解の数
                dimension (int): 次元数
                lower (np.array[float]): 設計変数の下界値
                upper (np.array[float]): 設計変数の上界値

            Returns:
                pop (np.ndarray[float]): 初期解
    """
    pop = np.zeros([num, dimension])
    b = upper[0] - lower[0]  # 各変数の範囲
    m = [[i for i in range(num)] for j in range(dimension)]  # 分割した時の格子の座標
    s = np.zeros((num, dimension))  # 選ばれた格子の座標
    for i in range(dimension):
        rd.shuffle(m[i])  # 乱序
    for i in range(num):
        for j in range(dimension):
            s[i][j] = m[j][i]
    for i in range(num):
        for j in range(dimension):
            pop[i][j] = (lower[j] + s[i][j] * b / num) + rd.random() * b / num

    # pdb.set_trace()
    return pop

def randfloat(num, l, h):
    if l > h:
        return None
    else:
        a = h - l
        b = h - a
        out = (np.random.rand(num) * a + b).tolist()
        return out