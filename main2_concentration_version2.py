import numpy as np
from scipy.stats import norm
import math
import random as rd
import sys
import C_LCB_concentrated
import tongji
import picture
import Computing_tool
import Test_Function_list
import Initialization
import k_mean
import test_ot

class bayesian():

    def __init__(self, acq, dimension,determinefunction,whether_local_error):
        """
                コンストラクタ　ベイズ最適化を行う

                Args:
                    acq (str): 獲得関数
                    n, (int): 初期探索点の数
                    dimension (int): 次元数
                    determinefunction(str):目標テスト関数
                    whether_local_error(bool):局所最適解を計算しますか。
                """
        self.n = 5  # int 初期探索点の数
        self.trans_dimension = dimension  # int 次元数
        self.num_search = 100  # int 全局近似関数誤差を計算するため選択するサンプル点数
        self.num_local_error_samplepoint = 10  # int 局所近似関数誤差を計算するため生成するサンプル点数
        self.determinefunction = determinefunction  # str 目標テスト関数
        self.loop_timer = 1  # 評価回数
        self.acq = acq #獲得関数
        self.A_LCB_K = [2.0]  # float(list) A_LCBのパラメータkの変化状況list
        self.C_LCB_K = [4.0]  # float(list) C_LCBのパラメータkの変化状況list
        self.GP_LCB_K = []  # float(list) GP_LCBのパラメータkの変化状況list

        self.lower_limit = []  # float(list) 目標テスト関数設計変数の下界値
        self.upper_limit = []  # float(list) 目標テスト関数設計変数の上界値
        self.l1 = np.zeros([self.n, self.trans_dimension]) #ndarray,探索点のlist
        self.l2 = []  # list 探索点の目標テスト関数値
        self.Initialization_tuple = Initialization.constrant_initialization(self.determinefunction,self.trans_dimension)
        self.lower_limit, self.upper_limit, self.l1, self.l3 = self.Initialization_tuple[0], self.Initialization_tuple[
            1], self.Initialization_tuple[2], self.Initialization_tuple[3]

        self.new_point_list = []# 新しい探索点的list
        self.new_point_list.append(self.l3[-1])

        self.mean = np.zeros(self.num_search)  # ndarray,全局探索点的均值
        self.sigma = np.zeros(self.num_search)  # ndarray,全局探索点的標準偏差
        self.mean1 = np.zeros(self.num_search)  # ndarray,探索点集中空間の均值
        self.sigma1 = np.zeros(self.num_search)  # ndarray,探索点集中空間の標準偏差
        self.mean2 = np.zeros(self.num_search)  # ndarray,探索点离散空間の均值
        self.sigma2 = np.zeros(self.num_search)  # ndarray,探索点离散空間の標準偏差
        self.whether_local_error = whether_local_error
        #self.start_compute_localerror = start_compute_localerror
        self.a,self.b = self.Ponit_List_Format() # 探索点のndarrayのformatを調整する
        #self.a: ndarray,全部探索点のlist
        #self.b: ndarray,全部探索点の目標テスト関数値lsit
        self.Statistics_data() #実験統計データ



    # 探索点のndarrayのformatを調整する
    def Ponit_List_Format(self):
        """

        :return:
        point_list(ndarray): 探索点のlist
        point_value_list(ndarray):探索点の目標テスト関数値

        """
        for x in self.l1:
            s = Test_Function_list.Test_Function(self.determinefunction, x, self.trans_dimension)
            self.l2.append(s)  # 通过L1中存储的自变量，计算因变量Y，储存于L2中

        self.initialization_point_list = self.l1  # 重新排列初始的测量点X值的结构
        self.initialization_values_list = np.asarray(self.l2)  # 重新排列初始的测量点的Y值的结构
        point_list = np.empty((0, self.initialization_point_list.shape[1]), dtype=self.initialization_point_list.dtype)
        point_value_list = np.array([])
        point_list = np.append(point_list, [self.initialization_point_list[0, :]], axis=0)  # 将初始探索点列表中的第一个元素加入self.a中
        point_value_list = np.append(point_value_list, self.initialization_values_list[0])  # 将初始探索点列表中的第一个元素对应的目标函数值加入self.b中
        self.initialization_point_list = self.initialization_point_list[1:, :]  # 将已经加入self.a的元素从初始探索点列表中删除
        self.initialization_values_list = np.delete(self.initialization_values_list, 0)  # 将已经加入self.b的目标函数值从初始列表中删除

        point_list = point_list.reshape(1, self.trans_dimension)  # ndarray,探索点のlist
        point_value_list = point_value_list.reshape(1, 1)  # ndarray 探索点の目標テスト関数値

        return point_list,point_value_list

    # 実験統計データ
    def Statistics_data(self):
        self.ybest = min(self.b)[0]  # 目の前の最適解
        self.ybest_list = []
        self.ybest_list.append(self.ybest)  # 最適解のlist
        self.acqlist = np.zeros(self.num_search)  # 全部サンプル点
        self.differ_surrogate_and_reality1 = []  # 近似関数絶対誤差のlist
        self.differ_surrogate_and_relative = []  # 近似関数相对誤差のlist
        self.density_list1 = []  # 计算取值范围的总密度
        self.density_list2 = []  # 将取值范围分成一个个小块，计算为一个小块的密度
        self.max_reality_deviation = 0  #評価回数=1時、近似関数絶対誤差
        self.max_relative_deviation = 0 #評価回数=1時、近似関数相对誤差
        self.reality_deviation_rate = []  # 絶対誤差变化率
        self.relative_deviation_rate = []  # 相対误差变化率
        self.K_distance_variance = [] #集中度のlist
        self.x_search1 = np.zeros([self.num_search, self.trans_dimension])  # ndarray 全局近似関数誤差を計算するため選択するサンプル点list

        #以下は局所近似関数誤差についてのデータ
        self.clustered_points_list = []  # 局所近似関数誤差を計算する時、サンプル点の集合のlist
        self.discrete_points_list = []
        self.clustered_points_index = []  # 在计算局部误差时的点簇中的每个点在原测试集中的序号
        self.concentration_cluster_ave_deviation = []  # 集中空間なかの平均误差
        self.unconcentration_points_deviation = []  # 离散空間的平均误差

    # カウス過程回帰
    def GPR(self):
        self.x_search1 = self.sampling("lhs", self.num_search, self.trans_dimension, self.lower_limit,self.upper_limit)
        self.mean,self.sigma = test_ot.ot_gp(self.a, self.b, self.x_search1, self.trans_dimension)

        #探索点集中空間と離散空間の近似関数誤差を計算する
        if  self.whether_local_error:
            self.clustered_points_list, self.discrete_points_list = k_mean.kmeans_clustering(data=self.a,
                                                                                             max_cluster_radius=0.5,
                                                                                             num_points_to_generate_per_cluster=self.num_local_error_samplepoint)
            self.mean1, self.sigma1 = test_ot.ot_gp(self.a, self.b, self.x_search1,
                                                           self.trans_dimension)  # 计算点集中区域的均值方差
            self.mean2, self.sigma2 = test_ot.ot_gp(self.a, self.b, self.x_search1,
                                                           self.trans_dimension)  # 计算点离散区域的均值方差
            concentration_ave_error = 0
            unconcentration_ave_error = 0
            for i, j in zip(self.mean1, self.clustered_points_list):
                concentration_ave_error = concentration_ave_error + math.fabs(Test_Function_list.Test_Function(self.determinefunction, j, self.trans_dimension) - i)
            concentration_ave_error = concentration_ave_error / len(self.clustered_points_list)
            self.concentration_cluster_ave_deviation.append(concentration_ave_error)
            for i, j in zip(self.mean2, self.discrete_points_list):
                unconcentration_ave_error = unconcentration_ave_error + math.fabs(Test_Function_list.Test_Function(self.determinefunction, j, self.trans_dimension) - i)
            unconcentration_ave_error = unconcentration_ave_error / len(self.discrete_points_list)
            self.unconcentration_points_deviation.append(unconcentration_ave_error)



    def FindNewPoint(self):  #次の探索点を決める
        if  self.initialization_point_list.any():                             #先将初始探索点遍历一遍，之后再通过获得函数找新的探索点
            self.a = np.append(self.a, [self.initialization_point_list[0, :]], axis=0) #将初始探索点列表中的下一个元素加入self.a中
            self.b = np.append(self.b, self.initialization_values_list[0]) #将对应的目标函数值加入self.b中
            self.initialization_point_list = self.initialization_point_list[1:, :] #删除已经加入self.a中的这个自变量元素
            self.initialization_values_list = np.delete(self.initialization_values_list, 0)#删除这个初始目标函数值

        else:   #次の探索点を決める前に、獲得関数のパラメータを調整する
            if self.acq == "A_LCB":
                self.Approximation_UCB_parameter()
            if self.acq == "C_LCB":
                self.concentration_score_parameter()
            if self.acq == "GP_LCB":
                self.GP_LCB_parameter()

            for i in range(self.num_search):#獲得関数を計算する
                self.acqlist[i] = self.acquisition_function(self.acq, self.mean[i], self.sigma[i], self.ybest)
            maxindex = 0
            if self.acq == "GP_LCB" or self.acq == "LCB" or self.acq == "A_LCB" or self.acq == "C_LCB":
                maxindex = np.argmin(self.acqlist)
            if self.acq == "PI" or self.acq == "EI" or self.acq == "UCB":
                maxindex = np.argmax(self.acqlist)

            new_point = self.x_search1[maxindex] #新しい探索点

            # 新しい探索点を探索点のlist self.aに加入する
            self.new_point_list.append(list(new_point))
            self.a = np.append(self.a, [new_point], axis=0)  # self.c[minindex]是float类型的数字

            # 新しい探索点的目標テスト関数値を計算して，目標テスト関数値のlist self.bに加入する
            new_point_value = Test_Function_list.Test_Function(self.determinefunction, new_point, self.trans_dimension)
            self.b = np.append(self.b, [new_point_value])
        self.Renew_Data() #様々な統計データを更新


    def Renew_Data(self):  # 統計データ更新

        self.a = self.a.reshape(len(self.a), self.trans_dimension)
        self.b = self.b.reshape(len(self.b), 1)

        # 目標テスト関数的最適解更新
        self.ybest = min(self.b)[0]
        if self.determinefunction == "Rosenbrock":
            self.ybest_list.append(math.log(self.ybest))
        else:
            self.ybest_list.append(self.ybest)

        self.loop_timer = self.loop_timer + 1  # 将評価回数+1
        if self.initialization_values_list.size == 0:
            self.n = self.n + 1


    def Statistics(self, d: bool, Concentration: bool, deviation: bool):
        """

        :param d: 探索点の密度を統計する/しない
        :param Concentration: 探索点分布情况(集中度)を統計する/しない
        :param deviation: 近似関数誤差を統計する/しない
        :return:
        """

        if d == True:
            # 探索点の密度を計算する
            density_list = Computing_tool.Finding_Point_Density1(self.lower_limit, self.upper_limit, len(self.a))
            self.density_list1.append(density_list)

        if Concentration == True and self.loop_timer>=4: # 探索点分布情况(集中度)を計算する

            self.K_distance_variance.append(C_LCB_concentrated.evaluate_concentration_multi_dimension2(self.a, 4))

        if deviation == True:# 近似関数誤差を計算する

            # 绝对误差
            reality_deviation_value = 0
            for i, j in zip(self.mean, self.x_search1):
                reality_value1 = Test_Function_list.Test_Function(self.determinefunction, j,
                                                                  self.trans_dimension)  # 真实值
                reality_deviation_value = reality_deviation_value + math.fabs(reality_value1 - i)  # 计算每一个样本点的误差之和
            reality_deviation_value = reality_deviation_value / self.num_search  # 然后再取均值
            if self.loop_timer == 1:
                self.max_reality_deviation = reality_deviation_value
                # print(self.reality_deviation_rate[-1])
            if self.differ_surrogate_and_reality1:
                self.reality_deviation_rate.append((reality_deviation_value / self.max_reality_deviation)-self.differ_surrogate_and_reality1[-1])
            self.differ_surrogate_and_reality1.append(reality_deviation_value / self.max_reality_deviation)

            # 相对误差
            relative_deviation_value = 0
            for a, b, c in zip(self.mean, self.sigma, self.x_search1):
                lower = a - b
                upper = a + b
                reality_value2 = Test_Function_list.Test_Function(self.determinefunction, c, self.trans_dimension)
                if reality_value2 >= lower and reality_value2 <= upper:
                    relative_deviation_value = relative_deviation_value + 0
                if reality_value2 < lower:
                    relative_deviation_value = relative_deviation_value + (lower - reality_value2)
                if reality_value2 > upper:
                    relative_deviation_value = relative_deviation_value + (reality_value2 - upper)
            relative_deviation_value = relative_deviation_value / self.num_search
            if self.loop_timer == 1:
                self.max_relative_deviation = relative_deviation_value
            self.differ_surrogate_and_relative.append(relative_deviation_value / self.max_relative_deviation)

            #探索点集中空間と離散空間の近似関数誤差を計算する
            if self.whether_local_error:
                concentration_ave_error = 0
                unconcentration_ave_error = 0
                for i,j in zip(self.mean1,self.clustered_points_list):
                    concentration_ave_error = concentration_ave_error + math.fabs(Test_Function_list.Test_Function(self.determinefunction,j,self.trans_dimension)-i)
                for q,w in zip(self.mean2,self.discrete_points_list):
                    unconcentration_ave_error = unconcentration_ave_error + math.fabs(Test_Function_list.Test_Function(self.determinefunction,w,self.trans_dimension)-q)
                concentration_ave_error = concentration_ave_error/len(self.clustered_points_list)
                unconcentration_ave_error = unconcentration_ave_error/len(self.discrete_points_list)
                self.concentration_cluster_ave_deviation.append(concentration_ave_error)
                self.unconcentration_points_deviation.append(unconcentration_ave_error)
                #pdb.set_trace()


    def acquisition_function(self, acq, mean, sigma, y_min):
        """
        獲得関数

        Args:
            acq (str): 獲得関数名
            mean (float): 平均値
            sigma (float): 分散
            best (float): 現在の最適値

        Returns:
           獲得関数の計算結果を返す

        """

        if acq == "PI":
            m = mean
            s = sigma
            if s == 0:
                s = 0.0001
            z = (y_min - m - 0.001) / s
            cdf = norm.cdf(z)
            PI = cdf  # 累计正态分布的Y值，可以衡量x处的y值小于目前最小值的概率, 已知正态分布函数曲线和x值，求函数x点左侧积分
            return PI

        elif acq == "EI":
            m = mean
            s = sigma
            if s == 0:
                s = 0.0001
            z = (y_min - m - 1) / s
            EI = (y_min - m - 1) * norm.cdf(z) + s * norm.pdf(z)  # pdf 相当于已知正态分布函数曲线和x值，求y值
            return EI

        elif acq == "LCB":
            k = 2
            lcb = mean - k * sigma
            return lcb

        elif acq == "UCB":
            k = 0.5
            ucb = mean + k * sigma
            return ucb

        elif acq == "GP_LCB":
            delta = 0.05
            nu = 0.5
            anti = self.loop_timer ** (self.trans_dimension / 2 + 2) * (np.pi ** 2) / (3 * delta)  # e是迭代回数
            beta = 2 * nu * math.log(anti)

            gp_lcb = mean - math.sqrt(beta) * sigma
            return gp_lcb

        elif acq == "A_LCB":

            Approximation_UCB = mean - self.A_LCB_K[-1] * sigma
            # pdb.set_trace()
            return Approximation_UCB

        elif acq == "C_LCB":
            original_concentration_LCB = mean - self.C_LCB_K[-1] * sigma
            return original_concentration_LCB


    # 以下は獲得関数パラメータの調整数式

    def GP_LCB_parameter(self): #GP_LCB
        delta = 0.05
        nu = 0.5
        anti = self.loop_timer ** (self.trans_dimension / 2 + 2) * (np.pi ** 2) / (3 * delta)  # e是迭代回数
        beta = 2 * nu * math.log(anti)
        self.GP_LCB_K.append(math.sqrt(beta))

    def Approximation_UCB_parameter(self):#A_LCB

        sigma = 0.03
        if self.differ_surrogate_and_reality1:
            if self.differ_surrogate_and_reality1[-1] >= 1:
                beta = math.tanh(self.differ_surrogate_and_reality1[-1] + sigma-1) + 1
            else:
                beta = math.tanh(self.differ_surrogate_and_reality1[-1] - sigma-1) + 1
            self.A_LCB_K.append(self.A_LCB_K[0] * beta)
            # print(beta)
            # print(math.exp(beta))
            # print(self.A_LCB_K)
            # print("\n")

    def concentration_score_parameter(self):  # C_LCB
        alpha = 1
        if self.K_distance_variance:
            beta = -(alpha / self.K_distance_variance[-1])
            self.C_LCB_K.append(self.C_LCB_K[-1] * math.exp(beta))


    # 采样
    def sampling(self, init, num, dimension, lower, upper):
        """
        初期化生成を行う

        Args:
            init (str): 初期化方法
            num (int): 初期解の数
            dimension (int): 次元数
            lower (np.array[float]): 設計変数の下界値 每一个自变量的下边界
            upper (np.array[float]): 設計変数の上界値 每一个自变量的上边界

        Returns:
            pop (np.ndarray[float]): 初期解

        """

        if init == "random":
            pop = self.random(num, dimension, lower, upper)

        elif init == "lhs":
            pop = self.lhs(num, dimension, lower, upper)

        else:
            print("その初期化方法は存在しないか設定されていません")
            sys.exit(num, dimension, lower, upper)

        return pop

    def random(self, num, dimension, lower, upper):
        """
        ランダムで初期解を生成

        Args:
            num (int): 初期解の数
            dimension (int): 次元数
            lower (np.array[float]): 設計変数の下界値 每一个自变量的下边界
            upper (np.array[float]): 設計変数の上界値 每一个自变量的上边距

        Returns:
            pop (np.ndarray[float]): 初期解
        """
        random_points = []

        for _ in range(num):
            point = tuple(rd.uniform(lower[i], upper[i]) for i in range(dimension))
            random_points.append(point)

        return random_points

    def lhs(self, num, dimension, lower, upper):
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


