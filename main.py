import numpy as np
from scipy.stats import norm
import math
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.decomposition import PCA
import random as rd
import sys
import C_LCB_concentrated
import tongji
import picture
import Computing_tool
import Test_Function_list
import Initialization
import pdb
from sklearn.gaussian_process.kernels import Matern

class bayesian():

    def __init__(self,acq):
        self.n = 10  # 选点个数
        self.trans_dimension = 5  # 维度设为2
        self.num_search = 100  # 选择100个样本点
        self.determinefunction = "rastrigin"
        self.lower_limit = []  # 自变量的下边界组
        self.upper_limit = []  # 自变量的上边界组
        self.x_search = np.zeros([self.num_search, self.trans_dimension])  # 计算获得函数的样本点位置
        self.x_search2 = np.zeros([self.num_search, self.trans_dimension])
        self.l1 = np.zeros([self.n, self.trans_dimension])
        self.Initialization_tuple = Initialization.lhs_initialization(self.determinefunction, self.trans_dimension,self.n)
        self.lower_limit, self.upper_limit, self.l1, self.l3 = self.Initialization_tuple[0], self.Initialization_tuple[1], self.Initialization_tuple[2], self.Initialization_tuple[3]
        self.outside_number = 0  # 跳出局部最优解的次数
        self.new_point_list = []
        self.new_point_list.append(self.l3[-1])  # 新的探索点的列表
        self.loop_timer = 1  # 贝叶斯迭代次数，评价次数
        self.Approximation_K = [2.0]
        self.variance_K = [2.0]
        self.concentration_K = [2.0]
        self.GP_LCB_K = []



        #self.random_initialization()

        self.l2 = []  #因变量y
        for x in self.l1:
            s = Test_Function_list.Test_Function(self.determinefunction,x,self.trans_dimension)
            self.l2.append(s)            #通过L1中存储的自变量，计算因变量Y，储存于L2中
        self.a = self.l1.reshape(self.n,self.trans_dimension)  #重新排列初始的测量点X值的结构

        self.b = np.asarray(self.l2).reshape(self.n, 1)   #重新排列初始的测量点的Y值的结构
        self.mean = np.zeros(self.num_search)   #所有样本点的均值
        self.sigma = np.zeros(self.num_search)   #所有样本点的方差
        self.acq = acq
        self.ybest = min(self.b)[0]  #当前最优解
        self.ybest_list = []
        self.ybest_list.append(self.ybest) #每一时刻的最优解的列表
        self.xbest_list = [] #每一时刻的最优解坐标的列表
        self.acqlist = np.zeros(self.num_search)  #所有样本点获得函数值
        self.N = 1  # 记录GPLCB获得函数中的世代数
        #self.xi = 1 #超参数
        self.xbest = np.zeros([1,self.trans_dimension])
        self.differ_surrogate_and_reality1 = []  # 近似函数绝对精度的列表
        self.differ_surrogate_and_reality2 = []  # 近似函数相对精度的列表
        self.x_differ = []
        self.y_up = []
        self.density_list1 = []#计算取值范围的总密度
        self.density_list2 = []#将取值范围分成一个个小块，计算为一个小块的密度
        self.variance = [] #方差的列表
        self.max_reality_deviation = 0
        self.max_relative_deviation = 0
        self.reality_deviation_rate = []#绝对误差变化率
        self.relative_deviation_rate = []#相对误差变化率
        self.concentration_error = [] #2维以上使用点与其他点的距离评估的探索点的集中评分
        self.Ave_Points_distance = []


    def random_initialization(self):
        self.l3 = []
        self.MakeOriginPoint()
        self.l5 = np.concatenate([[self.l3[0]], [self.l3[1]]], axis=0)  #

        for i in range(self.n):
            self.l1[i] = self.l5[:, i]  # 将L5中的数组转置之后填装到L1里，L1储存之后用的初始点的自变量值
        # self.l1 = self.l3.reshape(4,2)

    def MakeOriginPoint(self):

        for i in range(self.trans_dimension):
            a = self.randfloat(self.n, -5, 5)
            self.l3.append(a)

    # 随机取初始值
    def randfloat(self,num, l, h):
        if l > h:
            return None
        else:
            a = h - l
            b = h - a
            out = (np.random.rand(num) * a + b).tolist()
            out = np.array(out)
            return out
   #选择测试函数


    #高斯过程
    def GPR(self):
        kernel = Matern()
        gp = GaussianProcessRegressor(kernel = kernel)

        fitting = gp.fit(self.a,self.b)
        self.x_search = self.sampling("lhs", self.num_search, self.trans_dimension, self.lower_limit,self.upper_limit) #采样，每个方块中选一点计算均值方差
        if self.loop_timer == 1:
            self.x_search2 = self.x_search

        self.mean,self.sigma = gp.predict(self.x_search, return_std=True) #计算均值方差
        self.sigma = self.sigma.reshape(-1, 1)

        #plt.scatter(self.a,self.b, marker='o', color='r', label='3', s=15)
        #plt.plot(c, self.mean)
        #plt.show()

    def FindNewPoint(self):
        if self.acq =="Approximation_LCB":
            self.Approximation_UCB_parameter()
        if self.acq == "variance_LCB":
            self.variance_UCB_parameter()
        if self.acq == "original_concentration_LCB":
            self.concentration_score_parameter()
        for i in range(self.num_search):
           self.acqlist[i] = self.acquisition_function(self.acq, self.mean[i], self.sigma[i], self.ybest)

        if self.acq == "GP_UCB" or self.acq == "LCB" or self.acq == "original_Approximation_LCB" or self.acq == "original_concentration_LCB":
           maxindex = np.argmin(self.acqlist)  # 获得函数极值的下标
        if self.acq == "PI" or self.acq == "EI" or self.acq == "UCB":
           maxindex = np.argmax(self.acqlist)  # 获得函数极值的下标
        # print("AF=")
        # print("\n")
        # print(self.acqlist)
        # print("\n")
        # print(self.acqlist[maxindex])
        new_point = self.x_search[maxindex]
        last_new_point = self.new_point_list[-1]
        last_new_point_value = Test_Function_list.Test_Function(self.determinefunction, last_new_point,self.trans_dimension)
        self.x_differ.append(Computing_tool.Computing_X_distance(self.trans_dimension, last_new_point, new_point))  # 记录每次自变量的跨度

        # 将新找到的探索点加入列表
        self.new_point_list.append(list(new_point))
        self.a = np.append(self.a, [new_point], axis=0)  # self.c[minindex]是float类型的数字

        # 测量这个探索点的函数值，然后将这个函数值加入列表b当中
        new_point_value = Test_Function_list.Test_Function(self.determinefunction, new_point,self.trans_dimension)
        self.b = np.append(self.b, [new_point_value])

    def Statistics(self,d:bool,variance:bool,deviation:bool):

        if d == True:
            # 计算当下的探索点密度
            density_list = Computing_tool.Finding_Point_Density1(self.lower_limit, self.upper_limit, len(self.a))
            self.density_list1.append(density_list)

        if variance == True:
            #计算当下的探索点分布情况
            if self.trans_dimension == 2:
                concentration_score = C_LCB_concentrated.evaluate_concentration(self.a) #area_list是Voronoi图每一小块的面积列表
                self.variance.append(concentration_score)
            else:
                # density_dict = Computing_tool.Finding_Point_Density2(10, self.trans_dimension, self.lower_limit,
                #                                                      self.upper_limit, self.a)
                # self.density_list2.append(density_dict)
                # self.variance.append(Computing_tool.remark_density(density_dict))
                self.Ave_Points_distance.append(C_LCB_concentrated.evaluate_concentration_multi_dimension2(self.a))
                if len(self.Ave_Points_distance)>1:
                    self.concentration_error.append(self.Ave_Points_distance[-1]-self.Ave_Points_distance[-2])

        if deviation == True:
            # 计算这时的近似函数精确度
            # self.differ_surrogate_and_reality.append(
            #     Computing_tool.Computing_Approximation_Functon_Accuracy(self.mean[maxindex], new_point_value))
            # 绝对误差
            reality_deviation_value = 0
            for i, j in zip(self.mean, self.x_search2):
                reality_value1 = Test_Function_list.Test_Function(self.determinefunction, j,self.trans_dimension)
                reality_deviation_value = reality_deviation_value + math.fabs(reality_value1 - i)
            reality_deviation_value = reality_deviation_value / self.num_search
            if  self.loop_timer == 1:
                self.max_reality_deviation = reality_deviation_value
            if  self.loop_timer >1:
                self.reality_deviation_rate.append(self.differ_surrogate_and_reality1[-1]-(reality_deviation_value/self.max_reality_deviation))
                #print(self.reality_deviation_rate[-1])

            self.differ_surrogate_and_reality1.append(reality_deviation_value / self.max_reality_deviation)

            # 相对误差
            relative_deviation_value = 0
            for a, b, c in zip(self.mean, self.sigma, self.x_search2):
                lower = a - b
                upper = a + b
                reality_value2 = Test_Function_list.Test_Function(self.determinefunction, c,self.trans_dimension)
                if reality_value2 >= lower and reality_value2 <= upper:
                    relative_deviation_value = relative_deviation_value + 0
                if reality_value2 < lower:
                    relative_deviation_value = relative_deviation_value + (lower - reality_value2)
                if reality_value2 > upper:
                    relative_deviation_value = relative_deviation_value + (reality_value2 - upper)
            relative_deviation_value = relative_deviation_value / self.num_search
            if self.loop_timer == 1:
                self.max_relative_deviation = relative_deviation_value
            else:
                self.relative_deviation_rate.append(self.differ_surrogate_and_reality2[-1] - (relative_deviation_value / self.max_relative_deviation))
            self.differ_surrogate_and_reality2.append(relative_deviation_value/self.max_relative_deviation)

        # 计算这时的最优解改善量
        self.y_up.append(self.ybest - min(self.b))
        self.n = self.n + 1
        self.a = self.a.reshape(self.n, self.trans_dimension)
        self.b = self.b.reshape(self.n, 1)

        # 更新目标函数的最优解
        self.ybest = min(self.b)[0]
        self.ybest_list.append(self.ybest)
        self.xbest = self.a[np.argmin(self.b)]  # 这里的最优解的坐标与前面的新找到的探索点的坐标列表不一样，因为新找的探索点的函数值未必一定是当前最优
        self.xbest_list.append(self.xbest)
        self.loop_timer = self.loop_timer + 1

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
            z = (y_min - m-0.001) / s
            cdf = norm.cdf(z)
            PI = cdf  # 累计正态分布的Y值，可以衡量x处的y值小于目前最小值的概率, 已知正态分布函数曲线和x值，求函数x点左侧积分
            return PI

        elif acq == "EI":
            m = mean
            s = sigma
            if s == 0:
                s = 0.0001
            z = (y_min-m-1) / s
            EI = (y_min-m-1) * norm.cdf(z) + s * norm.pdf(z)  # pdf 相当于已知正态分布函数曲线和x值，求y值
            return EI

        elif acq == "LCB":
            k = 1
            lcb = mean - k * sigma
            return lcb

        elif acq == "UCB":
            k = 0.5
            ucb = mean + k * sigma
            return ucb

        elif acq == "GP_LCB":
            delta = 0.05
            nu = 0.5
            anti = self.loop_timer ** (self.trans_dimension / 2 + 2) * (np.pi ** 2) / (3 * delta)  #e是迭代回数
            beta = 2 * nu * math.log(anti)
            self.GP_LCB_K.append(math.sqrt(beta))
            gp_lcb = mean - math.sqrt(beta) * sigma
            return gp_lcb

        elif acq == "Approximation_LCB":

            # if self.reality_deviation_rate:
            #     if self.reality_deviation_rate[-1] <= 0.1:
            #         self.Approximation_UCB = self.Approximation_UCB - 0.01
            #     ucb = mean - self.Approximation_UCB * sigma
            # else:
            #     ucb = mean - self.Approximation_UCB * sigma
            Approximation_UCB = mean - self.Approximation_K[-1] * sigma
                #pdb.set_trace()
            return Approximation_UCB

        elif acq == "variance_LCB":
            variance_UCB = mean - self.variance_K[-1] * sigma
            return variance_UCB

        elif acq == "original_concentration_LCB":
            original_concentration_LCB = mean - self.concentration_K[-1] * sigma
            return original_concentration_LCB

    def Approximation_UCB_parameter(self):
        alpha = 4
        if self.reality_deviation_rate:
            beta = -alpha * self.reality_deviation_rate[-1]
            if beta<0:
                self.Approximation_K.append(self.Approximation_K[-1]* math.exp(beta))
            else:
                self.Approximation_K.append(self.Approximation_K[-1])
            # print(beta)
            # print(math.exp(beta))
            # print(self.Approximation_K)
            # print("\n")
    def variance_UCB_parameter(self):
        alpha = 0.2
        if self.reality_deviation_rate:
            beta = -(alpha / self.variance[-1])
            self.variance_K.append(self.variance_K[-1] * math.exp(beta))

    def concentration_score_parameter(self):
        alpha = 0.2
        if self.concentration_error:
            if self.concentration_error[-1]<0:
                beta = alpha * self.concentration_error[-1]
                self.concentration_K.append(self.concentration_K[-1] * math.exp(beta))
            else:
                if self.loop_timer<20:
                    beta = alpha/self.concentration_error[-1]
                    self.concentration_K.append(self.concentration_K[-1] * math.exp(beta))
                else:
                    self.concentration_K.append(self.concentration_K[-1])
    #采样
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
        pop = np.zeros([num, dimension])

        for i in range(num):
            for j in range(dimension):
                pop[i][j] = rd.uniform(lower[j], upper[j])

        return pop

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
        b = upper[0]-lower[0] #各変数の範囲
        m = [[i for i in range(num)] for j in range(dimension)] #分割した時の格子の座標
        s = np.zeros((num, dimension)) #選ばれた格子の座標
        for i in range(dimension):
            rd.shuffle(m[i])   #乱序
        for i in range(num):
            for j in range(dimension):
                s[i][j] = m[j][i]
        for i in range(num):
            for j in range(dimension):
                pop[i][j] = (lower[j]+s[i][j]*b/num)+rd.random()*b/num

        #pdb.set_trace()
        return pop




if __name__ == '__main__':
    list_acq = ["PI","variance_LCB","LCB","Approximation_LCB" ]
    ave = []
    test_timer = 50  # 重复实验次数
    op_timer = 500  # 每次实验进行的贝叶斯优化次数
    d_whether = False  # 是否统计探索点的密度
    variance_whether = True  # 是否统计探索点的分布情况
    Approximation_whether = True  # 是否统计近似函数误差
    drawpicture_order = (d_whether,variance_whether,Approximation_whether)  #作为绘图指令

    for q in list_acq:
        test1 = []

        limit_match_y = 0
        expectdict = {}
        expectdict1 = []  # 优化结果
        expectdict3 = []  # 将expectdict1变回LIST格式
        expectdict4 = []  # 将expectdict3变为int格式
        expectdict2 = {}  # 统计优化结果
        list1 = []  # 绘图X参数
        list2 = []  # 绘图Y参数
        averagevalue = 0  # 平均最优解
        sumvalue = 0
        goodanswernum = 0
        #ave_differ = []  # sum_differ 平均近似函数误差
        ave_y_up = []  # sum_y_up 平均y提升量
        ave_x_differ = []  # 平均自变量跨度
        ave_ybest = []
        ave_density = []  # 平均探索点密度
        ave_variance = []  # 平均方差
        ave_Approximation1 = []  # 平均近似関数绝对精度
        ave_Approximation2 = []  # 平均近似関数相对精度
        ave_Approximation1_rate = []
        ave_list = []

        for j in range(test_timer):  # 重复500次实验 0--499
            test1.append(bayesian(q))
        for j in range(test_timer):
            for i in range(op_timer):  # 每次实验进行7次贝叶斯优化，即再找两个最优解
                test1[j].GPR()
                test1[j].FindNewPoint()
                test1[j].Statistics(d=d_whether, variance=variance_whether, deviation=Approximation_whether)

            # print("differ of 近似関数 " + str(test1[j].differ_surrogate_and_reality))
            expectdict[j] = test1[j].ybest
            # print(test1[j].differ_surrogate_and_reality)
            print("序号" + str(j))
            print("每次选择的点")
            print(test1[j].a)
            # if j==10:
            #    picture.draw_Xpointpicture(test1[j].a,j,q)
            print("最优解")
            print(test1[j].ybest)
            print("最优解的位置")
            print(test1[j].xbest)
            print("\n")
            print(test1[j].differ_surrogate_and_reality1)

        # for a in range(499):
        #     sum_differ = np.sum([test1[a].differ_surrogate_and_reality, test1[a + 1].differ_surrogate_and_reality], axis=0)  # sum_differ 平均近似函数误差
        #     test1[a + 1].differ_surrogate_and_reality = sum_differ
        #     sum_y_up = np.sum([test1[a].y_up, test1[a + 1].y_up], axis=0)  # sum_y_up 平均y提升量
        #     test1[a + 1].y_up = sum_y_up
        #     sum_x_differ = np.sum([test1[a].x_differ, test1[a + 1].x_differ], axis=0)  # 平均自变量跨度
        #     test1[a + 1].x_differ = sum_x_differ
        #     sum_ybest = np.sum([test1[a].ybest_list, test1[a + 1].ybest_list], axis=0)
        #     test1[a + 1].ybest_list = sum_ybest

        # sum_differ = sum_differ / 500
        # sum_y_up = sum_y_up / 500
        # sum_x_differ = sum_x_differ / 500
        # sum_ybest = sum_ybest / 500
        print(test1[0].acq)
        sample1 = tongji.computing(test1)
        #ave_differ = sample1.computing_sum_differ()  # 近似函数精度
        ave_y_up = sample1.computing_sum_y_up()  # 改善量
        ave_x_differ = sample1.computing_sum_x_differ()  #
        ave_ybest = sample1.computing_ybest()  # 最优解
        ave_density = sample1.computing_sum_density()  # 探索点密度
        ave_variance = sample1.computing_sum_variance()  # 探索点分布情况，方差
        ave_Approximation1 = sample1.computing_sum_Approximation_Functon1()  # 近似函数绝对精度  和前面的ave_differ是一个东西
        ave_Approximation2 = sample1.computing_sum_Approximation_Functon2()  # 近似函数相对精度
        ave_Approximation1_rate = sample1.computing_sum_Approximation_reality_rate1()
        ave_Approximation2_rate = sample1.computing_sum_Approximation_reality_rate2()
        if q =="Approximation_LCB":
            ave_Approximation_K = sample1.computing_sum_Approximation_K()
        if q =="variance_LCB":
            ave_variance_K = sample1.computing_sum_variance_K()
        if q == "GP_LCB":
            ave_GP_LCB_K = sample1.computing_sum_GPLCB_K()

        # sum_outside_number = sample1.computing_sum_outside_number()

        ave_list.append(ave_y_up)
        ave_list.append(ave_x_differ)
        ave_list.append(ave_ybest)
        ave_list.append(ave_density)
        ave_list.append(ave_variance)
        ave_list.append(ave_Approximation1)
        ave_list.append(ave_Approximation2)
        ave_list.append(ave_Approximation1_rate)
        ave_list.append(ave_Approximation2_rate)
        ave.append(ave_list)

        expectdict1 = expectdict.values()
        print(expectdict1)

        f = open('F:/pycharm/Bayesian_test/graph1/rastrigin/2dim_vo/result.txt', 'a',encoding="UTF-8")
        f.write("acq:" + q + "\n")
        f.write(str(test1[0].Initialization_tuple))
        f.write(str(op_timer))

        f.write("目標関数:" + test1[0].determinefunction + "\n")
        f.write("次元数：" + str(test1[0].trans_dimension) + "次元"+ "\n")
        f.write("重复实验次数"+str(test_timer) + "\n")
        f.write("每次实验进行的贝叶斯优化次数"+str(op_timer))
        f.write('\n' + "ave_differ"+str(ave_Approximation1.tolist()) + "\n")
        f.write('\n' + "ave_y_up" + str(ave_y_up.tolist()) + "\n")
        f.write('\n' + "ave_x_differ"+str(ave_x_differ.tolist()) + "\n")
        f.write('\n' + "ave_ybest"+str(ave_ybest.tolist()) + "\n")

        #f.write('\n' + "sum_outside_number"+str(sum_outside_number) + "\n")
        f.write("\n")
        f.close()


    picture.data_cleaning(ave,op_timer,drawpicture_order)

