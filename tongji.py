import main
import numpy as np

class computing():
    """
    複数な試行回数の平均値を計算する

    """
    def __init__(self,test1):
        self.computing_object = []
        self.computing_object=test1
        self.sample_number=len(self.computing_object)

    def computing_ybest(self):         #最適解
        sum_ybest = self.computing_object[0].ybest_list
        for a in range(1,self.sample_number):
            sum_ybest = np.sum([self.computing_object[a].ybest_list, sum_ybest], axis=0)
        ave_ybest = sum_ybest / self.sample_number
        return ave_ybest



    def computing_sum_density(self): #密度
        sum_density = self.computing_object[0].density_list1
        for a in range(1,self.sample_number):
            sum_density = np.sum([self.computing_object[a].density_list1, sum_density], axis=0)
        ave_density = sum_density/self.sample_number
        return ave_density

    def computing_sum_K_distance_variance(self): #集中度
        sum_K_distance_variance = self.computing_object[0].K_distance_variance
        for a in range(1,self.sample_number):
            sum_K_distance_variance = np.sum([self.computing_object[a].K_distance_variance, sum_K_distance_variance], axis=0)
        ave_K_distance_variance = sum_K_distance_variance/self.sample_number
        return ave_K_distance_variance

    def computing_sum_Approximation_Functon1(self): #近似関数绝对精度
        sum_Approximation_Functon = self.computing_object[0].differ_surrogate_and_reality1
        for a in range(1,self.sample_number):
            sum_Approximation_Functon = np.sum([self.computing_object[a].differ_surrogate_and_reality1,sum_Approximation_Functon ], axis=0)
        ave_Approximation_Functon = sum_Approximation_Functon/self.sample_number
        return ave_Approximation_Functon

    def computing_sum_Approximation_Functon2(self): #近似関数相对精度
        sum_Approximation_Functon = self.computing_object[0].differ_surrogate_and_relative
        for a in range(1,self.sample_number):
            sum_Approximation_Functon = np.sum([self.computing_object[a].differ_surrogate_and_relative, sum_Approximation_Functon], axis=0)
        ave_Approximation_Functon = sum_Approximation_Functon/self.sample_number
        return ave_Approximation_Functon

    def computing_sum_concentration_deviation(self): #探索点集中区域の误差
        concentration_deviation = self.computing_object[0].concentration_cluster_ave_deviation
        for a in range(self.sample_number):
            concentration_deviation = np.sum([self.computing_object[a].concentration_cluster_ave_deviation,concentration_deviation ], axis=0)
        ave_concentration_deviation = concentration_deviation / self.sample_number
        return ave_concentration_deviation

    def computing_sum_unconcentration_deviation(self): #探索点离散区域の误差
        unconcentration_deviation = self.computing_object[0].unconcentration_points_deviation
        for a in range(self.sample_number - 1):
            unconcentration_deviation = np.sum([self.computing_object[a].unconcentration_points_deviation,unconcentration_deviation ], axis=0)
        ave_unconcentration_deviation = unconcentration_deviation / self.sample_number
        return ave_unconcentration_deviation


    def computing_sum_distance(self):   #计算平均点之间的距离
        sum_Points_distance = self.computing_object[0].Ave_Points_distance
        for a in range(1, self.sample_number):
            sum_Points_distance = np.sum([self.computing_object[a].Ave_Points_distance, sum_Points_distance], axis=0)
        ave_Points_distance = sum_Points_distance/self.sample_number
        return ave_Points_distance





    #パラメータの平均値
    def computing_sum_Approximation_K(self):
        sum_Approximation_K = self.computing_object[0].A_LCB_K
        for a in range(1, self.sample_number):
            sum_Approximation_K = np.sum([self.computing_object[a].A_LCB_K, sum_Approximation_K], axis=0)
        ave_Approximation_K = sum_Approximation_K / self.sample_number
        return ave_Approximation_K

    def computing_sum_variance_K(self):
        sum_variance_K = self.computing_object[0].variance_K
        for a in range(1, self.sample_number):
            sum_variance_K = np.sum([self.computing_object[a].variance_K, sum_variance_K], axis=0)
        ave_variance_K = sum_variance_K / self.sample_number
        return ave_variance_K

    def computing_sum_GPLCB_K(self):
        sum_GPLCB_K = self.computing_object[0].GP_LCB_K
        for a in range(1, self.sample_number):
            sum_GPLCB_K = np.sum([self.computing_object[a].GP_LCB_K, sum_GPLCB_K], axis=0)
        ave_GPLCB_K = sum_GPLCB_K / self.sample_number
        return ave_GPLCB_K

    def computing_sum_CLCB_K(self):
        sum_CLCB_K = self.computing_object[0].C_LCB_K
        for a in range(1, self.sample_number):
            sum_CLCB_K = np.sum([self.computing_object[a].C_LCB_K, sum_CLCB_K], axis=0)
        ave_CLCB_K = sum_CLCB_K / self.sample_number
        return ave_CLCB_K