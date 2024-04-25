import math
from scipy.spatial import KDTree
import numpy as np



def Computing_Approximation_Functon_Accuracy(Approximation_value,new_piont_value):
    return (Approximation_value-new_piont_value)


def Computing_X_distance(dimension,last_new_point,this_new_point):
    distance = 0
    for i in range(dimension):
        # distance = distance +math.pow(self.a[np.argmin(self.b)][i]-new_point[i],2)
        distance = distance + math.pow(last_new_point[i] - this_new_point[i], 2)
    distance = math.sqrt(distance)
    return distance


def Finding_Point_Density1(X_lower_limit,X_upper_limit,Point_number):
    X_list = []   #所有自变量的"边长列表"
    for i,j in zip(X_lower_limit,X_upper_limit):
        X_list.append(j-i)
    Volume = 1
    for i in range(len(X_list)):
        Volume=X_list[i]*Volume
    Density = Point_number/Volume #总密度
    return Density

def Finding_Point_Density2(side_Number,dimension,X_lower_limit,X_upper_limit,Sample_points_list): #将全部区域分成一个个小块，分别计算每一块的密度  side_Number一个维度上的坐标点数量
    start = X_lower_limit[0] #起始区间
    end = X_upper_limit[0] #终止区间
    side_length = (end-start)/side_Number #一个维度上的边长
    points = np.linspace(start,end-side_length,side_Number)
    points1 = np.meshgrid(*[points]*dimension)
    points2 = np.stack(points1,axis=-1).reshape(-1,dimension)
    points2 = tuple(points2)
    points3 = []
    for i in points2:
        points3.append(tuple(i))
    standard_point_dict = dict.fromkeys(points3,0)
    type(standard_point_dict)
    for Sample_point in Sample_points_list:
        index_list = []  #这个是样本点所在方块的序号，
        standard_point_position = []
        for x in Sample_point:
            index = math.floor(x/side_length)
            index_list.append(index)
        for index in index_list:
            standard_point_position.append(index*side_length)   #standard_point_position是该样本点所在方块的左下角的那个点的坐标
        standard_point_position = tuple(standard_point_position)
        standard_point_dict[standard_point_position] =standard_point_dict.get(standard_point_position)+1
    area_point_number = standard_point_dict.values()
    area_point_density = [i/pow(side_length,dimension) for i in area_point_number]
    for i ,key in enumerate(standard_point_dict.keys()):
        standard_point_dict[key] = area_point_density[i]



    return standard_point_dict

def remark_density(density_dict):
    unergodic_area_number = 0  #未遍历到的区域的数量
    for i in density_dict.values():
        if i == 0:
            unergodic_area_number = unergodic_area_number + 1

    #这里算一个每个区域密度数据的方差
    sum_density = 0
    for i in density_dict.values():
        sum_density = sum_density + i
    ave_density = sum_density/len(density_dict.values()) #平均值
    variance = 0 #方差
    for i in density_dict.values():
        variance = pow((i-ave_density),2)+variance
    return variance

def Computing_Variance(areas_list):
    sum_area = 0
    for i in areas_list:
        sum_area = i + sum_area
    ave_area = sum_area/len(areas_list)
    variance = 0
    for j in areas_list:
        variance = variance + pow((j-ave_area),2)
    variance = math.sqrt(variance)
    return variance

def calculate_total_distance(points):
    tree = KDTree(points)
    distances = tree.query(points, k=len(points))
    total_distance = np.sum(distances[0],axis=1)  # Sum all distances except the first (self-distance)
    ave_distance = np.mean(total_distance)
    return ave_distance
