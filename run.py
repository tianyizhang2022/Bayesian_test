import main2_concentration_version2
import picture
import tongji
import os
def read_input():

    # 获取当前文件的路径
    current_path = os.path.abspath(__file__)
    # 获取当前文件的上一级路径
    parent_path = os.path.dirname(current_path)
    # 要获取的文件名
    file_name = "input.txt"

    # 构建要获取的文件的路径
    file_path = os.path.join(parent_path, file_name)

    input_data_list =[]
    with open(file_path, "r", encoding='utf-8') as f:
        # readlines()：读取文件全部内容，以列表形式返回结果
        data = f.readlines()
        for item in data:
            if item[-1] =="\n":
                item = item[:-1]

            index = item.index(":")
            input_data = item[index+1:]
            input_data_list.append(input_data)

    a = int(input_data_list[0])
    b = int(input_data_list[1])
    c = int(input_data_list[2])
    d = input_data_list[3]
    return a,b,c,d
if __name__ == '__main__':
    """
    入力:
    test_timer(int):試行回数
    op_timer(int):評価回数
    dim:(int)次元数
    test_Function(str):目標テスト関数

    """
    list_acq = ["LCB", "GP_LCB", "A_LCB", "C_LCB"]
    ave = []
    test_timer,op_timer,dim,test_Function = read_input()
    start_compute_localerror = 5
    d_whether = False  # 探索点の密度を統計する/しない
    Concentration_whether = True  # 探索点の分布情况を統計する/しない
    Approximation_whether = True  # 近似関数误差を統計する/しない
    whether_local_error = False   # 局所近似関数誤差を統計する/しない
    ave_Approximation_K = []
    ave_variance_K = []
    ave_GP_LCB_K = []
    ave_concentration_K = []
    n = 5#初期探索点の数
    for q in list_acq:
        test1 = []

        limit_match_y = 0
        expectdict = {}
        expectdict1 = []  # 优化结果
        expectdict3 = []  # 将expectdict1变回LIST格式
        expectdict4 = []  # 将expectdict3变为int格式
        expectdict2 = {}  # 统计优化结果
        averagevalue = 0  # 平均最优解
        sumvalue = 0
        ave_ybest = []
        K_distance_variance = []  # k近傍法(集中度)
        ave_Approximation_reality_Accuracy = []  # 平均近似関数绝对精度Accuracy
        ave_Approximation_relative_Accuracy = []  # 平均近似関数相对精度Accuracy
        ave_concentration_region_error = []  # 平均集中区域误差
        ave_unconcentration_region_error = []  # 平均离散区域误差
        ave_Approximation1_rate = []
        ave_list = []

        for j in range(test_timer):  # 重复实验(試行)
            test1.append(main2_concentration_version2.bayesian(q,dim,test_Function,whether_local_error))
        for j in range(test_timer):
            for i in range(op_timer):
                test1[j].GPR()
                test1[j].Statistics(d=d_whether, Concentration=Concentration_whether,
                                    deviation=Approximation_whether)
                test1[j].FindNewPoint()

            expectdict[j] = test1[j].ybest
            print("序号" + str(j))
            print("每次选择的点")
            print(test1[j].a)
            print("最適解")
            print(test1[j].ybest)
            print("\n")
            print(test1[j].differ_surrogate_and_reality1)

        print(test1[0].acq)
        sample1 = tongji.computing(test1)
        ave_ybest = sample1.computing_ybest()  # 最適解
        ave_K_distance_variance = sample1.computing_sum_K_distance_variance()  # 探索点分布情况，集中度
        ave_Approximation_reality_Accuracy = sample1.computing_sum_Approximation_Functon1()  # 近似函数绝对誤差
        ave_Approximation_relative_Accuracy = sample1.computing_sum_Approximation_Functon2() #近似関数相対誤差
        ave_concentration_region_error = sample1.computing_sum_concentration_deviation() #集中空間の近似関数誤差
        ave_unconcentration_region_error = sample1.computing_sum_unconcentration_deviation()  # 離散空間近似関数誤差
        # ave_Approximation1_rate = sample1.computing_sum_Approximation_reality_rate1()
        # ave_Approximation2_rate = sample1.computing_sum_Approximation_reality_rate2()
        if q == "A_LCB":
            ave_Approximation_K = sample1.computing_sum_Approximation_K()
        if q == "variance_LCB":
            ave_variance_K = sample1.computing_sum_variance_K()
        if q == "C_LCB":
            ave_concentration_K = sample1.computing_sum_CLCB_K()
        if q == "GP_LCB":
            ave_GP_LCB_K = sample1.computing_sum_GPLCB_K()
        # sum_outside_number = sample1.computing_sum_outside_number()

        ave_list.append(ave_ybest)
        ave_list.append(ave_K_distance_variance)
        ave_list.append(ave_Approximation_reality_Accuracy)
        ave_list.append(ave_Approximation_relative_Accuracy)
        ave_list.append(ave_concentration_region_error)
        ave_list.append(ave_unconcentration_region_error)
        # ave_list.append(ave_Approximation1_rate)
        # ave_list.append(ave_Approximation2_rate)
        ave.append(ave_list)

        expectdict1 = expectdict.values()
        print(expectdict1)

        f = open('F:/pytharm/Bayesian_test/graph1/rastrigin/5dim2/result.txt', 'a', encoding="UTF-8")
        f.write("acq:" + q + "\n")
        f.write(str(test1[0].Initialization_tuple))
        f.write(str(op_timer))

        f.write("目標関数:" + test1[0].determinefunction + "\n")
        f.write("次元数：" + str(test1[0].trans_dimension) + "次元" + "\n")
        f.write("重复实验次数" + str(test_timer) + "\n")
        f.write("每次实验进行的贝叶斯优化次数" + str(op_timer))
        f.write('\n' + "ave_differ" + str(ave_Approximation_reality_Accuracy.tolist()) + "\n")
        f.write('\n' + "ave_ybest" + str(ave_ybest.tolist()) + "\n")
        f.write("\n")
        f.close()

    picture.data_cleaning(ave, op_timer, n)
    picture.draw_parameter(ave_GP_LCB_K, ave_Approximation_K, ave_concentration_K, op_timer)