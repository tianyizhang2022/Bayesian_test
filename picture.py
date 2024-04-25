import matplotlib.pyplot as plt
import numpy as np

def data_cleaning(ave, op_timer,n): #n是初始点数量
    """

    :param ave: 図のデータ
    :param op_timer:評価回数
    :param n:初期探索点数
    """

    ave_ybest = []
    ave_K_distance_variance = []  # 平均方差
    ave_Approximation_reality_Accuracy = []  # 平均近似関数绝对精度
    ave_Approximation_relative_Accuracy = []  # 平均近似関数相对精度
    ave_concentration_error = []
    ave_unconcentration_error = []

    LCB = ave[0]
    GP_LCB = ave[1]
    A_LCB = ave[2]
    C_LCB = ave[3]


    ave_ybest.append(LCB[0][1:op_timer + 1]) #最优解
    ave_ybest.append(GP_LCB[0][1:op_timer + 1])
    ave_ybest.append(A_LCB[0][1:op_timer + 1])
    ave_ybest.append(C_LCB[0][1:op_timer + 1])
    ave_K_distance_variance.append(LCB[1])  #探索点的分布情况
    ave_K_distance_variance.append(GP_LCB[1])
    ave_K_distance_variance.append(A_LCB[1])
    ave_K_distance_variance.append(C_LCB[1])
    ave_Approximation_reality_Accuracy.append(LCB[2]) #近似函数绝对误差（精度）
    ave_Approximation_reality_Accuracy.append(GP_LCB[2])
    ave_Approximation_reality_Accuracy.append(A_LCB[2])
    ave_Approximation_reality_Accuracy.append(C_LCB[2])
    ave_Approximation_relative_Accuracy.append(LCB[3]) #近似函数相对误差（精度）
    ave_Approximation_relative_Accuracy.append(GP_LCB[3])
    ave_Approximation_relative_Accuracy.append(A_LCB[3])
    ave_Approximation_relative_Accuracy.append(C_LCB[3])
    ave_concentration_error.append(LCB[4])
    ave_concentration_error.append(GP_LCB[4])
    ave_concentration_error.append(A_LCB[4])
    ave_concentration_error.append(C_LCB[4])
    ave_unconcentration_error.append(LCB[5])
    ave_unconcentration_error.append(GP_LCB[5])
    ave_unconcentration_error.append(A_LCB[5])
    ave_unconcentration_error.append(C_LCB[5])

    #以上绘图数据准备完毕

    draw_picture(ave_ybest, "evaluation value", op_timer,n)
    draw_picture(ave_K_distance_variance, "ave_K_distance_variance", op_timer, n)
    draw_picture(ave_Approximation_reality_Accuracy,"ave_reality_error",op_timer,n)
    draw_picture(ave_Approximation_relative_Accuracy, "ave_relative_error", op_timer,n)

def draw_parameter(GP_LCB_K,A_K,C_K,op_timer):
    x = []
    for i in range(op_timer):
        x.append(i + 1)
    plt.figure(figsize=(8, 8))
    plt.plot(x[0:146],GP_LCB_K[0:146],label = "GP_LCB",linestyle='-', linewidth=2)
    plt.plot(x[0:146],A_K[0:146],label = "A_LCB",linestyle='--', linewidth=2)  #R-LCB [0:199],C-LCB[0:200]
    plt.plot(x[0:146],C_K[0:146], label="C_LCB", linestyle=':', linewidth=2)
    plt.tick_params(axis='x', labelsize=20)  # x 轴刻度字号为 12
    plt.tick_params(axis='y', labelsize=20)  # y 轴刻度字号为 12
    plt.grid(True)
    plt.xlabel("number of evaluations",fontsize=20)
    plt.ylabel("parameter(k)",fontsize=20)
    plt.legend(fontsize=24)
    plt.savefig("F:/pytharm/Bayesian_test/graph1/rastrigin/5dim2/parameter.png")
def draw_loacl_error(ave,label,op_timer,start_compute_localerror):
    x = []
    for i in range(start_compute_localerror + 1, op_timer + 1):
        x.append(i)
    LCB = ave[0]
    GP_LCB = ave[1]
    A_LCB = ave[2]
    C_LCB = ave[3]
    plt.figure(figsize=(8, 8))
    plt.plot(x[0:op_timer - start_compute_localerror], LCB[0:op_timer - start_compute_localerror], label="LCB", linestyle='-', linewidth=3)
    plt.tick_params(axis='x', labelsize=20)  # x 轴刻度字号为 12
    plt.tick_params(axis='y', labelsize=20)  # y 轴刻度字号为 12
    plt.grid(True)
    plt.xlabel("number of evaluations", fontsize=18)
    plt.ylabel("{}".format(label), fontsize=18)
    plt.legend(fontsize=24)
    plt.savefig("F:/pytharm/Bayesian_test/graph1/rastrigin/5dim2/{}.png".format(label))

def draw_picture(ave,label,op_timer,n):
    x=[]
    for i in range(n+1,op_timer+1):
        x.append(i)
    LCB = ave[0]
    GP_LCB = ave[1]
    A_LCB = ave[2]
    C_LCB = ave[3]
    plt.figure(figsize=(8, 8))
    plt.plot(x[0:op_timer-n],LCB[0:op_timer-n],label = "LCB",linestyle='-', linewidth=3)
    plt.plot(x[0:op_timer-n], GP_LCB[0:op_timer-n], label="GP_LCB",linestyle='--', linewidth=3)
    plt.plot(x[0:op_timer-n], A_LCB[0:op_timer-n], label="A-LCB",linestyle=':', linewidth=3)
    plt.plot(x[0:op_timer-n], C_LCB[0:op_timer-n], label="C-LCB",linestyle='-.', linewidth=3)
    plt.tick_params(axis='x', labelsize=20)  # x 轴刻度字号为 12
    plt.tick_params(axis='y', labelsize=20)  # y 轴刻度字号为 12
    plt.grid(True)
    plt.xlabel("number of evaluations",fontsize=18)
    plt.ylabel("{}".format(label),fontsize=18)
    plt.legend(fontsize=24)
    plt.savefig("F:/pytharm/Bayesian_test/graph1/rastrigin/5dim2/{}.png".format(label))



def draw_Xpointpicture(X_point,q,j):
    x1=[]
    x2=[]
    for i in X_point:
        x1.append(i[0])
        x2.append(i[1])
    fig, ax = plt.subplots()
    ax.set_xlim(-6, 6)
    ax.set_ylim(-6, 6)
    ax.scatter(x1,x2)
    x_ticks = np.arange(-6, 6, 0.5)
    y_ticks = np.arange(-6, 6, 0.5)
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    ax.grid(True)
    plt.savefig("F:/pytharm/Bayesian_test/graph1/rastrigin/1/{}{}.png".format(q,j))
    # for i in range(len(x1)):
    #     plt.scatter(x1[i], x2[i])
    #     plt.pause(1)
    #     plt.savefig("D:/pycharm/Bayesian_test/graph1/rastrigin/1/{}.png".format(i))

    plt.legend()
    plt.show()


def data_cleaning_ucblcb(ave):
    ave_differ = []
    ave_y_up = []
    ave_x_differ = []
    ave_ybest = []
    lcb = ave[0]
    ucb = ave[1]
    ave_differ.append(lcb[0])
    ave_differ.append(ucb[0])
    ave_y_up.append(lcb[1])
    ave_y_up.append(ucb[1])

    ave_x_differ.append(lcb[2])
    ave_x_differ.append(ucb[2])

    ave_ybest.append(lcb[3][1:31])
    ave_ybest.append(ucb[3][1:31])
    draw_picture_ucblcb(ave_differ, "ave_differ")
    draw_picture_ucblcb(ave_y_up, "ave_y_up")
    draw_picture_ucblcb(ave_x_differ, "ave_x_differ")
    draw_picture_ucblcb(ave_ybest, "ave_ybest")

def draw_picture_ucblcb(ave,label):
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
    lcb = ave[0]
    ucb = ave[1]
    plt.figure(figsize=(8, 8))
    plt.plot(x, ucb, label="UCB")
    plt.plot(x, lcb, label="LCB")
    plt.xlabel("remark number")
    plt.ylabel("{}".format(label))
    plt.legend()
    plt.savefig("F:/pytharm/Bayesian_test/graph/rosenbrock/2dim/LCB_UCB/k=1_{}.png".format(label))

