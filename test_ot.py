import openturns as ot
import numpy as np
import pdb

def ot_gp(x_train_data,y_train_data,x_test_data,dim):

    """

    :param x_train_data: (ndarray) 探索点のlist
    :param y_train_data: (ndarray) 探索点の目標テスト関数値のlist
    :param x_test_data:  (ndarray) サンプル点list
    :param dim: (int) 次元数
    :return: y_test_MM_numpy: (ndarray) サンプル点の均値
             conditionalSigma_numpy: (ndarray) サンプル点の標準偏差
    """
    x_train = ot.Sample(x_train_data.tolist())
    y_train = ot.Sample(y_train_data.tolist())
    x_test  = ot.Sample(x_test_data.tolist())
    amplitude = [1.0]
    nu = 1.5
    dimension = dim
    basis = ot.ConstantBasisFactory(dimension).build()
    basis = ot.QuadraticBasisFactory(dimension).build()

    covarianceModel = ot.MaternModel([6.0] * dimension,amplitude,nu)
    amplitude_bounds = [1e-3, 1e3]

    algo = ot.KrigingAlgorithm(x_train, y_train, covarianceModel, basis)
    try:
        algo.run()
    except:
        try:
            basis = ot.LinearBasisFactory(dimension).build()
            covarianceModel = ot.MaternModel([1.0] * dimension, 2.5)
            algo = ot.KrigingAlgorithm(x_train, y_train, covarianceModel, basis)
            algo.run()
        except:
            print("Not optimizing covariance model parameters!!!!")
            y_test_MM = np.zeros(100)
            conditionalSigma = np.zeros(100)
            return y_test_MM, conditionalSigma
    krigingresult = algo.getResult()

    krigeageMM = krigingresult.getMetaModel()  # 返回一个MetaModel模型用于进行新点的预测、模拟等操作
    #covarianceModel.setParameter(ot.MaternModel.Amplitude(), [2])

    # 获取预测均值和方差
    y_test_MM = krigeageMM(x_test)  #均值
    sqrt = ot.SymbolicFunction(["x"], ["sqrt(x)"])
    epsilon = ot.Sample(100,[1.0e-8])
    varianceresult = krigingresult.getConditionalMarginalVariance(x_test) +epsilon
    varianceresult = np.nan_to_num(varianceresult, nan=0)  # NaNを強引に0に置き換えただけ，気に入らない
    # print("第几次循环")
    # print(y_test_MM)
    # print("\n")
    # print(varianceresult)
    conditionalSigma = sqrt(varianceresult)  #标准差

    y_test_MM_numpy = np.array(y_test_MM)
    conditionalSigma_numpy = np.array(conditionalSigma)
    return y_test_MM_numpy,conditionalSigma_numpy

# if __name__ == '__main__':
#     x = np.array([[-4,-4,-4], [4,4,4], [-5,-5,-4],[3,-3,4],[-4,5,-3], [-3,2,2],[0.999293,0.999293,1],[0.499293,0.699293,0.5],[0.199293,0.299293,0.3],[0.299293,0.599293,0.9]])
#     y = np.array([[32], [32], [50], [18], [41], [32],[12],[6],[2],[4]])
#     #x_test = np.array([[-1,1], [9,2], [-2,1]])
#     x_test = np.array([[-3.83723842,  3.37887186,2],
#        [-4.10071538, -0.37878224,-0.4],
#        [-1.7056484 , -4.23274984,2],
#
#        [-1.40734539, -1.53714213,1]])
#     mean,sigma = ot_gp(x,y,x_test,3)
#     print(mean)
#     print("\n")
#     print(sigma)