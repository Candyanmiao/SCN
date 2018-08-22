import numpy as np
from numpy.matlib import repmat
import math
import matlab.engine

# 导入matlab引擎，实现python、matlab混合编程的第一步
eng = matlab.engine.start_matlab()


# 通过api开启matlab
class SCN():
    # 版本
    def __init__(self, L_max, T_max, tol, Lambdas, r, nB):
        # 初始化SCN类
        # 定义一个SCN类
        self.Name = 'Stochastic Configuration Networks'
        # 类属性名
        self.verson = '1.0 beta'
        self.L = 1
        # 隐藏节点个数
        self.Lambdas = [0.5, 1, 3, 5, 7, 9, 15, 25, 50, 100, 150, 200]
        # 参数，用来确定输入权重w和偏差b范围的一个参数，是个列表
        self.r = [0.9, 0.99, 0.999, 0.9999, 0.99999, 0.99999]
        # 一个列表，每次用里面的一个数，逼近程度的一个参数，趋于1
        self.T_max = 300
        # 最大的参数配置次数
        self.nB = 1
        # 在一次训练中添加的隐藏节点个数
        self.tol = 0.1
        # 容忍度(误差应小于这个值）
        self.L_max=500
        self.verbose = 50
        self.W=[]
        self.b=[]

    def InequalityEq(self, eq, gk, r_L):
        # 判断下个函数找到的w和b是否正确用的不等式*****
        ksi = (np.dot(eq.T, gk) ** 2 / (np.dot(gk.T, gk) - (1 - r_L) * np.dot(eq.T, eq)))
        return ksi

    def SC_Search(self, X, E0):
        # 找W和b的方法（随机配置w,b的算法）
        Flag = 0
        WB = []
        bB = []
        WT = []
        bT = []
        # Ksi_t = 0
        d = X.shape[1]
        m= E0.shape[1]
        C = []
        nC=0
        list1=[]
        for i in range(0, len(self.Lambdas)-1):
            Lambda = self.Lambdas[i]
            WT = Lambda * (2.0 * np.random.rand(d, self.T_max) - 1)
            bT = Lambda * (2.0 * np.random.rand(1, self.T_max) - 1)

            HT = 1.0 / (1 + np.exp(-X.dot(WT) - bT))
            for i_r in range(0, len(self.r)-1):
                r_L = self.r[i_r]
                for t in range(0, self.T_max-1):
                    H_t = HT[:, t]
                    ksi_m = np.zeros((1,m))
                    for i_m in range(0, m-1):
                        eq = E0[:, i_m]
                        gk = H_t
                        ksi_m[i_m-1] = self.InequalityEq(eq, gk, r_L)
                    Ksi_t = sum(ksi_m)
                    # Ksi_t=np.array(Ksi_t)
                    if ksi_m.all() > 0:
                        C= C.append(Ksi_t)
                        WB = WB.append(WT[:, t])
                        bB = bB.append( bT[:, t])
                nC = len(C)
                if nC >= self.nB:
                    break
                else:
                    continue
            if nC >= self.nB:
                break
            else:
                continue
        if nC >= self.nB:
            [a, I] = sorted(C, reverse=True)
            I_nb = I[1:self.nB]
            WB = WB[:, I_nb]
            bB = bB[:, I_nb]
        if nC == 0 or nC < self.nB:
            print('End Searching...')
            Flag = 1
            list1 = [WB,bB,Flag]
        return list1

    def AddNodes(self, w_L, b_L):
        self.W = self.W.append( w_L)
        self.b = self.b.append( b_L)
        self.L = len(self.b)

    def UpgradeSCN(self, X, T):
        H = self.GetH(X)
        self.ComputeBeta(H, T)
        # O = np.dot(H, self.Beta)
        O=H.dot(self.Beta)
        E = T - O
        EN = E.size
        Error = np.sqrt(np.sum(np.sum(E ** 2) / EN))
        self.COST = Error

    def ComputeBeta(self, H, T):
        Beta = eng.pinv(H) * T
        self.Beta = Beta

    def Classification(self, X, T):
        Error = []
        Rate = []
        Flag = 0
        E = T
        EN =int(E.size)
        Error = np.sqrt(np.sum(np.sum(E ** 2) / EN))
        Rate = 0
        print(self.Name)
        while (self.L < self.L_max) and (Error > self.tol):

            if self.L % self.verbose == 0:
                print(self.L, Error, Rate)
            list2=self.SC_Search(X, E)
            if Flag == 1:
                break
            self.AddNodes(list2[0], list2[1])
            [obj, o, E, Error] = self.UpgradeSCN(X, T)
            O = self.GetLabel(X)
            Rate = 1 - eng.confusion(T.T, O.T)
            Error = np.hstack((Error, repmat(Error, 1, self.nB)))
            Rate = np.hstack((Rate, repmat(Rate,1, self.nB)))
        print(self.L, Error, Rate)
        print('*' * 30)
        return self, Error, Rate

    def GetH(self, X):
        H = self.ActivationFun(X)

    # sigmoid激活函数
    def ActivationFun(self, X):
        H = 1.0 / (1 + math.exp(-X * self.W - self.b))


    # 输出节点
    def GetOutput(self, X):
        H = self.GetH(X)
        O = H * self.Beta
        return O

    def GetLabel(self, X):
        O = self.GetOutput(X)
        O = eng.OneHotMatrix(O)

    def GetAccuracy(self, X, T):
        O = self.GetLabel(X)
        Rate = 1 - eng.confusion(T.T, O.T)

    def GetResult(self, X, T):
        H = self.GetH(X)
        O = H * self.Beta
        E = T - O
        EN = E.size
        Error = np.sqrt(np.sum(np.sum(E ** 2) / EN))


a = SCN(200, 300, [0.9, 0.99, 0.999, 0.9999, 0.99999], [0.01, 0.1, 1, 10, 100], [0.9, 0.99, 0.999, 0.9999, 0.99999], 1)
# read_csv（,encoding=utf-8）
X = np.genfromtxt("X.txt", encoding='utf-8',skip_header=1)
X = np.array(X)
print(type(X))
T = np.genfromtxt("T.txt", encoding='utf-8',skip_header=1)
T[0,0]=1
# T=eng.OneHotMatix(T)
[a, err, rate] = a.Classification(X, T)
X2 = np.genfromtxt("X2.txt", encoding='utf-8',skip_header=1)
T2 = np.genfromtxt("T2.txt", encoding='utf-8',skip_header=1)
accuracy = a.GetAccuracy(X2,T2)
print(accuracy)
