import matlab.engine
import numpy as np
import matplotlib.pyplot as plt
# from sklearn.metrics import confusion_matrix
from sklearn.metrics import confusion_matrix as confusion
# 导入matlab引擎，实现python、matlab混合编程的第一步
eng = matlab.engine.start_matlab()
[train_acc,test_acc,Rate,Error,y_pre,y_true]=eng.eval('BallMill_Classification_SCN()',nargout=6)
y_pre = np.array(y_pre)
y_pre=y_pre.reshape(-1,1)
# print(y_pre)
y_true = np.array(y_true)
y_true=y_true.reshape(-1,1)
# print(y_true)
Y=confusion(y_true, y_pre)
print(Y)
# eng.plotconfusion(Y)
plt.plot(Y)
plt.show()