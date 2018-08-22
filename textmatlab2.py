import matlab.engine
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix as confusion
# 导入matlab引擎，实现python、matlab混合编程的第一步
eng = matlab.engine.start_matlab()
# a=[1,2]
# b=str(a)
# c=[3,4]
# d=str(c)
# print('fun1('+b+','+d+')')
#
# c=eng.eval('fun1('+b+','+d+')',nargout=1)
# print(c)
# c=eng.eval(['fun1(a,b)'],nargout=1)
# import matplotlib.font_manager as fm
# myfont = fm.FontProperties(fname='C:/Windows/Fonts/simsun.ttc')
[train_acc,test_acc,Rate,Error,y_pre,y_true]=eng.eval('BallMill_Classification_SCN()',nargout=6)
# print(train_acc,test_acc)
y_pre = np.array(y_pre)
y_pre=y_pre.reshape(-1,1)
print(y_pre)
y_true = np.array(y_true)
y_true=y_true.reshape(-1,1)
print(y_true)
# Error = np.array(Error)
# print(Rate)
# plt.plot(Rate.T,label='training accuracy')
# plt.plot(Error.T,label='Training RMSE')
# plt.xlabel(u'隐含层节点个数',fontproperties=myfont)
# plt.ylabel('RMSE')
# plt.legend() # 显示图例
# T = np.genfromtxt("T.txt", encoding='utf-8',skip_header=1)
# T=np.array(T)
# Y=confusion(y_true,y_pre,labels=[0,1,2])
# print(Y)
# def plot_confusion_matrix(cm, title='Confusion Matrix',labels=y_true):
#     plt.imshow(cm, interpolation='nearest')
#     plt.title(title)
#     plt.colorbar()
#     xlocations = np.array(range(len(labels)))
#     plt.xticks(xlocations, labels, rotation=90)
#     plt.yticks(xlocations, labels)
#     plt.ylabel('True label')
#     plt.xlabel('Predicted label')
# cm = confusion_matrix(y_true, y_pred)
# plot_confusion_matrix(y_pre,y_true)
# plt.plot(Y)
# plt.show()