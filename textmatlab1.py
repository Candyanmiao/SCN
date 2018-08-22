# coding=utf-8

from numpy.matlib import repmat
import math
import matlab.engine
# eng=matlab.engine.start_matlab()
# from . import SCN
# import matplotlib.pyplot as plt
# X = np.genfromtxt("rawx.txt", encoding='utf-8')
# Y =np.fft.fft(X)
# plt.xlabel('time')
# plt.ylabel('幅度')
# plt.plot(X,Y)
# plt.show()

# for i in range(0,19):
#
# figure
# plot(X(1:20,:)','r-o');             % 干磨 红 1
# hold on;
# plot(X(21:56,:)','black-*');        % 空砸 黑 2
# hold on;
# plot(X(57:end,:)','g--');
# import numpy as np
# import math

# import pandas as pd
# x=np.arange(0.05,3,0.05)
# y1=[math.log(a,1.5)for a in x]
# y2=[math.log(a,2)for a in x]
# y3=[math.log(a,3)for a in x]
# plot1=plt.plot(x,y1,'-g',label="log1.5(x)")
# plot2=plt.plot(x,y2,'-r',label="log2(x)")
# plot3=plt.plot(x,y3,'-b',label="log3(x)")
# plt.legend(loc='lower right')
# plt.show()
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
myfont = fm.FontProperties(fname='C:/Windows/Fonts/simsun.ttc')
X = np.genfromtxt("gmpy.txt", encoding='utf-8',skip_header=1)
# Y=np.log10(X.T)
plt.xlabel(u'功率谱',fontproperties=myfont)
plt.ylabel(u'频率/Hz',fontproperties=myfont)
plt.title(u'干磨状态下的振动频域图',fontproperties=myfont)
plt.axis([0,12800,0,2000])
plt.plot(X)
plt.show()
# X = pd.read_table('gm.txt')
# X = np.genfromtxt('smpy.txt',encoding='utf-8',skip_header=1)
# Fs=8000
# t=np.arange(0,len(X))/Fs
# print(t[25600])
# plt.plot(X)
# plt.ylabel(u'功率谱',fontproperties=myfont)
# plt.xlabel(u'频率/Hz',fontproperties=myfont)
# plt.title(u'振动频域波形图',fontproperties=myfont)
# plt.show()