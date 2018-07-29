from matplotlib import pyplot as plt
import numpy as np
import cv2
from pylab import mpl         #改matplotlib显示字体格式
mpl.rcParams['font.sans-serif'] = ['FangSong'] # 指定默认字体（仿宋）
train_pts = 100
# 创建测试的数据点，2类
# 以(-1.5, -1.5)为中心
rand = np.random.rand(train_pts, 2)
print("rand",rand)
rand1 = np.ones((train_pts ,2)) * (-2) + np.random.rand(train_pts, 2)
print('rand1：')
print(rand1)

# 以(1.5, 1.5)为中心,长宽为1的正方形
rand2 = np.ones((train_pts ,2)) + np.random.rand(train_pts, 2)
print('rand2:')
print(rand2)

# 合并随机点，得到训练数据
train_data = np.vstack((rand1, rand2))
train_data = np.array(train_data, dtype='float32')
train_label = np.vstack( (np.zeros((train_pts ,1), dtype='int32'), np.ones((train_pts ,1), dtype='int32')))
print("train_data" , train_data)
print("train_label" , train_label)
# 显示训练数据
fig1 = plt.figure(1)
plt.plot(rand[: ,0], rand[: ,1], '*',label = "rand")
plt.plot(rand1[: ,0], rand1[: ,1], 'o',label = "rand1")
plt.plot(rand2[: ,0], rand2[: ,1], 'o',label = "rand2" )

fig2 = plt.figure(2)
plt.plot(train_data[: ,0], train_data[: ,1], 'o',label = "train_data")
# plt.plot(train_label[: ,0], train_label[: ,1], 'o')
svm = cv2.ml.SVM_create()
print(type(svm))
svm.setType(cv2.ml.SVM_C_SVC)
svm.setKernel(cv2.ml.SVM_LINEAR)
svm.setC(1.0)
ret = svm.train(train_data , cv2.ml.ROW_SAMPLE , train_label)
#实验数据
pt = np.array(np.random.rand( 20 , 2)* 4 - 2 ,dtype = "float32" )
#fig3 根据训练结果，将实验数据分类
fig3 = plt.figure(3)
plt.plot(pt[: ,0], pt[: ,1], 'o', label = "all_ex_data")
print("pt" , pt)
#通过训练样本得到的predict, 将实验数据传递给predict，predict根据训练样本计算实验数据分类，按照顺序给 0，   1. 分类序号
ret , res = svm.predict( pt )
print("pt", pt)
print("res:" , res.ravel())
res = np.hstack((res , res))
#相当于挑选出res对应为 1 的pt
type_data = pt[ res < 0.5]
print("type_data:" , type_data)
print("type_data", type_data.ravel())
type_data = np.reshape(type_data, (int(type_data.shape[0] / 2), 2))
print("type_data:" , type_data)
fig3,plt.plot(type_data[:,0], type_data[:,1], '+',label = "type_1_data")

type_data = pt[ res >= 0.5]
type_data = np.reshape(type_data, (int(type_data.shape[0] / 2), 2))
fig3,plt.plot(type_data[:,0], type_data[:,1],"*",color = "black",label = "type_2_data")
#获取支持向量

vec = svm.getSupportVectors()
print("支持向量",vec)
#添加图例
fig1.legend( )
fig2.legend( )
fig3.legend( )
plt.show( )