python 调用opencv内置svm分类器进行训练、sample预测
#对于已有训练数据的svm
 #创建svm分类器，同时进行params设置（应该与得到训练数据params设置相同）
 # .load() 加载训练结果数据，.load()将关联相应的svm分类器，同时生成相对应的predict()预测函数
 # 将sample传递给predict()，predict()返回该sample预测得到的label，（label与训练有关）
 # 根据返回的label判断sample类型

