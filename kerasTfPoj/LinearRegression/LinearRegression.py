import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
import numpy as np
import matplotlib.pyplot as plt

## 线性回归
x_data = np.random.rand(100) # 随机生成一堆数据

noise = np.random.normal(0,0.01,x_data.shape) # 正态分布 生成 噪点
y_data = 0.1*x_data + 0.2 + noise

plt.scatter(x_data,y_data)
plt.show()

model = Sequential()
model.add(Dense(1, input_shape=(1,)))

# [编译模型] 配置模型，损失函数采用，优化采用Adadelta，将识别准确率作为模型评估
model.compile(loss=keras.losses.MSE, optimizer=SGD())

for step in range(5000):
    cost = model.train_on_batch(x_data,y_data)
    if step % 500 == 0 :
        print("step:",step,"    cost:",cost)

W,B = model.layers[0].get_weights()
print("W:",W,"   B:",B)

y_pred = model.predict(x_data)
plt.scatter(x_data,y_data)
plt.plot(x_data,y_pred,'r-',lw=3)
plt.show()










