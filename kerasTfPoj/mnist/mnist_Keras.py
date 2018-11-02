import tensorflow as tf

import keras
from keras.models import Sequential
from keras.layers import Dense,Dropout, Flatten,Conv2D, MaxPooling2D
from keras.optimizers import SGD

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

print('tf',tf.__version__)
print('keras',keras.__version__)


# batch大小，每处理128个样本进行一次梯度更新
batch_size = 64
# 训练素材类别数
num_classes = 10
# 迭代次数
epochs = 5



f = np.load("mnist.npz")
x_train, y_train = f['x_train'], f['y_train']
x_test, y_test = f['x_test'], f['y_test']
f.close()
print(x_train.shape,"  ",y_train.shape)
print(x_test.shape,"  ",y_test.shape)

# im=plt.imshow(x_train[0],cmap="gray")
# plt.show()

## 维度合并784 = 28*28
x_train = x_train.reshape(60000, 784).astype('float32')
x_test = x_test.reshape(10000, 784).astype('float32')

## 归一化，像素点的值 转成 0-1 之间的数字
x_train /= 255
x_test /= 255

# print(x_train[0])

# 标签转换为独热码
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
# print(y_train[0]) ## 类似 [0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]

print(x_train.shape,"  ",y_train.shape)
print(x_test.shape,"  ",y_test.shape)

# 构建模型
model = Sequential()
model.add(Dense(512, activation='relu',input_shape=(784,)))
model.add(Dense(256, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))
model.summary()

# [编译模型] 配置模型，损失函数采用交叉熵，优化采用Adadelta，将识别准确率作为模型评估
model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])
#  validation_data为验证集
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test))

# 开始评估模型效果 # verbose=0为不输出日志信息
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1]) # 准确度


model.save('mnist_model_weights.h5') # 保存训练模型





