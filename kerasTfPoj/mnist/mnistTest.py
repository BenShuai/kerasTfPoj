import tensorflow as tf

from PIL import Image
import numpy as np
from keras.models import load_model

# 构建模型
model = load_model('mnist_model_weights.h5') # 加载训练模型
# model.summary()

def read_image(img_name):
    im = Image.open(img_name).resize((28,28),Image.ANTIALIAS).convert('L') # 将要识别的图缩放到训练图一样的大小，并且灰度化
    data = np.array(im)
    return data

images=[]
images.append(read_image("test.png"))
# print(images)

X = np.array(images)
print(X.shape)
X=X.reshape(1, 784).astype('float32')
print(X.shape)
X /=255
# print(X[0:1])
result=model.predict(X[0:1])[0] # 识别出第一张图的结果，多张图的时候，把后面的[0] 去掉，返回的就是多张图结果
num=0 # 用来分析预测的结果
for i in range(len(result)): # result的长度是10
    # print(result[i]*255)
    if result[i]*255>result[num]*255: # 值越大，就越可能是结果
        num=i

print("预测结果",num)