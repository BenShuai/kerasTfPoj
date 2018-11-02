import wave
import numpy as np
import os
from keras.models import load_model



def get_wav_mfcc(wav_path):
    f = wave.open(wav_path,'rb')
    params = f.getparams()
    # print("params:",params)
    nchannels, sampwidth, framerate, nframes = params[:4]
    strData = f.readframes(nframes)#读取音频，字符串格式
    waveData = np.fromstring(strData,dtype=np.int16)#将字符串转化为int
    waveData = waveData*1.0/(max(abs(waveData)))#wave幅值归一化
    waveData = np.reshape(waveData,[nframes,nchannels]).T
    f.close()

    ### 对音频数据进行长度大小的切割，保证每一个的长度都是一样的【因为训练文件全部是1秒钟长度，16000帧的，所以这里需要把每个语音文件的长度处理成一样的】
    data = list(np.array(waveData[0]))
    # print(len(data))
    while len(data)>16000:
        del data[len(waveData[0])-1]
        del data[0]
    # print(len(data))
    while len(data)<16000:
        data.append(0)
    # print(len(data))

    data=np.array(data)
    # 平方之后，开平方，取正数，值的范围在  0-1  之间
    data = data ** 2
    data = data ** 0.5
    return data


if __name__ == '__main__':
    # 构建模型
    model = load_model('asr_model_weights.h5') # 加载训练模型
    wavs=[]
    wavs.append(get_wav_mfcc("D:\\wav\\trunk\\1b.wav"))
    X=np.array(wavs)
    print(X.shape)
    result=model.predict(X[0:1])[0] # 识别出第一张图的结果，多张图的时候，把后面的[0] 去掉，返回的就是多张图结果
    print("识别结果",result)
    #  因为在训练的时候，标签集的名字 为：  0：seven   1：stop    0 和 1 是下标
    name = ["seven","stop"] # 创建一个跟训练时一样的标签集
    ind=0 # 结果中最大的一个数
    for i in range(len(result)):
        if result[i] > result[ind]:
            ind=1
    print("识别的语音结果是：",name[ind])




