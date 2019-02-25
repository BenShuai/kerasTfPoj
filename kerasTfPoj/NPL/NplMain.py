import jieba
import jieba.posseg as psg

s = u'我想和女朋友一起去北京故宫博物院参观和闲逛。'

cut = jieba.cut(s)

print ("输出:",cut)

print ("精确模式:",','.join(cut))

print ("全模式:",','.join(jieba.cut(s,cut_all = True)))

print ("搜索引擎模式:",','.join(jieba.cut_for_search(s)))

print ("词性:",[(x.word,x.flag) for x in psg.cut(s)])

print ("词性过滤:",[(x.word,x.flag) for x in psg.cut(s) if x.flag.startswith('n')])

