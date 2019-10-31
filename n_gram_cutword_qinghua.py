#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 09:12:21 2019

@author: daniel
"""

## N-gram cutword Bilstm test


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from keras.models import Model
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
from keras.optimizers import RMSprop,Adam
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.callbacks import EarlyStopping

## 添加 Sequential, Bidirectional
from keras.models import Sequential
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.layers import Dense, Activation, Flatten

## 添加
from keras.layers.convolutional import Conv1D, MaxPooling1D
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# 指定不需要GUI的backend（Agg, Cairo, PS, PDF or SVG，防止ssh 端报错
plt.switch_backend('agg')
## 设置字体
# =============================================================================
# from matplotlib.font_manager import FontProperties
# fonts = FontProperties(fname = "‪C:\Windows\Fonts\STXINGKA.TTF",size=14)
# =============================================================================

### 读取测数据集
#train_df = pd.read_csv("cnews-LSTM/corpus_train_split.csv")
#val_df = pd.read_csv("cnews-LSTM/corpus_val_split.csv")
#test_df = pd.read_csv("cnews-LSTM/corpus_test_split.csv")
## 读取测数据集
train_df = pd.read_csv("trainR.csv")
val_df = pd.read_csv("valR.csv")
test_df = pd.read_csv("testR.csv")
print(train_df['label'].value_counts())
cata = list(train_df['label'].value_counts().index)
catalen = len(cata)
train_df.head()

plt.figure()
sns.countplot(train_df.label)
plt.xlabel('Label')
plt.xticks()
plt.show()
plt.figure()
sns.countplot(train_df.label)
plt.xlabel('Label')
plt.xticks()
plt.savefig('TTNRtrain.pdf')
##plt.show()

print(train_df.cutwordnum.describe())
plt.figure()
plt.hist(train_df.cutwordnum,bins=100)
plt.xlabel("length")
plt.ylabel("number")
plt.title("Training set")
plt.savefig("TTNRcutwords.pdf")
#plt.show()

## 对数据集的标签数据进行编码
train_y = train_df.label
val_y = val_df.label
test_y = test_df.label
le = LabelEncoder()
train_y = le.fit_transform(train_y).reshape(-1,1)
val_y = le.transform(val_y).reshape(-1,1)
test_y = le.transform(test_y).reshape(-1,1)

## 对数据集的标签数据进行one-hot编码
ohe = OneHotEncoder()
train_y = ohe.fit_transform(train_y).toarray()
val_y = ohe.transform(val_y).toarray()
test_y = ohe.transform(test_y).toarray()

## 使用Tokenizer对词组进行编码
## 当我们创建了一个Tokenizer对象后，使用该对象的fit_on_texts()函数，以空格去识别每个词,
## 可以将输入的文本中的每个词编号，编号是根据词频的，词频越大，编号越小。
max_words = 5000
max_len = 60
tok = Tokenizer(num_words=max_words)  ## 使用的最大词语数为5000
tok.fit_on_texts(train_df.cutword)

## 使用word_index属性可以看到每次词对应的编码
## 使用word_counts属性可以看到每个词对应的频数
for ii,iterm in enumerate(tok.word_index.items()):
    if ii < 10:
        print(iterm)
    else:
        break
print("===================")  
for ii,iterm in enumerate(tok.word_counts.items()):
    if ii < 10:
        print(iterm)
    else:
        break

## 对每个词编码之后，每句新闻中的每个词就可以用对应的编码表示，即每条新闻可以转变成一个向量了：
train_seq = tok.texts_to_sequences(train_df.cutword)
val_seq = tok.texts_to_sequences(val_df.cutword)
test_seq = tok.texts_to_sequences(test_df.cutword)
## 将每个序列调整为相同的长度
train_seq_mat = sequence.pad_sequences(train_seq,maxlen=max_len)
val_seq_mat = sequence.pad_sequences(val_seq,maxlen=max_len)
test_seq_mat = sequence.pad_sequences(test_seq,maxlen=max_len)

print(train_seq_mat.shape)
print(val_seq_mat.shape)
print(test_seq_mat.shape)


##d定义bilstm层
model = Sequential()
model.add(Embedding(max_words+1, 128, input_length = max_len))
model.add(Dropout(0.5))

model.add(Conv1D(filters=128, kernel_size = 5, padding = 'same', activation = 'relu'))
model.add(MaxPooling1D(pool_size = 3))

model.add(Conv1D(filters=128, kernel_size = 5, padding = 'same', activation = 'relu'))
model.add(MaxPooling1D(pool_size = 3))

model.add(Bidirectional(LSTM(128, return_sequences = True), merge_mode = 'concat'))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(128, activation = 'relu'))
model.add(Dropout(0.5))
#model.add(Dense(128, activation = 'sigmoid'))
#model.add(Dropout(0.5))
model.add(Dense(catalen, activation = 'softmax'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
model_fit = model.fit(train_seq_mat,train_y,batch_size=128,epochs=30,
                      validation_data=(val_seq_mat,val_y), verbose = 1)

score = model.evaluate(test_seq_mat,test_y, verbose = 1)
print("Test socre:", score[0])
print("Test accuracy:", score[1])



 ### 定义LSTM模型
 #inputs = Input(name='inputs',shape=[max_len])
 #    ## Embedding(词汇表大小,batch大小,每个新闻的词长)
 #layer = Embedding(max_words+1,128,input_length=max_len)(inputs)
 #layer = LSTM(128)(layer)
 #layer = Dense(128,activation="relu",name="FC1")(layer)
 #layer = Dropout(0.5)(layer)
 #layer = Dense(10,activation="softmax",name="FC2")(layer)
 #model = Model(inputs=inputs,outputs=layer)
 #model.summary()
 #model.compile(loss="categorical_crossentropy",optimizer=Adam(),metrics=["accuracy"])
 #    
 #    
 #model_fit = model.fit(train_seq_mat,train_y,batch_size=128,epochs=10,
 #                          validation_data=(val_seq_mat,val_y)
 #                         )
 #
 ####    callbacks=[EarlyStopping(monitor='val_loss',min_delta=0.0001)] ## 当val-loss不再提升时停止训练
 
 ## 对测试集进行预测
test_pre = model.predict(test_seq_mat)
 
 ## 评价预测效果，计算混淆矩阵
confm = metrics.confusion_matrix(np.argmax(test_pre,axis=1),np.argmax(test_y,axis=1))
 ## 混淆矩阵可视化
#Labname = ["财经","房产","家居","科技","时尚","体育","游戏","彩票","股票","教育","社会","时政","星座","娱乐"]
Labname = cata
plt.figure(figsize=(8,8))
sns.heatmap(confm.T, square=True, annot=True,
            fmt='d', cbar=False,linewidths=.8,
            cmap="YlGnBu")
plt.xlabel('True label',size = catalen)
plt.ylabel('Predicted label',size = catalen)
plt.xticks(np.arange(catalen)+1,Labname)
plt.yticks(np.arange(catalen)+1,Labname)
plt.savefig("TTNRmix.pdf")
plt.show()
 
 
print(metrics.classification_report(np.argmax(test_pre,axis=1),np.argmax(test_y,axis=1)))