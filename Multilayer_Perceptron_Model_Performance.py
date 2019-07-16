
# coding: utf-8

# In[3]:


import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto()
config.gpu_options.visible_device_list='2'
config.gpu_options.per_process_gpu_memory_fraction = 0.25
set_session(tf.Session(config=config))


# In[4]:


import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.datasets import reuters

#1. 데이터셋 준비
max_words = 1000
(x_train, y_train), (x_test, y_test) = reuters.load_data(num_words=max_words,
                                                         test_split=0.2)
word_index = reuters.get_word_index(path="reuters_word_index.json")

print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

num_classes = np.max(y_train) + 1
print(num_classes, 'classes')


# In[4]:


# train sequences(train 문장수): 8982 
# test sequences(test 문장수): 2246
# classes(주제수): 46


# In[5]:


## 데이터 보기
index_to_word = {}
for key, value in word_index.items():
    index_to_word[value] = key
print(' '.join([index_to_word[x] for x in x_train[0]]))


# In[6]:


print(y_train[0]) # 주제에서 3으로 분류됨


# In[7]:


# data preprocessing
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical

 # for convering sentences to matrix
tokenizer = Tokenizer(num_words=max_words) # 가장 많이 나오는 내용 중 1,000가지 선별
x_train = tokenizer.sequences_to_matrix(x_train, mode='binary')
x_test = tokenizer.sequences_to_matrix(x_test, mode='binary')
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

#Convert class vector to binary class matrix for use with categorical_crossentropy
y_train = to_categorical(y_train, num_classes) # to_categorical: y_train을 46가지로 분류
y_test = to_categorical(y_test, num_classes)
print('y_train shape:', y_train.shape)
print('y_test shape:', y_test.shape)


# In[8]:


#2. model design for relu activation 
from keras.datasets import reuters
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers.noise import AlphaDropout

# week2 ppt 참조
model = Sequential()
model.add(Dense(16, input_shape=(1000,), # max_word = 1000과 동일
                    kernel_initializer='glorot_uniform')) 
# kernel_initialize: glorot_uniform - limit -> fan_in: 1000, fan_out:16 -> 총 16,000건 
model.add(Activation('relu'))
model.add(Dropout(0.5))

for i in range(5):
    model.add(Dense(16, kernel_initializer='glorot_uniform'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

model.add(Dense(46))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', 
                  optimizer='sgd', # sgd: Stochastic Gradient Descent
                  metrics=['accuracy'])

history_model_relu = model.fit(x_train,
                            y_train,
                            batch_size=16,
                            epochs=40,
                            verbose=1,
                            validation_split=0.1)

score_model_relu = model.evaluate(x_test,
                               y_test,
                               batch_size=16,
                               verbose=1)


# In[9]:


model = Sequential()
model.add(Dense(16, input_shape=(1000,),
                    kernel_initializer='lecun_normal'))   
model.add(Activation('selu'))
model.add(AlphaDropout(0.1))

for i in range(5):
    model.add(Dense(16, kernel_initializer='lecun_normal'))
    model.add(Activation('selu'))  # selu: self normalization 
    model.add(AlphaDropout(0.1))

model.add(Dense(46))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
                  optimizer='sgd',
                  metrics=['accuracy'])

history_model_selu = model.fit(x_train,
                            y_train,
                            batch_size=16,
                            epochs=40,
                            verbose=1,
                            validation_split=0.1)

score_model_selu = model.evaluate(x_test,
                               y_test,
                               batch_size=16,
                               verbose=1)


# In[10]:


print('relu result')
print('Test score:', score_model_relu[0])
print('Test accuracy:', score_model_relu[1])
print('selu result')
print('Test score:', score_model_selu[0])
print('Test accuracy:', score_model_selu[1])

epochs=40
plt.figure()
plt.plot(range(epochs),
         history_model_relu.history['val_loss'],
         'g-',
         label='relu Val Loss')
plt.plot(range(epochs),
         history_model_selu.history['val_loss'],
         'r-',
         label='selu Val Loss')
plt.plot(range(epochs),
         history_model_relu.history['loss'],
         'g--',
         label='relu Loss')
plt.plot(range(epochs),
         history_model_selu.history['loss'],
         'r--',
         label='selu Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

