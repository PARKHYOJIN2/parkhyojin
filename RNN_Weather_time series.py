
# coding: utf-8

# # GPU 할당

# In[1]:


import tensorflow as tf 
from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto()
config.gpu_options.visible_device_list='2' # 학생 개별 GPU NUMBER
config.gpu_options.per_process_gpu_memory_fraction = 0.25
set_session(tf.Session(config=config))


# # Import

# In[2]:


import pandas as pd
from functools import reduce
import numpy as np
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras import layers
from keras import regularizers
from keras.optimizers import RMSprop
import time


# # 데이터 불러오기

# In[3]:


get_ipython().system('tar -zxvf SEOUL_2008_2017.tar.gz')


# In[4]:


weather_data = []
# 10개의 csv파일을 pandas dataframe으로 Load
for i in range(10):
    weather_data.append(pd.read_csv('SEOUL_2008_2017/SEOUL_'+str(int(2008+i))+'.csv', encoding='euc-kr'))

# 10개 테이블 한개로 Merge
weather_df = reduce(lambda  left,right: pd.concat([left,right]), weather_data)


# # 데이터 미리보기(Pandas Dataframe)

# In[5]:


weather_df[:10]


# In[6]:


weather_df.columns


# In[7]:


weather_df = weather_df.drop(['지점'], axis=1)
weather_df = weather_df.drop(['일시'], axis=1)
weather_df = weather_df.drop(['일사(MJ/m2)'], axis=1)
weather_df = weather_df.drop(['적설(cm)'], axis=1)
weather_df = weather_df.drop(['3시간신적설(cm)'], axis=1)
weather_df = weather_df.drop(['운형(운형약어)'], axis=1)
weather_df = weather_df.drop(['최저운고(100m )'], axis=1)
weather_df = weather_df.drop(['지면상태(지면상태코드)'], axis=1)
weather_df = weather_df.drop(['현상번호(국내식)'], axis=1)


# # 데이터 전처리1

# ## NaN 처리

# In[8]:


# 강수량 NaN 처리: 0으로 채움
weather_df = weather_df.fillna({'강수량(mm)':0})
# 기타 NaN 처리
weather_df = weather_df.apply(lambda x: x.fillna(x.rolling(min_periods=1, center=True, window=48).mean()), axis=0)
# Feature 줄이기(몇 개의 컬럼 제거)
#weather_df = weather_df.drop(['지점','일시','일사(MJ/m2)', '적설(cm)','3시간신적설(cm)','운형(운형약어)','최저운고(100m )','지면상태(지면상태코드)','현상번호(국내식)'], axis=1)


# ## 전처리 후 데이터 미리보기(Pandas Dataframe)

# In[9]:


weather_df[:10]


# ## Pandas Dataframe to Numpy Array

# In[10]:


weather_df_value = weather_df.values
weather_df_value


# ## 전처리 후 데이터 미리보기(Numpy)

# In[11]:


weather_df_value[:10]


# ## 전처리 후 데이터 미리보기(Plot)

# ### 전체 데이터 Plot

# In[14]:


temp = weather_df_value[:, 0] # 온도
plt.plot(range(len(temp)), temp)


# ### 처음 10일간 온도 Plot

# In[15]:


plt.plot(range(240), temp[: 240]) # 1시간마다 데이터가 기록되므로 하루에 24시간, 10일 = 240이다. 따라서 2008년 1월 1일~1월 10일의 온도 그래프다.


# ## 데이터 정규화

# In[16]:


# Train 6년[2008~2013](0~52622)/ Validation 2년[2014~2015](52623~70142)/ Test 2년[2016~2017](70143~87685)
mean = weather_df_value[:52622].mean(axis=0)
weather_df_value -= mean
std = weather_df_value[:52622].std(axis=0)
weather_df_value /= std


# ## 정규화 후 데이터 미리보기(Numpy)

# In[14]:


weather_df_value[:10]
weather_df_value.shape


# # 데이터 전처리2

# ## Generator(PPT 참고)

# In[19]:


'''
data : 원본 배열
lookback : 입력으로 사용하기 위해 거슬러 올라갈 타임스텝,
           10일간의 데이터를 사용한다면 1시간에 1번 기록하므로 24 * 10 = 240이 된다.
delay : 타깃으로 사용할 미래의 타임스텝, 24시간이 지난 데이터가 타깃이 된다면 24
min_index와 max_index : 추출할 타임스텝의 범위를 지정하기 위한 data 배열의 인덱스.
                        데이터와 테스트 데이터를 분리하는 데 사용합니다.
shuffle : 샘플을 섞을지 시간 순서대로 추출할지 결정합니다.
batch_size : 배치의 샘플 수
step : 데이터를 샘플링할 타임스텝 간격. 
       한 시간에 하나의 데이터 포인트를 추출하기 위해 1으로 지정하겠습니다.

열흘(10일, 240시간)치의 데이터를 입력으로 해서, 하루(1일, 24시간)의 기온을 
예측하려는 것
'''
def generator(data, lookback, delay, min_index, max_index,
              shuffle=False, batch_size=64, step=1):
    if max_index is None:
        max_index = len(data) - delay - 1 
    i = min_index + lookback 
    while 1:
        if shuffle:
            rows = np.random.randint(
                min_index + lookback, max_index, size=batch_size) 
        else:
            if i + batch_size >= max_index:
                i = min_index + lookback
            rows = np.arange(i, min(i + batch_size, max_index))
            i += len(rows)

        samples = np.zeros((len(rows),
                           lookback // step,
                           data.shape[-1]))
        targets = np.zeros((len(rows),)) 
        for j, row in enumerate(rows): 
            indices = range(rows[j] - lookback, rows[j], step)
            samples[j] = data[indices]
            targets[j] = data[rows[j] + delay][0]
        yield samples, targets


# ## Train, Validation, Test Data Generator

# In[24]:


# Train 6년[2008~2013](0~52622)/ Validation 2년[2014~2015](52623~70142)/ Test 2년[2016~2017](70143~87685)
def data_gen(data, lookback = 240, step = 1, delay = 24, batch_size = 128):
    train_gen = generator(data,
                          lookback=lookback,
                          delay=delay,
                          min_index=0,
                          max_index=52622,
                          shuffle=True,
                          step=step, 
                          batch_size=batch_size)
    val_gen = generator(data,
                        lookback=lookback,
                        delay=delay,
                        min_index=52623,
                        max_index=70142,
                        step=step,
                        batch_size=batch_size)
    test_gen = generator(data,
                         lookback=lookback,
                         delay=delay,
                         min_index=70143,
                         max_index=None,
                         step=step,
                         batch_size=batch_size)
    # 전체 검증 세트를 순회하기 위해 val_gen에서 추출할 횟수, lookback을 빼는 이유는 lookback 개수씩 timestep을 pick하기 때문에
    val_steps = (70142 - 52623 - lookback) // batch_size
    
    # 전체 테스트 세트를 순회하기 위해 test_gen에서 추출할 횟수
    test_steps = (len(data) - 70143 - lookback) // batch_size
    return train_gen, val_gen, test_gen, val_steps, test_steps


# ## 데이터 Generator 생성

# In[25]:


train_gen, val_gen, test_gen, val_steps, test_steps = data_gen(weather_df_value, lookback = 240, step = 1, delay = 24, batch_size = 128)


# # 학습하기

# ## Train Loss, Validation Loss 그래프 함수

# In[26]:


def plot_train_validation_loss(history, subtitle):
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    epochs = range(1, len(loss) + 1)
    
    plt.figure()
    
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss: '+subtitle)
    plt.legend()
    
    plt.show() 


# In[27]:


lookback = 240
step = 1
delay = 24
batch_size = 128


# ## 학습하기(Dense)

# In[28]:


start_time = time.time() 

model = Sequential()
model.add(layers.Flatten(input_shape=(lookback // step, weather_df_value.shape[-1])))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(1))

model.compile(optimizer=RMSprop(), loss='mae')
history = model.fit_generator(train_gen,
                              steps_per_epoch=400,
                              epochs=20,
                              validation_data=val_gen,
                              validation_steps=val_steps)

print("--- %s seconds ---" %(time.time() - start_time))


# ## Train Loss, Validation Loss 그래프

# In[49]:


plot_train_validation_loss(history,'Dense')


# ## CuDNNLSTM

# ### 학습하기(CuDNNLSTM)

# In[51]:


start_time = time.time() 

model = Sequential()
model.add(layers.Flatten(input_shape=(lookback // step, weather_df_value.shape[-1])))
model.add(layers.CuDNNLSTM(32, input_shape=(None, weather_df_value.shape[-1])))
model.add(layers.Dense(1))

model.compile(optimizer=RMSprop(), loss='mae')
history = model.fit_generator(train_gen,
                              steps_per_epoch=400,
                              epochs=20,
                              validation_data=val_gen,
                              validation_steps=val_steps)

print("--- %s seconds ---" %(time.time() - start_time))


# ### Train Loss, Validation Loss 그래프

# In[ ]:


plot_train_validation_loss(history, 'CuDNNLSTM')


# ### 학습하기(CuDNNLSTM, Stacking)

# In[ ]:


start_time = time.time() 

model = Sequential()
model.add(layers.CuDNNLSTM(32,input_shape=(None, weather_df_value.shape[-1]), return_sequences=True))
model.add(layers.CuDNNLSTM(64,input_shape=(None, weather_df_value.shape[-1])))
model.add(layers.Dense(1))

model.compile(optimizer=RMSprop(), loss='mae')
history = model.fit_generator(train_gen,
                              steps_per_epoch=400,
                              epochs=20,
                              validation_data=val_gen,
                              validation_steps=val_steps)

print("--- %s seconds ---" %(time.time() - start_time))


# ### Train Loss, Validation Loss 그래프

# In[ ]:


plot_train_validation_loss(history, 'CuDNNLSTM, Stacking')


# ### 학습하기(CuDNNLSTM, Bidirectional)

# In[ ]:


start_time = time.time() 

model = Sequential()
model.add(layers.Bidirectional(layers.CuDNNLSTM(32),input_shape=(None, weather_df_value.shape[-1])))
model.add(layers.Dense(1))

model.compile(optimizer=RMSprop(), loss='mae')
history = model.fit_generator(train_gen,
                              steps_per_epoch=400,
                              epochs=20,
                              validation_data=val_gen,
                              validation_steps=val_steps)

print("--- %s seconds ---" %(time.time() - start_time))


# ### Train Loss, Validation Loss 그래프

# In[ ]:


plot_train_validation_loss(history, 'CuDNNLSTM, Bidirectional')


# ### 학습하기(CuDNNLSTM, Network Down Sizing)

# In[ ]:


start_time = time.time() 

model = Sequential()
# Downsizing(32 -> 16)
model.add(layers.CuDNNLSTM(16, input_shape=(None, weather_df_value.shape[-1])))
model.add(layers.Dense(1))

model.compile(optimizer=RMSprop(), loss='mae')
history = model.fit_generator(train_gen,
                              steps_per_epoch=400,
                              epochs=20,
                              validation_data=val_gen,
                              validation_steps=val_steps)

print("--- %s seconds ---" %(time.time() - start_time))


# ### Train Loss, Validation Loss 그래프

# In[ ]:


plot_train_validation_loss(history, 'CuDNNLSTM, Network Down Sizing')


# ### 학습하기(CuDNNLSTM, Recurrent Regularizer)

# In[ ]:


start_time = time.time() 

model = Sequential()
model.add(layers.CuDNNLSTM(16, input_shape=(None, weather_df_value.shape[-1]), recurrent_regularizer=regularizers.l2(0.001)))
model.add(layers.Dense(1))

model.compile(optimizer=RMSprop(), loss='mae')
history = model.fit_generator(train_gen,
                              steps_per_epoch=400,
                              epochs=20,
                              validation_data=val_gen,
                              validation_steps=val_steps)

print("--- %s seconds ---" %(time.time() - start_time))


# ### Train Loss, Validation Loss 그래프

# In[ ]:


plot_train_validation_loss(history, 'CuDNNLSTM, Recurrent Regularizer')


# ### 학습하기(CuDNNLSTM, Kernel Reguliarzer, Recurrent Regularizer)

# In[ ]:


start_time = time.time() 

model = Sequential()
model.add(layers.CuDNNLSTM(16, input_shape=(None, weather_df_value.shape[-1]), kernel_regularizer=regularizers.l2(0.01), recurrent_regularizer=regularizers.l2(0.001)))
model.add(layers.Dense(1))

model.compile(optimizer=RMSprop(), loss='mae')
history = model.fit_generator(train_gen,
                              steps_per_epoch=400,
                              epochs=20,
                              validation_data=val_gen,
                              validation_steps=val_steps)

print("--- %s seconds ---" %(time.time() - start_time))


# ### Train Loss, Validation Loss 그래프

# In[ ]:


plot_train_validation_loss(history, 'CuDNNLSTM, Kernel Reguliarzer, Recurrent Regularizer')


# ### 학습하기(Conv1D, CuDNNLSTM, Kernel Reguliarzer, Recurrent Regularizer)

# In[ ]:


start_time = time.time() 

model = Sequential()
model.add(layers.Conv1D(16, 5, activation='relu', input_shape=(None, weather_df_value.shape[-1])))
model.add(layers.MaxPooling1D(3))
model.add(layers.Conv1D(16, 5, activation='relu'))
model.add(layers.CuDNNLSTM(16, kernel_regularizer=regularizers.l2(0.01), recurrent_regularizer=regularizers.l2(0.001)))
model.add(layers.Dense(1))

model.compile(optimizer=RMSprop(), loss='mae')
history = model.fit_generator(train_gen,
                              steps_per_epoch=400,
                              epochs=20,
                              validation_data=val_gen,
                              validation_steps=val_steps)

print("--- %s seconds ---" %(time.time() - start_time))


# ### Train Loss, Validation Loss 그래프

# In[ ]:


plot_train_validation_loss(history, 'Conv1D, CuDNNLSTM, Kernel Reguliarzer, Recurrent Regularizer')


# ### 학습하기(Conv1D, Dropout, BatchNormalization, CuDNNLSTM, Kernel Reguliarzer, Recurrent Regularizer)

# In[ ]:


start_time = time.time() 

model = Sequential()
model.add(layers.Conv1D(16, 5, activation=None, input_shape=(None, weather_df_value.shape[-1])))
model.add(layers.BatchNormalization())
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling1D(3))
model.add(layers.Dropout(0.2))
model.add(layers.Conv1D(16, 5, activation=None))
model.add(layers.BatchNormalization())
model.add(layers.Activation('relu'))
model.add(layers.CuDNNLSTM(16, kernel_regularizer=regularizers.l2(0.01), recurrent_regularizer=regularizers.l2(0.001)))
model.add(layers.Dense(1))

model.compile(optimizer=RMSprop(), loss='mae')
history = model.fit_generator(train_gen,
                              steps_per_epoch=400,
                              epochs=20,
                              validation_data=val_gen,
                              validation_steps=val_steps)

print("--- %s seconds ---" %(time.time() - start_time))


# ### Train Loss, Validation Loss 그래프

# In[ ]:


plot_train_validation_loss(history, 'Conv1D, Dropout, BatchNormalization, CuDNNLSTM, Kernel Reguliarzer, Recurrent Regularizer')


# ## CuDNNGRU

# ### 학습하기(CuDNNGRU)

# In[29]:


start_time = time.time() 

model = Sequential()
model.add(layers.CuDNNGRU(32, input_shape=(None, weather_df_value.shape[-1])))
model.add(layers.Dense(1))

model.compile(optimizer=RMSprop(), loss='mae')
history = model.fit_generator(train_gen,
                              steps_per_epoch=400,
                              epochs=20,
                              validation_data=val_gen,
                              validation_steps=val_steps)

print("--- %s seconds ---" %(time.time() - start_time))


# ### Train Loss, Validation Loss 그래프

# In[ ]:


plot_train_validation_loss(history, 'CuDNNGRU')


# ### 학습하기(CuDNNGRU, Network Down Sizing)

# In[30]:


start_time = time.time() 

model = Sequential()
# Downsizing(32 -> 16)
model.add(layers.CuDNNGRU(16, input_shape=(None, weather_df_value.shape[-1])))
model.add(layers.Dense(1))

model.compile(optimizer=RMSprop(), loss='mae')
history = model.fit_generator(train_gen,
                              steps_per_epoch=400,
                              epochs=20,
                              validation_data=val_gen,
                              validation_steps=val_steps)

print("--- %s seconds ---" %(time.time() - start_time))


# ### Train Loss, Validation Loss 그래프

# In[ ]:


plot_train_validation_loss(history, 'CuDNNGRU, Network Down Sizing')


# ### 학습하기(CuDNNGRU, Recurrent Regularizer)

# In[ ]:


start_time = time.time() 

model = Sequential()
model.add(layers.CuDNNGRU(16, input_shape=(None, weather_df_value.shape[-1]), recurrent_regularizer=regularizers.l2(0.001)))
model.add(layers.Dense(1))

model.compile(optimizer=RMSprop(), loss='mae')
history = model.fit_generator(train_gen,
                              steps_per_epoch=400,
                              epochs=20,
                              validation_data=val_gen,
                              validation_steps=val_steps)

print("--- %s seconds ---" %(time.time() - start_time))


# ### Train Loss, Validation Loss 그래프

# In[ ]:


plot_train_validation_loss(history, 'CuDNNGRU, Recurrent Regularizer')


# ### 학습하기(CuDNNGRU, Kernel Reguliarzer, Recurrent Regularizer)

# In[ ]:


start_time = time.time() 

model = Sequential()
model.add(layers.CuDNNGRU(16, input_shape=(None, weather_df_value.shape[-1]), kernel_regularizer=regularizers.l2(0.01), recurrent_regularizer=regularizers.l2(0.001)))
model.add(layers.Dense(1))

model.compile(optimizer=RMSprop(), loss='mae')
history = model.fit_generator(train_gen,
                              steps_per_epoch=400,
                              epochs=20,
                              validation_data=val_gen,
                              validation_steps=val_steps)

print("--- %s seconds ---" %(time.time() - start_time))


# ### Train Loss, Validation Loss 그래프

# In[ ]:


plot_train_validation_loss(history, 'CuDNNGRU, Kernel Reguliarzer, Recurrent Regularizer')


# ### 학습하기(Conv1D, CuDNNGRU, Kernel Reguliarzer, Recurrent Regularizer)

# In[ ]:


start_time = time.time() 

model = Sequential()
model.add(layers.Conv1D(16, 5, activation='relu', input_shape=(None, weather_df_value.shape[-1])))
model.add(layers.MaxPooling1D(3))
model.add(layers.Conv1D(16, 5, activation='relu'))
model.add(layers.CuDNNGRU(16, kernel_regularizer=regularizers.l2(0.01), recurrent_regularizer=regularizers.l2(0.001)))
model.add(layers.Dense(1))

model.compile(optimizer=RMSprop(), loss='mae')
history = model.fit_generator(train_gen,
                              steps_per_epoch=400,
                              epochs=20,
                              validation_data=val_gen,
                              validation_steps=val_steps)

print("--- %s seconds ---" %(time.time() - start_time))


# ### Train Loss, Validation Loss 그래프

# In[ ]:


plot_train_validation_loss(history, 'Conv1D, CuDNNGRU, Kernel Reguliarzer, Recurrent Regularizer')


# ### 학습하기(Conv1D, Dropout, BatchNormalization, CuDNNGRU, Kernel Reguliarzer, Recurrent Regularizer)

# In[ ]:


start_time = time.time() 

model = Sequential()
model.add(layers.Conv1D(16, 5, activation=None, input_shape=(None, weather_df_value.shape[-1])))
model.add(layers.BatchNormalization())
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling1D(3))
model.add(layers.Dropout(0.2))
model.add(layers.Conv1D(16, 5, activation=None))
model.add(layers.BatchNormalization())
model.add(layers.Activation('relu'))
model.add(layers.CuDNNGRU(16, kernel_regularizer=regularizers.l2(0.01), recurrent_regularizer=regularizers.l2(0.001)))
model.add(layers.Dense(1))

model.compile(optimizer=RMSprop(), loss='mae')
history = model.fit_generator(train_gen,
                              steps_per_epoch=400,
                              epochs=20,
                              validation_data=val_gen,
                              validation_steps=val_steps)

print("--- %s seconds ---" %(time.time() - start_time))


# ### Train Loss, Validation Loss 그래프

# In[ ]:


plot_train_validation_loss(history, 'Conv1D, Dropout, BatchNormalization, CuDNNGRU, Kernel Reguliarzer, Recurrent Regularizer')


# ## LSTM, GRU, Recurrent Dropout

# ### 학습하기(LSTM, Recurrent Dropout 적용, 시간이 오래 걸리므로 모든 실습 뒤에 실행해볼 것)

# In[ ]:


start_time = time.time() 

model = Sequential()
model.add(layers.LSTM(32, dropout=0.2, recurrent_dropout=0.2, input_shape=(None, weather_df_value.shape[-1])))
model.add(layers.Dense(1))

model.compile(optimizer=RMSprop(), loss='mae')
history = model.fit_generator(train_gen,
                              steps_per_epoch=400,
                              epochs=40,
                              validation_data=val_gen,
                              validation_steps=val_steps)

print("--- %s seconds ---" %(time.time() - start_time))


# ### Train Loss, Validation Loss 그래프

# In[ ]:


plot_train_validation_loss(history, 'LSTM, Recurrent Dropout')


# ### 학습하기(GRU, Recurrent Dropout 적용, 시간이 오래 걸리므로 모든 실습 뒤에 실행해볼 것)

# In[ ]:


start_time = time.time() 

model = Sequential()
model.add(layers.GRU(32,
                     dropout=0.2,
                     recurrent_dropout=0.2,
                     input_shape=(None, weather_df_value.shape[-1])))
model.add(layers.BatchNormalization())
model.add(layers.Dense(1))

model.compile(optimizer=RMSprop(), loss='mae')
history = model.fit_generator(train_gen,
                              steps_per_epoch=400,
                              epochs=40,
                              validation_data=val_gen,
                              validation_steps=val_steps)

print("--- %s seconds ---" %(time.time() - start_time))


# ### Train Loss, Validation Loss 그래프

# In[ ]:


plot_train_validation_loss(history, 'GRU, Recurrent Dropout')

