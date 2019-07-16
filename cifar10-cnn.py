
# coding: utf-8

# # Practice2. CIFAR10

# In[1]:


import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="2"

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = .24
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

from keras import backend as K
K.set_session(session)


# In[2]:


import cv2
import matplotlib.pyplot as plt
import numpy as np

# set default plot options
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = (10.0, 8.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


# ## 1. Set up input preprocessing

# In[3]:


from utils import get_CIFAR10_data


# In[4]:


X_tr, Y_tr, X_te, Y_te, mean_img = get_CIFAR10_data()
print ('Train data shape : %s,  Train labels shape : %s' % (X_tr.shape, Y_tr.shape))
print ('Test data shape : %s,  Test labels shape : %s' % (X_te.shape, Y_te.shape))

classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


# ### Input data를 CNN Model의 Input에 맞는 3차원 형태의 데이터로 만들기

# In[10]:


print ('X_tr : ', X_tr.shape)
print ('X_te : ', X_te.shape)


# In[11]:


X_tr_input = X_tr.reshape(-1,32,32,3)# channel 갯수: 3
X_te_input = X_te.reshape(-1,32,32,3)


# In[12]:


print ('X_tr : ', X_tr_input.shape)
print ('X_te : ', X_te_input.shape)

print('np.shape(X_tr_input[0]) = {}'.format(X_tr_input[0].shape))


# ### Label data를 one-hot vector로 만들기

# In[13]:


print ('Y_tr : ', Y_tr[0:10])
print ('Y_te : ', Y_te[0:10])


# In[14]:


from keras.utils import to_categorical
Y_te_onehot = to_categorical(Y_te, num_classes=10)
Y_tr_onehot = to_categorical(Y_tr, num_classes=10)


# In[15]:


print ('Y_tr_onehot : \n', Y_tr_onehot[0:10])
print ('Y_te_onehot : \n', Y_te_onehot[0:10])
print ('Y_tr_onehot.shape : \n', Y_tr_onehot.shape)
print ('Y_te_onehot.shape : \n', Y_te_onehot.shape)


# ### Keras 모델 빌드하기
# #### 모델을 build 할 때 필요한 데이터 정보 가져오기

# In[16]:


input_shape = X_tr_input.shape[1:]
output_shape = Y_te_onehot.shape[1]

print ('input_shape : ', input_shape )
print ('output_shape : ', output_shape )


# #### Keras 모델 설계

# In[18]:


from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Activation, Add, Input, Average
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import plot_model

input_layer = Input(shape=input_shape)
        
layer = input_layer

layer = Conv2D(32, (3, 3), kernel_initializer='RandomNormal', padding='same')(layer) # (3, 3)를 32개 구성
layer = Activation('selu')(layer)
layer = MaxPooling2D((2,2))(layer)

layer = Conv2D(64, (3, 3), kernel_initializer='RandomNormal', padding='same')(layer)
layer = Activation('selu')(layer)
layer = MaxPooling2D((2,2))(layer)

layer = Conv2D(128, (3, 3), kernel_initializer='RandomNormal', padding='same')(layer) 
layer = Activation('selu')(layer)
layer = MaxPooling2D((2,2))(layer)

layer = Conv2D(128, (3, 3), kernel_initializer='RandomNormal', padding='same')(layer) 
layer = Activation('selu')(layer)
layer = MaxPooling2D((2,2))(layer)

layer = Flatten()(layer) # Flatten() : https://datascienceschool.net/view-notebook/17608f897087478bbeac096438c716f6/
layer = Dense(256, kernel_initializer='RandomNormal')(layer)
layer = Dropout(0.2)(layer)
layer = Activation('selu')(layer)
layer = Dense(256, kernel_initializer='RandomNormal')(layer)
layer = Dropout(0.2)(layer)
layer = Activation('selu')(layer)
layer = Dense(output_shape, kernel_initializer='RandomNormal', activation='softmax')(layer)

output_layer = layer
model = Model(inputs=[input_layer], outputs=[output_layer])


# #### Optimizer  및 Loss function 선택 후 Model 컴파일

# In[ ]:


from keras.optimizers import SGD
sgd=SGD(lr=0.01, momentum=0.001, decay=0.0001, nesterov=True)
model.compile(loss='categorical_cros', optimizer=sgd ,metrics=['accuracy'])


# #### 컴파일된 모델 정보 출력 1

# In[21]:


print (model.summary())


# #### Training model

# In[24]:


from keras.callbacks import ModelCheckpoint

is_train = True

if is_train:
    filepath = './checkpoints/model-{epoch:02d}-{val_acc:.2f}.hdf5'
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks = [checkpoint]

    history = model.fit(X_tr_input, Y_tr_onehot,validation_data=[X_te_input, Y_te_onehot], epochs=200, batch_size=128, verbose=1, callbacks=callbacks)
    
    # list all data in history
    print(history.history.keys())

    plt.figure(figsize=[10,8])
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    plt.figure(figsize=[10,8])
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    
    print ('best accuracy : ', max(history.history['val_acc']))
else:
    filepath = './checkpoints/model-49-0.46.hdf5'
    model.load_weights(filepath=filepath)


# In[27]:


test_img = cv2.resize(cv2.imread('./cat.jpg'), (32, 32))
plt.imshow(test_img)

test_imgs = np.expand_dims(test_img, axis=0) # expand_dims : https://code.i-harness.com/ko/docs/numpy~1.13/generated/numpy.expand_dims
print('test_imgs.shape =', test_imgs.shape)

p = model.predict(test_imgs, batch_size=1, verbose=1)
print('p = {}'.format(p))
print('class = {}'.format(classes[np.argmax(p)])) # argmax : https://rfriend.tistory.com/356

