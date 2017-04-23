
# coding: utf-8

# <h1 align="center"> Deep Learning gender from name - RNN LSTMs </h1>

# #### we will use an LSTM RNN to learn gender as f(name). we will use a stacked LSTM with many-to-one architecture feeding charecter inputs and predicting a binary outcome M/F. loss function used will be binary_crossentropy (a special case of categorical_crossentropy with m=2) and using adam optimizer (modified SGD) sample input /output would like this <br> ['r','a','k','e','s','h',' '] - male<br> ['p','r','a','d','e','e','p'] - male<br> ['g','a','n','g','a',' '] - female<br> and so on...

# <img src="LSTM_RNN_architecture.jpg" width="800" height="600"/>
regexp applied
[^a-zA-Z0-9 ,.\r\n] = remove
[ ]+ = ' '
[^a-zA-Z ,.\r\n] = remove
[ ]{3}+ - regex to check where 3 consecutive space occurs.
# In[199]:

from __future__ import print_function

from sklearn.preprocessing import OneHotEncoder
from keras.layers.core import Dense, Activation, Dropout
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.datasets import imdb
import pandas as pd
import numpy as np
import os


# In[86]:

#parameters
maxlen = 30
labels = 2


# In[158]:

input = pd.read_csv("gender_data.csv",header=None)
input.columns = ['name','m_or_f']
input['namelen']= [len(str(i)) for i in input['name']]
input1 = input[(input['namelen'] >= 2) ]


# In[159]:

input1.groupby('m_or_f')['name'].count()


# In[160]:

names = input['name']
gender = input['m_or_f']
vocab = set(' '.join([str(i) for i in names]))
vocab.add('END')
len_vocab = len(vocab)


# In[161]:

print(vocab)
print("vocab length is ",len_vocab)
print ("length of input is ",len(input1))


# In[162]:

char_index = dict((c, i) for i, c in enumerate(vocab))


# In[163]:

print(char_index)


# In[164]:

#train test split
msk = np.random.rand(len(input1)) < 0.8
train = input1[msk]
test = input1[~msk]     


# In[165]:

#take input upto max and truncate rest
#encode to vector space(one hot encoding)
#padd 'END' to shorter sequences
train_X = []
trunc_train_name = [str(i)[0:30] for i in train.name]
for i in trunc_train_name:
    tmp = [char_index[j] for j in str(i)]
    for k in range(0,maxlen - len(str(i))):
        tmp.append(char_index["END"])
    train_X.append(tmp)


# In[166]:

np.asarray(train_X).shape


# In[179]:

def set_flag(i):
    tmp = np.zeros(39);
    tmp[i] = 1
    return(tmp)


# In[184]:

set_flag(3)


# #### modify the code above to also convert each index to one-hot encoded representation

# In[195]:

#take input upto max and truncate rest
#encode to vector space(one hot encoding)
#padd 'END' to shorter sequences
#also convert each index to one-hot encoding
train_X = []
train_Y = []
trunc_train_name = [str(i)[0:maxlen] for i in train.name]
for i in trunc_train_name:
    tmp = [set_flag(char_index[j]) for j in str(i)]
    for k in range(0,maxlen - len(str(i))):
        tmp.append(set_flag(char_index["END"]))
    train_X.append(tmp)
for i in train.m_or_f:
    if i == 'm':
        train_Y.append([1,0])
    else:
        train_Y.append([0,1])
    


# In[196]:

np.asarray(train_X).shape


# In[197]:

np.asarray(train_Y).shape


# #### build model in keras ( a stacked LSTM model with many-to-one arch ) here 30 sequence and 2 output each for one category(m/f)

# In[212]:

#build the model: 2 stacked LSTM
print('Build model...')
model = Sequential()
model.add(LSTM(512, return_sequences=True, input_shape=(maxlen,len_vocab)))
model.add(Dropout(0.2))
model.add(LSTM(512, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(2))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])


# In[206]:

test_X = []
test_Y = []
trunc_test_name = [str(i)[0:maxlen] for i in test.name]
for i in trunc_test_name:
    tmp = [set_flag(char_index[j]) for j in str(i)]
    for k in range(0,maxlen - len(str(i))):
        tmp.append(set_flag(char_index["END"]))
    test_X.append(tmp)
for i in test.m_or_f:
    if i == 'm':
        test_Y.append([1,0])
    else:
        test_Y.append([0,1])
    


# In[207]:

print(np.asarray(test_X).shape)
print(np.asarray(test_Y).shape)


# In[215]:

batch_size=1000
model.fit(train_X, train_Y,batch_size=batch_size,nb_epoch=10,validation_data=(test_X, test_Y))


# In[216]:

score, acc = model.evaluate(test_X, test_Y)
print('Test score:', score)
print('Test accuracy:', acc)


# In[288]:

name=["sandhya","jaspreet","rajesh"]
X=[]
trunc_name = [i[0:maxlen] for i in name]
for i in trunc_name:
    tmp = [set_flag(char_index[j]) for j in str(i)]
    for k in range(0,maxlen - len(str(i))):
        tmp.append(set_flag(char_index["END"]))
    X.append(tmp)
pred=model.predict(np.asarray(X))


# In[289]:

pred


# #### Lets train more, clearly some very simple female names it doesnt get right like mentioned above (inspite it exists in training data)

# In[290]:

batch_size=1000
model.fit(train_X, train_Y,batch_size=batch_size,nb_epoch=50,validation_data=(test_X, test_Y))


# In[460]:

score, acc = model.evaluate(test_X, test_Y)
print('Test score:', score)
print('Test accuracy:', acc)


# <h3 align="center"> lets look at the loss and accuracy chart as a function of epochs </h3><img src="loss_charts.bmp" alt="loss charts" width="500" height="350"/><img src="acc_charts.bmp" alt="loss charts"  width="500" height="350"/>

# In[342]:

name=["sandhya","jaspreet","rajesh","kaveri","aditi deepak","arihant","sasikala","aditi","ragini rajaram"]
X=[]
trunc_name = [i[0:maxlen] for i in name]
for i in trunc_name:
    tmp = [set_flag(char_index[j]) for j in str(i)]
    for k in range(0,maxlen - len(str(i))):
        tmp.append(set_flag(char_index["END"]))
    X.append(tmp)
pred=model.predict(np.asarray(X))
pred


# In[345]:

name=["abhi","abhi deepak","mr. abhi"]
X=[]
trunc_name = [i[0:maxlen] for i in name]
for i in trunc_name:
    tmp = [set_flag(char_index[j]) for j in str(i)]
    for k in range(0,maxlen - len(str(i))):
        tmp.append(set_flag(char_index["END"]))
    X.append(tmp)
pred=model.predict(np.asarray(X))
pred


# In[502]:

name=["rajini","rajinikanth","mr. rajini"]
X=[]
trunc_name = [i[0:maxlen] for i in name]
for i in trunc_name:
    tmp = [set_flag(char_index[j]) for j in str(i)]
    for k in range(0,maxlen - len(str(i))):
        tmp.append(set_flag(char_index["END"]))
    X.append(tmp)
pred=model.predict(np.asarray(X))
pred


# In[450]:

#save our model and data
model.save_weights('gender_model',overwrite=True)
train.to_csv("train_split.csv")
test.to_csv("test_split.csv")


# In[464]:

evals = model.predict(test_X)
prob_m = [i[0] for i in evals]


# In[479]:

out = pd.DataFrame(prob_m)
out['name'] = test.name.reset_index()['name']
out['m_or_f']=test.m_or_f.reset_index()['m_or_f']


# In[483]:

out.head(10)
out.columns = ['prob_m','name','actual']
out.head(10)
out.to_csv("gender_pred_out.csv")


# In[ ]:



