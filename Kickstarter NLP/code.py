from keras.optimizers import SGD, Adam, Nadam, RMSprop
from keras.models import Sequential,Model,load_model
from keras.layers import Embedding,Conv1D,MaxPooling1D
from keras.layers.core import Dense, Activation,Dropout ,Flatten
from keras.layers.recurrent import LSTM
from keras.layers.recurrent import GRU
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import sequence
from keras.preprocessing.text import text_to_word_sequence,one_hot,Tokenizer
from keras.constraints import maxnorm
from keras.callbacks import ModelCheckpoint,TensorBoard, ReduceLROnPlateau,EarlyStopping
from keras.applications import Xception
from keras import regularizers
from keras import backend as K
import keras
import numpy as np
import pandas as pd
import cv2
import os
import glob
import math
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
import nltk



def load_TrainingData(path):
    D = pd.read_csv(path, sep=',', header=0, engine='python')
    feature_names = np.array(list(D.columns.values))
    X_train = np.array(list(D['blurb']))
    Y_train = np.array(list(D['state']))
    #     X_train = X_train.reshape(X_train.shape[0] , 1)
    #     Y_train = Y_train.reshape(Y_train.shape[0] , 1)
    return X_train, Y_train, feature_names


# def load_TestingData(path):     #loads data , caluclate Mean & subtract it data, gets the COV. Matrix.
#     D = pd.read_csv(path, sep='\t', header=0)
#     X_test=np.array(list(D['Phrase']))
#     X_test_PhraseID=np.array(list(D['PhraseId']))
#     return  X_test,X_test_PhraseID

def shuffle_2(a, b):  # Shuffles 2 arrays with the same order
    s = np.arange(a.shape[0])
    np.random.shuffle(s)
    return a[s], b[s]


def loadGloveEmbeddings(fileName):
    embeddings_index = {}
    f = open(fileName)
    for line in f:
        values = line.split()
        word = values[0]
        value = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = value
    f.close()
    print('Loaded %s word vectors.' % len(embeddings_index))
    return embeddings_index


def removeStoppingWords(X, stop_index):
    allSentences = []
    for i in range(len(X)):
        sentenceIndices = [index for index in X[i] if index not in stop_indices]
        allSentences.append(sentenceIndices)
    return allSentences

#get stopping word indecies to be removed later
stop_words = set(stopwords.words('english'))
stop_indices = [word_index[stop] for stop in stop_words if stop in word_index ]


seed = 7
np.random.seed(seed)

X_train, Y_train, feature_names = load_TrainingData('./df_text_eng.csv')
# X_test,X_test_PhraseID = load_TestingData('./test.tsv')
print('============================== Training data shapes ==============================')
print('X_train.shape is ', X_train.shape)
print('Y_train.shape is ',Y_train.shape)

print("before" , X_train[10])
X_train , Y_train = shuffle_2(X_train , Y_train)
print("after" , X_train[10])

tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)

Tokenizer_vocab_size = len(tokenizer.word_index) + 1
print("Vocab size",Tokenizer_vocab_size)


################################ separate data into training, validation, and testing ######################################3
num_val = int(0.1 * X_train.shape[0])
mask = range(num_test)

Y_Val = Y_train[:num_val]
X_Val = X_train[:num_val]

X_train = X_train[num_val:]
Y_train = Y_train[num_val:]

Y_test = Y_train[:num_val]
X_test = X_train[:num_val]

X_train = X_train[num_val:]
Y_train = Y_train[num_val:]


maxWordCount= 28
maxDictionary_size=Tokenizer_vocab_size



# convert sentences with sequence of numbers with each number representing a word in the sentence
encoded_words = tokenizer.texts_to_sequences(X_train)
encoded_words2 = tokenizer.texts_to_sequences(X_Val)
encoded_words3 = tokenizer.texts_to_sequences(X_test)

# remove stopping words
encoded_words = removeStoppingWords(encoded_words,stop_indices)
encoded_words2 = removeStoppingWords(encoded_words2,stop_indices)
encoded_words3 = removeStoppingWords(encoded_words3,stop_indices)


#padding all text to same size
X_Train_encodedPadded_words = sequence.pad_sequences(encoded_words, maxlen=maxWordCount)
X_Val_encodedPadded_words = sequence.pad_sequences(encoded_words2, maxlen=maxWordCount)
X_test_encodedPadded_words = sequence.pad_sequences(encoded_words3, maxlen=maxWordCount)

# zero one encoding
Y_train = (Y_train == 'successful').astype(int)
Y_Val   = (Y_Val == 'successful').astype(int)
Y_test   = (Y_test == 'successful').astype(int)


#shuffling the traing Set
shuffle_2(X_Train_encodedPadded_words,Y_train)

print('Featu are ',feature_names)
print('============================== After extracting a validation set of '+ str(num_test)+' ============================== ')
print('============================== Training data shapes ==============================')
print('X_train.shape is ', X_train.shape)
print('Y_train.shape is ',Y_train.shape)
print('============================== Validation data shapes ==============================')
print('Y_Val.shape is ',Y_Val.shape)
print('X_Val.shape is ', X_Val.shape)
# print('============================== Test data shape ==============================')
print('X_test.shape is ', X_test.shape)





print('============================== After padding all text to same size of '+ str(maxWordCount)+' ==============================')
print('============================== Training data shapes ==============================')
print('X_train.shape is ', X_Train_encodedPadded_words.shape)
# print('Y_train.shape is ',Y_train.shape)
print('============================== Validation data shapes ==============================')
# print('Y_Val.shape is ',Y_Val.shape)
print('X_Val.shape is ', X_Val_encodedPadded_words.shape)
# print('============================== Test data shape ==============================')
print('X_test.shape is ', X_test_encodedPadded_words.shape)

######################### Add Embedding Layer ########################################
embeddings_index = loadGloveEmbeddings("glove.6B.100d.txt")
word_index = tokenizer.word_index
embedding_dimension = 100
embedding_matrix = np.zeros((len(word_index) + 1, embedding_dimension))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector[:embedding_dimension]

print(embedding_matrix.shape)


embedding_layer = Embedding(maxDictionary_size, embedding_dimension , input_length=maxWordCount , trainable=True )

embedding_layer.build((None,))

# Set the weights of the embedding layer to the embedding matrix. Your layer is now pretrained.
embedding_layer.set_weights([embedding_matrix])


###################### build the sequential model ###########################

model = Sequential()

model.add(embedding_layer) #to change words to ints

model.add(LSTM(128 , return_sequences =True) )
model.add(Dropout(0.5))


model.add(Dense(64, activation='relu',W_constraint=maxnorm(1)))

model.add(Dense(1, activation='sigmoid'))

model.summary()




learning_rate=0.0001

sgd = SGD(lr=learning_rate, nesterov=True, momentum=0.7, decay=1e-4)
Nadam = keras.optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)
model.compile(loss='binary_crossentropy', optimizer=Nadam, metrics=['accuracy'])


tensorboard = keras.callbacks.TensorBoard(log_dir='./logs/log_25', histogram_freq=0, write_graph=True, write_images=False)
checkpointer = ModelCheckpoint(filepath="models.{epoch:02d}-{val_acc:.2f}.hdf5", verbose=1, save_best_only=True, monitor="val_acc")
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=0, verbose=1, mode='auto', cooldown=0, min_lr=1e-6)
earlyStopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=6, verbose=1)



print ("=============================== Training =========================================")


epochs = 24
batch_size = 4096
history  = model.fit(X_Train_encodedPadded_words, Y_train, epochs = epochs, batch_size=batch_size, verbose=1,
                    validation_data=(X_Val_encodedPadded_words, Y_Val), callbacks=[tensorboard, checkpointer])




###################### evaluating the Results ##################################

scores = model.evaluate(X_test_encodedPadded_words, Y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
