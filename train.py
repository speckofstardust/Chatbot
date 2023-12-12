import random
import json
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from nltk.stem import WordNetLemmatizer
import nltk

lemmatizer = WordNetLemmatizer()
nltk.download('omw-1.4')
nltk.download('punkt')
nltk.download('wordnet')

intents = json.loads(open('intents.json').read())

words = []
classes = []
documents = []
ignoreLetters = ['?', '!', '.', ',','#']

for intent in intents['intents']:
    for pattern in intent['patterns']:
        # take each word and tokenize it
        wordList = nltk.word_tokenize(pattern)
        words.extend(wordList)
        documents.append((wordList, intent['tag']))
        # adding classes to our class list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# lemmatization
words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in ignoreLetters]
words = sorted(set(words))

classes = sorted(set(classes))

pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

training = []
outputEmpty = [0] * len(classes)

for document in documents:
    bag = []
    wordPatterns = document[0]
    wordPatterns = [lemmatizer.lemmatize(word.lower()) for word in wordPatterns]
    for word in words:
        bag.append(1) if word in wordPatterns else bag.append(0)

    outputRow = list(outputEmpty)
    outputRow[classes.index(document[1])] = 1
    training.append(bag + outputRow)

random.shuffle(training)
training = np.array(training)

trainX = training[:, :len(words)]
trainY = training[:, len(words):]

model = Sequential()
model.add(Dense(128, input_shape=(len(trainX[0]),), activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(len(trainY[0]), activation='softmax'))

adam = Adam(learning_rate=0.01)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

# Add Early Stopping
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

hist = model.fit(np.array(trainX), np.array(trainY), epochs=200, batch_size=5, verbose=1, validation_split=0.2, callbacks=[early_stopping])
model.save('chatbot_model.h5', hist)
print('Done')
