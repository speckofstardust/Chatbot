import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from keras.models import load_model

lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.h5')


def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

def bag_of_words (sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class (sentence):
    bow = bag_of_words (sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.10
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes [r[0]], 'probability': str(r[1])})
    return return_list

def get_response(intents_list, intents_json):
    result = "I'm sorry, I didn't understand that"
    if intents_list:
        tag = intents_list[0]['intent']
        list_of_intents = intents_json['intents']
        for i in list_of_intents:
            if i['tag'] == tag:
                result = random.choice (i['responses'])
                break
    print(f"Predicted Intent: {tag}")
    return result

exit_conditions=("bye", "exit", "goodbye", "that's all","see ya")

from tkinter import *

# GUI
root = Tk()
root.title("Chatbot")

BG_GRAY = "#ABB2B9"
BG_COLOR = "#17202A"
TEXT_COLOR = "#EAECEE"

FONT = "Helvetica 14"
FONT_BOLD = "Helvetica 13 bold"

# Send function
def send():
    user_input_display = "You -> " + e.get()
    txt.insert(END, "\n" + user_input_display + "\n")
    message = e.get().lower()
    ints = predict_class(message)
    res = get_response(ints, intents)
    txt.insert(END, "\n" + "Felicia: " + res+"\n")
    
    e.delete(0, END)

label1 = Label(root, bg=BG_COLOR, fg=TEXT_COLOR, text="Welcome", font=FONT_BOLD, pady=10, width=20, height=1).grid(
    row=0, column=0, columnspan=2)

txt = Text(root, bg=BG_COLOR, fg=TEXT_COLOR, font=FONT, width=60, height=20, wrap='word')
txt.grid(row=1, column=0, sticky=N+S+E+W)

txt.insert(END,"Hi! I'm Felicia the first aid bot(^^) \nHow can I help you?\n")

scrollbar = Scrollbar(root, orient = 'vertical', command=txt.yview)
scrollbar.grid(row=1, column=1, sticky=N+S)

txt.config(yscrollcommand=scrollbar.set)

e = Entry(root, bg="#2C3E50", fg=TEXT_COLOR, font=FONT, width=55)
e.grid(row=2, column=0)

send_button = Button(root, text="Send", font=FONT_BOLD, bg=BG_GRAY, command=send)
send_button.grid(row=2, column=1)

root.mainloop()

    