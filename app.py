from flask import Flask,render_template,request
from textblob import TextBlob
import os
import numpy as np
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, plot_confusion_matrix


train_dir = "train-mails"
test_dir = "test-mails"

# Making the dictionary and columns of the word_matrix

emails = [os.path.join(train_dir,f) for f in os.listdir(train_dir)]

all_words = []

for mail in emails:
    with open(mail) as m:
        for i,line in enumerate(m):
            if i==2:
                words = line.split()
                all_words += words

dictionary = Counter(all_words)

list_to_remove = list(dictionary.keys())

for item in list_to_remove:
    if item.isalpha() == False:
        del dictionary[item]
    elif len(item) == 1:
        del dictionary[item]

dictionary = dictionary.most_common(3000)

vocab = {}

for i, (word,frequency) in enumerate(dictionary):
    vocab[word] = i

# Created the dictionary and columns of the word_matrix


def making_word_dictionary(mail_dir):

    files = [os.path.join(mail_dir,f) for f in os.listdir(mail_dir)]

    docId = 0
    matrix = np.zeros((len(files), 3000))

    for file in files:
        with open(file) as m:
            for i,line in enumerate(m):
                if i == 2:
                    words = line.split()

                    wordId = 0
                    for word in words:
                        if word in vocab.keys():
                            wordId = vocab[word]
                            matrix[docId, wordId] = words.count(word)

            docId += 1

    return matrix



train_word_matrix = making_word_dictionary(train_dir)

train_labels = np.zeros(702)
train_labels[351:701] = 1

model = LogisticRegression()
model.fit(train_word_matrix, train_labels)











app=Flask(__name__)
@app.route('/')
def first():
    return render_template('form.html')
@app.route('/',methods=['POST'])
def review():
    if request.method=='POST':
        data=request.form
        subject = data['name']
        mail = data['mail']
        
        test_word_matrix = np.zeros((1, 3000))

        words = mail.split()
        wordsId = 0
        for word in words:
            if word in vocab.keys():
                wordId = vocab[word]
                test_word_matrix[0, wordId] = words.count(word)

        result = model.predict(test_word_matrix)
        # print(result)

        if result == 0:
            return render_template('nospam.html')
        else:
            return render_template('spam.html')





if __name__=='__main__':
    app.run(debug=True)
