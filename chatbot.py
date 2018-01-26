import numpy as np
import tensorflow as tf
import re
import time

##Data Preprocessing

lines =  open('movie_lines.txt', encoding ='utf-8', errors = 'ignore').read().split('\n')
conversations =  open('movie_conversations.txt', encoding ='utf-8', errors = 'ignore').read().split('\n')

#Directionary for line and its id
# id2line = {
#              id(int) : line (string)
#           }
id2line = {}

# line => L1044 +++$+++ u2 +++$+++ m0 +++$+++ CAMERON +++$+++ They do to!
for line in lines:
    _line = line.split(' +++$+++ ')
    if len(_line) == 5:
        id2line[_line[0]] = _line[4]
    
# conversations_ids = [
#                       ['L194', 'L195', 'L196', 'L197'],
#                       ['L196', 'L197']
#                     ]
conversations_ids = []

#All conversations
#conversations = > u0 +++$+++ u2 +++$+++ m0 +++$+++ ['L204', 'L205', 'L206']
for conversation in conversations[:-1]:
    _conversation = conversation.split(' +++$+++ ')[-1][1:-1].replace("'", "").replace(" ", "")
    conversations_ids.append(_conversation.split(','))
    
#List of strings
questions = []
answers = []

#Saving conversation into questions and answers
for conversation in conversations_ids:
    for i in range(len(conversation) - 1):
        questions.append(id2line[conversation[i]])
        answers.append(id2line[conversation[i + 1]])

#Cleaning Texts Function
def clean_text(text):
    text = text.lower()
    
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"that's", "that is", text)
    
    text = re.sub(r"what's", "what is", text)
    text = re.sub(r"where's", "where is", text)
    
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'d", " would", text)
    
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"[-()\"#/@;:<>{}+=~|.?,]", "", text)
                     
    print(text)
    return text

#Clean the questions
clean_questions =  []

for question in questions:
    clean_questions.append(clean_text(question))
    
#Clean the answers
clean_answers =  []

for question in questions:
    clean_answers.append(clean_text(question))


#Words and it's occurences
#word2count = {
#                 word (int) : number of words (int)
#           }
word2count = {}

for question in clean_questions:
    
    for word in question.split():
        if word not in word2count:
            word2count[word] = 1
        else:
            word2count[word]+= 1

for answer in clean_answers:
    
    for word in answer.split():
        if word not in word2count:
            word2count[word] = 1
        else:
            word2count[word]+= 1


#question to unique ID
threshold = 20
questionswords2int = {}
word_number = 0 

for word, count in word2count.items():
    if count >= threshold:
        questionswords2int[word] = word_number
        word_number += 1
        
#answer to unique ID   
answerwords2int = {}
word_number = 0 

for word, count in word2count.items():
    if count >= threshold:
        answerwords2int[word] = word_number
        word_number += 1


#Add encoder and decoder
# PAD => pad
# EOS => end of string
# OUT => ALl filtered out due to threshold
# SOS =>First word to start decording
tokens = ['<PAD>', '<EOS>', '<OUT>', '<SOS>']

for token in tokens:
    questionswords2int[token] = len(questionswords2int) + 1

for token in tokens:
    answerwords2int[token] = len(answerwords2int) + 1

#Inverse answerwords2int dictionary 
answersint2word = {w_i: w for w, w_i in answerwords2int.items()}


for i in range(len(clean_answers)):
    clean_answers[i] += ' <EOS>'
    
#Translate all the questions and answers into integers
questions_to_int = []

for question in clean_questions:
    ints = []
    
    for word in question.split():
        
        if word not in questionswords2int:
            ints.append(questionswords2int['<OUT>']) 
        else:
            ints.append(questionswords2int[word]) 
            
    questions_to_int.append(ints)  
            
answers_to_int = []

for answer in clean_answers:
    ints = []
    
    for word in answer.split():
        
        if word not in answerwords2int:
            ints.append(answerwords2int['<OUT>']) 
        else:
            ints.append(answerwords2int[word]) 
            
    answers_to_int.append(ints)  

#Sorting questions and answers by the length of questions
sorted_clean_questions = []
sorted_clean_answers = []

for length in range(1, 25 + 1):
    for i in enumerate(questions_to_int):
        if len(i[1]) == length:
            sorted_clean_questions.append(questions_to_int[i[0]])
            sorted_clean_answers.append(answers_to_int[i[0]])




####Seq2Seq2 Model######
            
#inputs and targets
def model_inputs():
    inputs = tf.placeholder(tf.int32, [None, None], name = 'input')
    targets = tf.placeholder(tf.int32, [None, None], name = 'target')
    lr = tf.placeholder(tf.float32, name = 'learning_rate')
    keep_prob = tf.placeholder(tf.float32, name = 'keep_prob')
    
    return inputs, targets, lr, keep_prob


#preprocessing targets
def preprocess_targets(targets):