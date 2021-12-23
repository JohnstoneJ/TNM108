import pandas as pd
import nltk
import numphy as np
import re
from nltk.stem import wordnet #to perform lemmitization
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import  pos_tag #for parts of speech
from sklearn.metrics import pairwise_distances # to perform cosine similarity
from nltk import word_tokenize #to create tokens
from nltk.corpus import stopwords #for stop words
import matplotlib.pyplot as plt 

from algorithms import *
from textProcessing import *      

dataFile = pd.read_excel("Q_and_A.xlsx") 

dataFile.ffill(axis=0, inplace = True)

### Applying to dataset
dataFile['lemmatized_text'] = dataFile['Questions'].apply(text_normalizer)

menu = {}
menu['1']="Test the algorithms" 
menu['2']="Exit"

algo = ['TFIDF','BOW']

counter_A = 0
counter_B = 0
counter_Equal = 0
numberOfQuestions = 0

while True: 
    print('\n--------------------- ')
    options=menu.keys()
    sorted(options)
    for entry in options: 
        print (entry, menu[entry])
    selection = input("Please select: ") 
    print('--------------------- ')
  

    if selection =='1': 
      print ('   TEST ITERATION')

      while numberOfQuestions < 10:
        print('--------------------- ')
        Question = input('\nAsk a question: ')

        numberOfQuestions += 1
        print(numberOfQuestions)

        print('(A)  TFIDF: ' + dataFile['Questions'].loc[tfidf(Question,dataFile['lemmatized_text'])])

        print('(B)  BOW: ' + dataFile['Questions'].loc[bagOfWords(Question,dataFile['lemmatized_text'])])

        best_algo = input('\nWhich algorithm gave the best answer? A / B / E (Equal) \nAnswer: ' )
        print('You choose ' + best_algo)
        
      
        if best_algo=='A' or best_algo =='a':
          counter_A += 1
          print(counter_A)
        elif best_algo == 'B' or best_algo == 'b': 
          counter_B += 1
          print(counter_B)
        elif best_algo == 'E' or best_algo == 'e' :
          counter_Equal += 1
          print(counter_Equal)
        else: 
          break


      print('\n--------------------- ')
      print ('      RESULT')
      print('--------------------- ')
      print("\nTFIDF was best " + str(counter_A) + " times")
      print("BOW was best " + str(counter_B) + " times")
      print("They were equal in performance " + str(counter_Equal) + " times")

 
      # defining labels
      activities = ['TFIDF', 'BOW', 'EQUAL',]
      
      # portion covered by each label
      slices = [counter_A, counter_B, counter_Equal]
      
      # color for each label
      colors = ['#ff9999', '#99ff99', '#99e6ff']
      
      # plotting the pie chart
      plt.pie(slices, labels = activities, colors=colors,
              startangle=90, shadow = False, explode = (0, 0, 0),
              radius = 1.2, autopct = '%1.1f%%', labeldistance=1.2)

      plt.title('RESULT', fontdict={'fontsize': 17})        
      
      # plotting legend
      plt.legend()
      
      # showing the plot
      plt.show()
      
      break 

    elif selection == '2':
     break
    elif selection == '': 
      break
    else: 
      print ('Unknown Option Selected')



