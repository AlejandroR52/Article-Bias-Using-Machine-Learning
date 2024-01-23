#import sklearn packeges

from sklearn import tree
from sklearn.feature_extraction.text import CountVectorizer

import matplotlib.pyplot as plt


import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')


import os
import newsArticles
#declares all the varibles
lib = []
semi_lib = []
centrial = []
semi_concer = []
concer = []
test_texts = []  #takes input


#turns the files into list
def reading_files(media, text_list):

  f = open("newsArticles/" + media + ".txt" , "r")

  for i in range(0,500):
    text_list.append(f.readline())

#use to test with chinies test in fiels



def part_of_speach(text_list, song):
    tokens = nltk.word_tokenize(song)
    pos = nltk.pos_tag(tokens)
    songAdjAndNoun = []

    for i in range(0, len(pos) - 1):
        if pos[i][1] == "JJ" or pos[i][1] == "JJR" or pos[i][1] == "JJS":
            songAdjAndNoun.append(pos[i][0])

        if pos[i][1] == "NN" or pos[i][1] == "NNS" or pos[i][1] == "NNPS":
            songAdjAndNoun.append(pos[i][0])
        if pos[i][1] == "RB" or pos[i][1] == "RBB" or pos[i][1] == "RBS":
            songAdjAndNoun.append(pos[i][0])
        if pos[i][1] == "VB" or pos[i][1] == "VBD" or pos[i][1] == "VBG"  or pos[i][1] == "VBN" or pos[i][1] == "VBP"  or pos[i][1] == "VBZ":
            songAdjAndNoun.append(pos[i][0])


    return " ".join(songAdjAndNoun)



#trains the model
#training liberial biase
reading_files("CNN", lib)
reading_files("Vox", lib) 


reading_files("Atlantic", semi_lib)
reading_files("NPR", semi_lib)

reading_files("Reuters", centrial)
reading_files("BBC", centrial)

reading_files("New_York_Post", semi_concer)

reading_files("Fox_News", concer)
reading_files("Breitbart", concer)

#reading_files("New_York_Times", test_texts)

f = open("newsArticles/Atlantic.txt", "r")
for i in range(0, 1000):
  test_texts.append(f.readline())

#concerv  itive
predicList = []

training_texts = lib + semi_lib + centrial + semi_concer + concer
training_labels = ["liberal bias"] * len(lib) + ["leaning liberal bias"] * len(
      semi_lib) + ["centrial"] * len(centrial) + ["leaning concervative bias"] * len(semi_concer) + ["conservative bias"] * len(concer)

vectorizer = CountVectorizer()
vectorizer.fit(training_texts)

training_vectors = vectorizer.transform(training_texts)
testing_vectors = vectorizer.transform(test_texts)

classifier = tree.DecisionTreeClassifier()
classifier.fit(training_vectors, training_labels)
classifier.predict(testing_vectors)

guess = classifier.predict(testing_vectors)
#predicList = guess
print(guess)

liberal_bias = 0
centrial = 0
leaning_liberal_bias = 0
leaning_concervative_bias = 0
concervative = 0



for i in guess:
  if(i == 'liberal bias'): 
    liberal_bias += 1
  if(i == 'centrial'): 
    centrial += 1
  if(i == 'leaning liberal bias'): 
    leaning_liberal_bias += 1
  if(i == 'conservative bias'): 
    concervative += 1  
  if(i == 'leaning conservative bias'): 
    leaning_concervative_bias += 1
    
#print(predicList)

print("liberal_bias: " , liberal_bias)
print("central: ", centrial)
print("leaning_liberal_bias: " , leaning_liberal_bias)
print("leaning_conservative_bias", leaning_concervative_bias)
print("conservative: ", concervative)


fig = plt.figure(figsize=(5, 5))
tree.plot_tree(classifier,
               feature_names=vectorizer.get_feature_names(),
               rounded=True,
               filled=True)
fig.savefig('tree.png')







