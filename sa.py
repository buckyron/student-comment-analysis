
from textblob import TextBlob as blob
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt
import nltk
from nltk.stem.porter import PorterStemmer

doc_a = "The lectures were not conducted well and the class-discussion were highly interactive with inputs and insights being shared. "
doc_b = "The lecture notes are precise and the content is really interesting. "
doc_c = "The seminars in particular were very interesting and useful. "
doc_d = "The syllabus was challenging. The lecture take too much of time to finish the topic. "
doc_e = "Seminar is useful to communicate to each other." 

# compile sample documents into a single string
doc_bag = doc_a + doc_b + doc_c + doc_d + doc_e

sentences = nltk.sent_tokenize(doc_bag) #tokenize sentences
nouns = [] #empty to array to hold all nouns

#select nouns froms the sentences
for sentence in sentences:
     for word,pos in nltk.pos_tag(nltk.word_tokenize(str(sentence))):
         if (pos == 'NN' or pos == 'NNP' or pos == 'NNS' or pos == 'NNPS'):
             nouns.append(word)
#stemming the nouns
p_stemmer = PorterStemmer()
nouns = [p_stemmer.stem(i) for i in nouns]
nouns = set(nouns)
nouns = list(nouns)
print(nouns)

#creating textblobs
stu1 = blob(doc_a)
stu2 = blob(doc_b)
stu3 = blob(doc_c)
stu4 = blob(doc_d)
stu5 = blob(doc_e)

#creating a list of textblobs
blob_set = [stu1, stu2, stu3, stu4, stu5]

#creating the opinion list
opinion = [0] * len(nouns)
for i in range(0,len(nouns)):
    opinion[i]=0
    for j in blob_set:
        if nouns[i] in j:
            opinion[i] += j.sentiment[0]
print(opinion)


#creating objects to plot on the graph
objects = nouns #x-axis
y_pos = np.arange(len(objects))
performance = opinion   #y-axis

#plotting the graph
plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('opinion')
plt.title('Feedback on')

#displaying the graph
plt.show()

