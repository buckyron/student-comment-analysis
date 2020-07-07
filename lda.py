from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
import gensim

tokenizer = RegexpTokenizer(r'\w+')

# create English stop words list
en_stop = get_stop_words('en')

# Create p_stemmer of class PorterStemmer
p_stemmer = PorterStemmer()
    
# create sample documents
doc_a = "The lecturer were conducted well and the class discussion were highly interactive with inputs and insights being shared."
doc_b = "Tje lecture notes are precise and the content is really interesting."
doc_c = "The seminars in particular were very interesting and useful."
doc_d = "The syllabus was challenging. The lecture take too much of time to finish the topic."
doc_e = "Seminar is useful to communicate to each other." 

# compile sample documents into a list
doc_set = [doc_a, doc_b, doc_c, doc_d, doc_e]

# list for tokenized documents in loop
texts = []

# loop through document list
for i in doc_set:
    
    # clean and tokenize document string
    raw = i.lower()
    tokens = tokenizer.tokenize(raw)

    # remove stop words from tokens
    stopped_tokens = [i for i in tokens if not i in en_stop]
    
    # stem tokens
    stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
    
    # add tokens to list
    texts.append(stemmed_tokens)

# turn our tokenized documents into a id <-> term dictionary
dictionary = corpora.Dictionary(texts)
    
# convert tokenized documents into a document-term matrix
corpus = [dictionary.doc2bow(text) for text in texts]

# generate LDA model
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=2, id2word = dictionary, passes=20)

print(ldamodel.print_topics(num_topics=3, num_words=3))







stu1 = blob('The lecturer were conducted well and the class discussion were highly interactive with inputs and insights being shared.')
stu2 = blob('The lecture notes are precise and the content is really interesting.')
stu3 = blob('The seminars in particular were very interesting and useful.')
stu4 = blob('The syllabus was challenging. The lecture take too much of time to finish the topic.')
stu5 = blob("Seminar is useful to communicate to each other")
stu5 = blob("Lecture is very bad I hate it")

