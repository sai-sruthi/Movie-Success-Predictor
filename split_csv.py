
import nltk.classify.util
from nltk.classify import NaiveBayesClassifier # we use NaiveBayes to classify the reviews as positive or negative
from nltk.stem import WordNetLemmatizer #
from nltk.tokenize import word_tokenize # 
from collections import Counter
import pandas as pd

imdb = pd.read_csv('IMDB Dataset.csv')
positive =  imdb['sentiment']=='positive'
pos_review = imdb[positive]
positive_reviews = pos_review[1:500]
negative = imdb['sentiment']=='negative'
neg_review = imdb[negative]
negative_reviews = neg_review[1:500]

stopwords = set(w.rstrip() for w in open('stopwords.txt'))

wordnet_lemmatizer = WordNetLemmatizer()

word_index_map = {}
current_index = 0

orig_reviews = []
def my_tokenizer(s):
    s = s.lower() # Convert into lowercase
    tokens = nltk.tokenize.word_tokenize(s) # split string into words (tokens)
    tokens = [t for t in tokens if len(t) > 2] # remove short words, they're probably not useful
    tokens = [wordnet_lemmatizer.lemmatize(t) for t in tokens] # put words into base form
    tokens = [t for t in tokens if t not in stopwords] # remove stopwords
    my_dict = dict([(t, True) for t in tokens])
    return my_dict # return dictionary
neg_reviews = []
for reviews in negative_reviews['review']:
    neg_reviews.append((my_tokenizer(reviews), "negative"))
pos_reviews = []
for reviews in positive_reviews['review']:
    pos_reviews.append((my_tokenizer(reviews), "positive"))
train_set = neg_reviews + pos_reviews
classifier = NaiveBayesClassifier.train(train_set)


moviereviews = pd.read_csv('rotten_tomatoes_reviews.csv',na_filter= False)
moviereviews = moviereviews[1:500]
grouped = moviereviews.groupby('rotten_tomatoes_link') 
final = []
for name, group in grouped:
        #print(name)
        #print(group)
        tag=[]
        for review in group['review_content']:
            tokened=my_tokenizer(review)
            tag.append(classifier.classify(tokened))
        counter=Counter(tag)
        final.append((name,counter['positive']-counter['negative']))

df = pd.DataFrame(final,columns =['movielink', 'val']) 
print(df)
