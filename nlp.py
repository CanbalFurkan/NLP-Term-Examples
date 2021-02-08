import pandas as pd
import matplotlib.pyplot as plt
import wordcloud
from nltk.stem import PorterStemmer
import nltk
from nltk.corpus import stopwords as _stopwords
from wordcloud import WordCloud, STOPWORDS
import re
from scipy import special
import numpy as np
from operator import itemgetter
from nltk.lm import MLE
from nltk.lm import KneserNeyInterpolated
from nltk import tokenize
from nltk.lm.preprocessing import pad_both_ends, padded_everygram_pipeline
from nltk import word_tokenize
from nltk.tokenize import sent_tokenize
import gensim
from statistics import mean
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

nltk.download('punkt')
nltk.download('stopwords')


def create_WordCloud(list_doc,dimensions,output_path,mode,stopwords):
	if mode=="TF":
		stops = set(_stopwords.words('turkish'))
		unique_string=(" ").join(list_doc)
		unique_string=unique_string.lower()
		word_tokens = word_tokenize(unique_string)
		filtered_sentence = [] 
		if stopwords==True:	
			for w in word_tokens: 
				if w not in stops: 
					filtered_sentence.append(w) 
			str1= " "
			filtered_sentence=str1.join(filtered_sentence)
		else:
			filtered_sentence=unique_string	
		wordcloud = WordCloud(random_state=1, background_color='salmon', colormap='Pastel1', collocations=False, stopwords = stops).generate(filtered_sentence)
		plt.figure() 
		plt.imshow(wordcloud) 
		plt.axis("off")
		plt.savefig(output_path, format="png") 
		
	elif mode =="TFIDF":
		tfidf_vectorizer=TfidfVectorizer(use_idf=True) 
		tfidf_vectorizer_vectors=tfidf_vectorizer.fit_transform(list_doc)
		first_vector_tfidfvectorizer=tfidf_vectorizer_vectors[0] 
		df = pd.DataFrame(first_vector_tfidfvectorizer.T.todense(), index=tfidf_vectorizer.get_feature_names(), columns=["tfidf"])
		df.sort_values(by=["tfidf"],ascending=False)
		stops = set(_stopwords.words('turkish'))
		if stopwords==True:
			for index,row in df.iterrows():
				if row[0] in stops:
					df.drop(index,inplace=True)

		Cloud = WordCloud(background_color="white", max_words=50).generate_from_frequencies(df.sum(axis=1))
		plt.figure()
		plt.imshow(Cloud, interpolation="bilinear")
		plt.axis("off")
		plt.savefig(output_path, format="png") 
 		

def create_ZiphsPlot(list_doc,output_path):
	unique_string=(" ").join(list_doc)
	unique_string=unique_string.lower()
	words = re.findall(r'(\b[A-Za-z][a-z]{2,9}\b)', unique_string)
	frequency = {}
	for word in words:
		count=frequency.get(word,0) 
		frequency[word] = count + 1

	frequency={k: v for k, v in sorted(frequency.items(), key=lambda item: item[1],reverse=True)}
	vals=list(frequency.values())
	from math import log
	numbers=list(range(1, len(frequency)+1))
	numbers=[log(y,10) for y in numbers]
	vals=[log(z,10) for z in vals]
	plt.plot(numbers,vals)
	plt.savefig(output_path)
def create_HeapsPlot(list_doc,output_path):
	list_size=len(list_doc)
	total_numb=[]
	unq_numb=[]
	current_corpus=''
	for i in range (list_size):
		print(i)
		unique_string=("").join(list_doc[i])
		unique_string=unique_string.lower()
		current_corpus+=unique_string
		nltk_tokens = nltk.word_tokenize(current_corpus)
		total_numb.append(len(current_corpus.split()))
		unq_numb.append(len(nltk_tokens))

	plt.plot(total_numb,unq_numb)
	plt.savefig(output_path)
	





##todo
###The function will create the Heaps’s plot and save it in the provided output file in png
##format
	return 1
def create_LanguageModel(list_doc,model_type,ngram):
	ps = PorterStemmer()
	unique_string=("").join(list_doc)
	tokenized_text = [list(map(str.lower, word_tokenize(sent))) 
                  for sent in sent_tokenize(unique_string)]

	train_data, padded_sents = padded_everygram_pipeline(ngram, tokenized_text)
	if model_type=="MLE":
		model = MLE(ngram)
		model.fit(train_data,padded_sents)
		return model
	elif model_type=="KneserNeyInterpolated":
		model = KneserNeyInterpolated(ngram)
		model.fit(train_data,padded_sents)
		return model
	

def generate_sentence(trained_model,text):
	result="a"
	base=[]
	base.append(text)
	new=text
	while result != "</s>":
		result=trained_model.generate(text_seed=base)
		base.append(result)
		new+=" "+result

	print(base)
	print(new)
	base.pop()
	new_txt=""
	for tt in base:
		new_txt+=' '+tt
	score=trained_model.perplexity(new_txt)
	print(score)
	print(new_txt)
	print(base)
	return new,score

def create_WordVectors(list_doc,dimensions,type_model,window_size):
	unique_string=("").join(list_doc)
	print("girdim")
	tokenized_text = [list(map(str.lower, word_tokenize(sent))) 
                  for sent in sent_tokenize(unique_string)]
	if type_model == "cbow":
		model = gensim.models.Word2Vec(tokenized_text, min_count = 1,  
                              size = dimensions, window = window_size) 
	else:
		model = gensim.models.Word2Vec(tokenized_text, min_count = 1, size = dimensions, 
                                             window = window_size, sg = 1) 
		
	print("hey")
	return model	

def use_WordRelationship(model,tuple_list,tuple_missing):
	print("deneme")
	score=[]
	for x in tuple_list:
		print(x)
		try:
			score.append(model.similarity(x[0],x[1]))
		except :
			print("pas")
	avg=mean(score)
	
	my_vec=model.wv[tuple_missing[0]]
	my_vec2=my_vec+avg
	print(my_vec)
	print(my_vec2)
	print(model.similar_by_vector(my_vec2,topn=5))


	return model.similar_by_vector(my_vec2,topn=5)




    


