import io
import nltk
import re
import string
import pandas as pd
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.cluster import KMeans
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA

def change_file(i):
# word_tokenize accepts
# a string as an input, not a file.

	stop_words = set(stopwords.words('english'))
	file_name=("text_"+str(i)+".txt")
	file_name1=("text"+str(i)+".txt")
	file = open(file_name) 
	filtered_list = []
	  
	# Use this to read file content as a stream: 
	line = file.read()
	dict_table = str.maketrans('', '', string.digits)
	line = line.translate(dict_table)
	line = re.sub((r'[^\w\s]'),'',line)
	words=word_tokenize(line.lower())
	for r in words:
		if not r in stop_words:
				appendFile = open(file_name1,'a')
				appendFile.write(" "+r) 
				appendFile.close()
				filtered_list.append(r)
	return filtered_list

def computeTF(wordDict, bagOfWords):
    tfDict = {}
    bagOfWordsCount = len(bagOfWords)
    for word, count in wordDict.items():
        tfDict[word] = count / float(bagOfWordsCount)
    return tfDict

def computeIDF(documents):
    import math
    N = len(documents)
    
    idfDict = dict.fromkeys(documents[0].keys(), 0)
    for document in documents:
        for word, val in document.items():
            if val > 0:
                idfDict[word] += 1
    
    for word, val in idfDict.items():
        idfDict[word] = math.log(N / float(val))
    return idfDict

def get_unique_words():
    with open('text1.txt') as fp:
    	data1 = fp.read()
    with open('text2.txt') as fp:
    	data2 = fp.read()
    with open('text3.txt') as fp:
    	data3 = fp.read()
    with open('text4.txt') as fp:
    	data4 = fp.read()
    with open('text5.txt') as fp:
    	data5 = fp.read()
    with open('text6.txt') as fp:
    	data6 = fp.read()
    with open('text7.txt') as fp:
    	data7 = fp.read()
    with open('text8.txt') as fp:
    	data8= fp.read()
    with open('text9.txt') as fp:
    	data9 = fp.read()
    with open('text10.txt') as fp:
    	data10 = fp.read()
    data1=data1.split(" ")
    data2=data2.split(" ")
    data3=data3.split(" ")
    data4=data4.split(" ")
    data5=data5.split(" ")
    data6=data6.split(" ")
    data7=data7.split(" ")
    data8=data8.split(" ")
    data9=data9.split(" ")
    data10=data10.split(" ")
    total = set(data1).union(set(data2)).union(set(data3)).union(set(data4)).union(set(data5)).union(set(data6)).union(set(data7)).union(set(data8)).union(set(data9)).union(set(data10))
    return total

def computeTFIDF(tfBow, idfs):
    tfidf = {}
    for word, val in tfBow.items():
        tfidf[word] = val*idfs[word]
    return(tfidf)

def prepare_file():
    with open('text1.txt') as fp:
    	data1 = fp.read()
    with open('text2.txt') as fp:
    	data2 = fp.read()
    with open('text3.txt') as fp:
    	data3 = fp.read()
    with open('text4.txt') as fp:
    	data4 = fp.read()
    with open('text5.txt') as fp:
    	data5 = fp.read()
    with open('text6.txt') as fp:
    	data6 = fp.read()
    with open('text7.txt') as fp:
    	data7 = fp.read()
    with open('text8.txt') as fp:
    	data8= fp.read()
    with open('text9.txt') as fp:
    	data9 = fp.read()
    with open('text10.txt') as fp:
    	data10 = fp.read()
    data1=data1.split(" ")
    data2=data2.split(" ")
    data3=data3.split(" ")
    data4=data4.split(" ")
    data5=data5.split(" ")
    data6=data6.split(" ")
    data7=data7.split(" ")
    data8=data8.split(" ")
    data9=data9.split(" ")
    data10=data10.split(" ")
    total = set(data1).union(set(data2)).union(set(data3)).union(set(data4)).union(set(data5)).union(set(data6)).union(set(data7)).union(set(data8)).union(set(data9)).union(set(data10))
    wordDict1 = dict.fromkeys(total, 0) 
    wordDict2 = dict.fromkeys(total, 0)
    wordDict3 = dict.fromkeys(total, 0) 
    wordDict4 = dict.fromkeys(total, 0)
    wordDict5 = dict.fromkeys(total, 0) 
    wordDict6 = dict.fromkeys(total, 0)
    wordDict7 = dict.fromkeys(total, 0) 
    wordDict8 = dict.fromkeys(total, 0)
    wordDict9 = dict.fromkeys(total, 0) 
    wordDict10 = dict.fromkeys(total, 0)
    for word in data1:
    	wordDict1[word]+=1
    for word in data2:
    	wordDict2[word]+=1
    for word in data3:
    	wordDict3[word]+=1    
    for word in data4:
    	wordDict4[word]+=1
    for word in data5:
    	wordDict5[word]+=1    
    for word in data6:
    	wordDict6[word]+=1
    for word in data7:
    	wordDict7[word]+=1    
    for word in data8:
    	wordDict8[word]+=1
    for word in data9:
    	wordDict9[word]+=1
    for word in data10:
    	wordDict10[word]+=1
    tf1 = computeTF(wordDict1, data1)
    tf2 = computeTF(wordDict2, data2)
    tf3 = computeTF(wordDict3, data3)
    tf4 = computeTF(wordDict4, data4)
    tf5 = computeTF(wordDict5, data5)
    tf6 = computeTF(wordDict6, data6)
    tf7 = computeTF(wordDict7, data7)
    tf8 = computeTF(wordDict8, data8)
    tf9 = computeTF(wordDict9, data9)
    tf10 = computeTF(wordDict10, data10)
    tf = pd.DataFrame([tf1, tf2,tf3,tf4,tf5,tf6,tf7,tf8,tf9,tf10])
    File1 = open('tf_result.txt','a')
    File1.write(str(tf)) 
    File1.close()

    print(tf)
    idfs = computeIDF([wordDict1, wordDict2,wordDict3, wordDict4,wordDict5, wordDict6,wordDict7, wordDict8,wordDict9, wordDict10])
    File2 = open('idf_result.txt','a')
    File2.write(str(idfs)) 
    File2.close()
    print(idfs)
    idf1 = computeTFIDF(tf1, idfs)
    idf2 = computeTFIDF(tf2, idfs)
    idf3 = computeTFIDF(tf3, idfs)
    idf4 = computeTFIDF(tf4, idfs)
    idf5 = computeTFIDF(tf5, idfs)
    idf6 = computeTFIDF(tf6, idfs)
    idf7 = computeTFIDF(tf7, idfs)
    idf8 = computeTFIDF(tf8, idfs)
    idf9 = computeTFIDF(tf9, idfs)
    idf10 = computeTFIDF(tf10, idfs)
    #putting it in a dataframe
    idf= pd.DataFrame([idf1, idf2,idf3, idf4,idf5, idf6,idf7, idf8,idf9, idf10])
    print(idf)
    File3 = open('tf-idf_result.txt','a')
    File3.write(str(idf)) 
    File3.close()
    return idf

def create_random_centriods(clusters_num, len):
    centroids = []
    for i in range(0, clusters_num):
        centroid = []
        for coordinate in range(0, len):
            centroid.append(random.uniform(0, 1))
        centroids.append(centroid)

    return centroids


if __name__=='__main__':
	for i in range(1,11):
		file=change_file(i)
	_data=prepare_file()
	features = list(_data.columns)[:-10]

### Get the features data
	data = _data[features]
	clustering_kmeans = KMeans(n_clusters=3)
	data['clusters'] = clustering_kmeans.fit_predict(data)
	pca_num_components = 2

	reduced_data = PCA(n_components=pca_num_components).fit_transform(data)
	results = pd.DataFrame(reduced_data,columns=['pca1','pca2'])

	sns.scatterplot(x="pca1", y="pca2", hue=data['clusters'], data=results)
	plt.title('K-means Clustering with 2 dimensions')
	plt.show()