stop_words = nltk.corpus.stopwords.words('english')
stop_words.extend(['i', 'I','also', 'robinhood', 'app' , 'nt' ,'acorn', 'stash', 'e*trade', 'fidelity', 'td', 'ameritrade',
                      'robin hood', 'e trade', 'ameri trade', 'emoods', 'emood', 'e-moods', 'happify',
                   'stockpile', 'pile', 'front', 'wealthfront', 'wealth-front', 'wealth',
                   'go-puff', 'puff', 'deliverycom', 'delivery.com'
                 'schwab', 'personal capital', 'capital', 'ubereats', 'doordash', 'grubhub', 'postmates', 'seamless',
                  'uber', 'door dash', 'uber eats', 'grub hub', 'post mates', 'seam less',
                  'calm','headspace','sanvello','talkspace','shine', 
                  'would', 'could', 'can', 'ca', 'lot', 'much', 'every', 'something', 'everything', 'u', 'thing'])
lemma = nltk.stem.WordNetLemmatizer()
glove_model = KeyedVectors.load_word2vec_format(glove_path, binary=False)
df = pd.read_csv("Food Delivery.csv", encoding = "ISO-8859-1") # Change the dataset here

# Pre-processing 
def remove_ascii_words(df):
    non_ascii_words = []
    for i in range(len(df)):
        
        if not pd.isnull(df.loc[i, 'content']): 
            for word in df.loc[i, 'content'].split(' '):
                if len(word)< 3 and len(word) > 15:
                    if any([ord(character) >= 128 for character in word]):
                        non_ascii_words.append(word)
                        df.loc[i, 'content'] = df.loc[i, 'content'].replace(word, '')
    return non_ascii_words

def get_good_tokens(sentence):
    replaced_punctation = list(map(lambda token: re.sub('[^A-Za-z]+', '', token), sentence))
    removed_punctation = list(filter(lambda token: token, replaced_punctation))
    return removed_punctation

def df_get_good_tokens(df):
    df['content'] = df.content.str.lower()
    df['tokenized_content'] = list(map(nltk.word_tokenize, df.content))
    df['tokenized_content'] = list(map(get_good_tokens, df.tokenized_content))

def remove_stopwords(df):
    df['stopwords_removed'] = list(map(lambda doc:
                                       [word for word in doc if word not in stop_words],
                                       df['tokenized_content']))
def lemma_words(df):
    df['lemmatized_text'] = list(map(lambda sentence:list(map(lemma.lemmatize, sentence)),df.stopwords_removed))
    df['stemmed_text'] = list(map(lambda sentence:
                                  list(map(p_stemmer.stem, sentence)),
                                  df.lemmatized_text))
    return df.lemmatized_text

def document_to_bow(df):
    dictionary = Dictionary(documents=df.lemmatized_text.values)
    dictionary.filter_extremes(no_above=0.8, no_below=3)
    dictionary.compactify()
    df['bow'] = list(map(lambda doc: dictionary.doc2bow(doc), df.lemmatized_text))
    
def preprocessing(df):
    remove_ascii_words(df)
    df_get_good_tokens(df)
    remove_stopwords(df)
    texts = lemma_words(df)
    document_to_bow(df)

	
# Calculate Hybrid TF.IDF
def wordListToFreqDict(wordlist):
    wordfreq = [wordlist.count(p) for p in wordlist]
    return dict(list(zip(wordlist,wordfreq)))
def sortFreqDict(freqdict):
    aux = [(freqdict[key], key) for key in freqdict]
    aux.sort()
    aux.reverse()
    return aux

word_set = []
for text in texts:
    for word in text:
        if word not in word_set:
            word_set.append(word)

sentences = texts
word_set = set(word_set)
total_documents = len(sentences) #Total documents in our corpus

index_dict = {} #Creating an index for each word in our vocab.
i = 0
for word in word_set:
    index_dict[word] = i
    i += 1

def count_dict(sentences): #Create a count dictionary
    word_count = {}
    for word in word_set:
        word_count[word] = 0
        for sent in sentences:
            if word in sent:
                word_count[word] += 1
    return word_count
 
def computeTF(wordDict, bagOfWords):
    tfDict = {}
    bagOfWordsCount = len(bagOfWords)
    for word, count in wordDict.items():
        tfDict[word] = count / float(bagOfWordsCount)
    return tfDict
def computeIDF(wordDict, reviews):
    import math
    N = len(reviews)
    
    idfDict = dict.fromkeys(wordDict.keys(), 0)
    for review in reviews:
        for word in review:
            idfDict[word] += 1
    
    for word, val in idfDict.items():
        idfDict[word] = math.log(N / float(val))
    return idfDict

def computeTfIdf(tfDict, idfDict):
    tf_idfDict = {}
    for word, val in tfDict.items():
        tf_idfDict[word] = val*idfDict[word]
    return tf_idfDict

word_count = count_dict(sentences)
tfDict = computeTF(word_count, all_words)
idfDict = computeIDF(word_count, data_lemmatized)
tf_idfDict = computeTfIdf(tfDict, idfDict)

def fillDfTfIdf(): #Average Hybrid TF.IDF for each review
    df['tfidf'] = ''
    for i in range(len(df)):
        words = df.loc[i, 'processed']
        length = len(words)
        tfidf_sum = 0
        for word in words:
            tfidf_sum += tf_idfDict[word]
        avg = tfidf_sum/length if length > 0 else 0
        df.loc[i, 'tfidf'] = avg
		
def fillDfGloVe(model): #Average Glove vectors for each review
    df['glove'] = '' 
    index2word_set = set(model.wv.vocab.keys())
    
    for i in range(len(df)):
        featureVec = np.zeros(model.vector_size, dtype="float32")
        words = df.loc[i, 'processed']
        nwords = 0
        for word in words:
            if word in index2word_set and len(word) > 2: 
                featureVec = np.add(featureVec, model[word])
                nwords += 1.
        if nwords > 0:
            featureVec = np.divide(featureVec, nwords)
        df.at[i, 'glove'] = featureVec
		
def AdjustDFTfIdf(df): # increase the weight of privacy-related keywords and recalculate the weights
    sorted_tfidf_dict = sortFreqDict(tf_idfDict)
	max_tf_idf = sorted_tfidf_dict[0][0]
	df['tfidf_adjusted'] = ''
    for i in range(len(df)):
        words = df.loc[i, 'processed']
        length = len(words)
        tfidf_sum = 0
          
        for word in words:
            if word in privacy_keywords:
                tfidf_sum += max_tf_idf
            else:
                tfidf_sum += tf_idfDict[word]
        avg = tfidf_sum/length if length > 0 else 0
        df.loc[i, 'tfidf_adjusted'] = avg

def selectTopScoreReview_tfIdf(df, hybrid = True): #select top score review according to their Hybrid TF.IDF
    if hybrid:
		df = df.sort_values(by=['tfidf'],ascending=False)
		return list(df.index)
	else:
		df = df.sort_values(by=['tfidf_adjusted'],ascending=False)
		return list(df.index)

def computeTfIdfCosineSimilarity(review1, review2): #Calculate cosine Similarity of reviews based on their Hybrid TF.IDF
    v1 = []
    v2 = []
    U = set(review1).union(set(review2))
    for word in U:
        v1.append(tf_idfDict[word]) if word in review1 else v1.append(0)
        v2.append(tf_idfDict[word]) if word in review2 else v2.append(0)
    a = np.array(v1)
    b = np.array(v2)
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b)
	
def computeEmbeddingSimilarity(vector1, vector2): #Calculate the Glove Similarity of reviews
    dot_product = np.dot(vector1, vector2)
    norm_a = np.linalg.norm(vector1)
    norm_b = np.linalg.norm(vector2)
    return dot_product / (norm_a * norm_b)
	
def selectSummaries_hybridtfidf(k, threshold, indexes, min_len = 5):
    selected = []
    for ind in indexes:
        if ind not in selected and len(selected) != k and min_len < len(df.loc[ind, 'processed']):
            sim = []
            for item in selected:
                sim.append(computeTfIdfCosineSimilarity(df.loc[item, 'processed'], df.loc[ind, 'processed']))
            if len(selected) == 0:
                selected.append(ind)
                print(df.loc[ind, 'content']) 
                print(df.loc[ind, 'tfidf'])
            elif max(sim) < threshold:
                selected.append(ind)
                print(df.loc[ind, 'content']) 
                print(df.loc[ind, 'tfidf'])
	return selected   

def selectSummaries_glove(k, threshold, indexes, min_len = 5):
    selected = []
    for ind in indexes:
        if ind not in selected and len(selected) != k and min_len < len(df.loc[ind, 'processed']) :
            sim = []
            for item in selected:
                sim.append(computeEmbeddingSimilarity(df.loc[item, 'glove'], df.loc[ind, 'glove']))
            if len(selected) == 0:
                selected.append(ind)
                print(df.loc[ind, 'content']) 
                print(df.loc[ind, 'tfidf_adjusted'])
            elif max(sim) < threshold:
                selected.append(ind)
                print(df.loc[ind, 'content']) 
                print(df.loc[ind, 'tfidf_adjusted'])

    return selected 

Hybrid_tf_idf_summaries = selectSummaries_hybridtfidf(5, 0.2, selectTopScoreReview_tfIdf(df, True))
selectSummaries_glove = selectSummaries_hybridtfidf(5, 0.4, selectTopScoreReview_tfIdf(df, False))


#LDA
corpus = df.bow
dictionary = Dictionary(documents=df.lemmatized_text.values)
dictionary.filter_extremes(no_above=0.8, no_below=3)
dictionary.compactify() 
data_lemmatized = df.lemmatized_text.values
texts = data_lemmatized

lda_model = LdaMulticore(corpus=corpus,
                        id2word=dictionary,
                        num_topics=5,
                        workers=10,
                        passes=100,
                        chunksize=50,
                        per_word_topics=True,
                        iterations= 10000,
                        alpha='asymmetric')
for index, topic in lda_model.show_topics(formatted=False, num_words= 10):
    print('Topic: {} \nWords: {}'.format(index, [w[0] for w in topic]))