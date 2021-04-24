from visualization import *
porter = PorterStemmer()
stopWords = set(stopwords.words('english'))

# Normalize and clean sentences
def cleanSentences(string, reg = RegexpTokenizer(r'[a-z]+')):

    string = string.lower().replace('subject','')
    tokens = reg.tokenize(string)
    cleanedSentence = [porter.stem(w) for w in tokens if not w in stopWords and w.isalpha()] # Ask from teacher can I remove digits from message!!!
    return cleanedSentence

def preprocessData(dataset):
    dataset = dataset[1:] #  removing header
    #clean messages and insert to new column
    dataset['clean_text'] = dataset['text'].apply(cleanSentences)
    dataset['Label'].value_counts(normalize=True)
    
    # Randomize the dataset and calculate index for split 20%/80%
    dataRandomized = dataset.sample(frac=1, random_state=1)
    trainingSize = round(len(dataRandomized) * 0.8)
    
    trainingSet,testSet = splitData(dataset,trainingSize)
    
    vocabulary = createVocabulary(trainingSet)
    
    trainingSetClean = oneHotEncoding(trainingSet, vocabulary)
    
    return trainingSetClean, testSet, vocabulary

def splitData(dataset,trainingSize):
    # Split into training and test sets
    trainingSet = dataset[:trainingSize].reset_index(drop=True)
    testSet = dataset[trainingSize:].reset_index(drop=True)

    describeTrainAndTestSet(trainingSet,testSet)
    return trainingSet,testSet

def createVocabulary(trainingSet):
    # Go through sentences and collect words to list 
    vocabulary = []
    for text in trainingSet['clean_text']:
      for word in text:
          vocabulary.append(word)
    # Convert list to set for unique words
    vocabulary = list(set(vocabulary))
    print("Total unique words in dataset: ",len(vocabulary))
    
    return vocabulary

def oneHotEncoding(trainingSet,vocabulary):
    wordCountsPerText = {uniqueWord: [0] * len(trainingSet['clean_text']) for uniqueWord in vocabulary}

    for index, text in enumerate(trainingSet['clean_text']):
       for word in text:
          wordCountsPerText[word][index] += 1

    wordCounts = pd.DataFrame(wordCountsPerText)
    trainingSetClean = pd.concat([trainingSet, wordCounts], axis=1)
    
    return trainingSetClean
    
    
def prepareForWordCloud(spamTexts,hamTexts):
    #Take first 50 words and convert them to list for visualize
    spamLists = spamTexts[:50]["clean_text"].tolist()
    spamVocabulary = []
    for l in spamLists:
        spamVocabulary += l

    hamLists = hamTexts[:50]["clean_text"].tolist()
    hamVocabulary = []
    for l in hamLists:
        hamVocabulary += l

    return spamVocabulary, hamVocabulary