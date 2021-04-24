import pandas as pd
import numpy as np
from preprocess import *
# Upload stopwords
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
#from preprocessor import preprocessData, prepareForWordCloud

def loadDataSet():
    # Import dataset with pandas and label columns
    ds = pd.read_csv('/spam_ham_dataset.csv',
    header=None, names=['id','Label', 'text','label_num'])

    print(ds.shape)
    ds.head()
    return ds

def trainProbabilities(ds):
    trainingSetClean, testSet, vocabulary = preprocessData(ds)
    # Split spam and ham texts from training set
    spamTexts = trainingSetClean[trainingSetClean['Label'] == 'spam']
    hamTexts = trainingSetClean[trainingSetClean['Label'] == 'ham']

    #Create Word Cloud with spam and ham words
    displaySpamAndHamWords(spamTexts,hamTexts)

    # Calculate probabilty P(Spam) and P(Ham)
    pSpam = len(spamTexts) / len(trainingSetClean)
    pHam = len(hamTexts) / len(trainingSetClean)

    # Number of words in Spam sentences
    nWordsPerSpamText = spamTexts['clean_text'].apply(len)
    nSpam = nWordsPerSpamText.sum()

    # Number of words in Ham sentences
    nWordsPerHamText = hamTexts['clean_text'].apply(len)
    nHam = nWordsPerHamText.sum()

    # Number of words in dataset
    nVocabulary = len(vocabulary)

    # Laplace smoothing
    alpha = 1 # alpha = 2

    # Initiate parameters
    parametersSpam = {uniqueWord:0 for uniqueWord in vocabulary}
    parametersHam = {uniqueWord:0 for uniqueWord in vocabulary}

    # Calculate probabilities
    for word in vocabulary:
        nWordGivenSpam = spamTexts[word].sum() # spam texts already defined
        if not isinstance(nWordGivenSpam, (np.int64, np.float64)):
            continue
        pWordGivenSpam = (nWordGivenSpam + alpha) / (nSpam + alpha*nVocabulary)
        parametersSpam[word] = pWordGivenSpam

        nWordGivenHam = hamTexts[word].sum() # ham_texts already defined
        pWordGivenHam = (nWordGivenHam + alpha) / (nHam + alpha*nVocabulary)
        parametersHam[word] = pWordGivenHam
    
    parameters = { 'SPAM': parametersSpam , 'HAM': parametersHam }
    probabilities = { "SPAM": pSpam, 'HAM': pHam }
    
    return parameters, probabilities, testSet
    
   
def predict(text,probabilities,parameters):

    pSpamGivenText = probabilities["SPAM"]
    pHamGivenText = probabilities["HAM"]
    parametersSpam = parameters["SPAM"]
    parametersHam = parameters["HAM"]
    for word in text:
        if word in parametersSpam:
            pSpamGivenText *= parametersSpam[word]

        if word in parametersHam:
            pHamGivenText *= parametersHam[word]

    if pHamGivenText > pSpamGivenText:
        return 'ham'
    elif pSpamGivenText > pHamGivenText:
        return 'spam'
    else:
        return 'Misclassified'
    
def evaluateTest(parameters, probabilities, testSet):
    testSet['predicted'] = testSet['clean_text'].apply(predict,args=(probabilities, parameters))
    print(testSet.head(50))

    correct = 0
    total = testSet.shape[0]

    for row in testSet.iterrows():
      row = row[1]
      if row['Label'] == row['predicted']:
          correct += 1

    print('Correct:', correct)
    print('Incorrect:', total - correct)
    print('Accuracy:', correct/total)
    
    
def displaySpamAndHamWords(spamTexts,hamTexts):
    spamVocabulary, hamVocabulary = prepareForWordCloud(spamTexts,hamTexts)
    generate_wordcloud(spamVocabulary,"SPAM")
    generate_wordcloud(hamVocabulary,"HAM")

