# Create WordCloud with spam and ham data.
from wordcloud import WordCloud
import matplotlib.pyplot as plt

def generate_wordcloud(data,title):
  uniqueString=(" ").join(data)
  wc = WordCloud(width=400, height=330, max_words=150,colormap="Dark2").generate(uniqueString)
  plt.figure(figsize=(10,8))
  plt.imshow(wc, interpolation='bilinear')
  plt.axis("off")
  plt.title(title,fontsize=13)
  plt.show()

    
def describeTrainAndTestSet(trainingSet,testSet):
    print("Training set shape: ",trainingSet.shape)
    print("Test set shape: ",testSet.shape)
    
    print("Proportion in training dataset")
    print(trainingSet['Label'].value_counts(normalize=True))

    print("\nProportion in test dataset")
    print(testSet['Label'].value_counts(normalize=True))