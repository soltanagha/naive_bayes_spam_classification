# Spam filter

Bayesian classifier belongs to the category of machine learning. The main aspect is this: the system, which is faced with the task of determining whether the next text is spam, is trained in advance with a certain number of texts that are known exactly where "spam" and where "not spam". It has already become clear that this is training with a teacher, where we play the role of a teacher. The Bayesian classifier represents a document (in our case, a texts) as a set of words that supposedly do not depend on each other (this is where that naivety comes from).

It is necessary to calculate the score for each class (spam / not spam) and select the one that turned out to be the maximum. For this we use the following formula:

<img width="274" alt="Bayes classifier" src="https://user-images.githubusercontent.com/78455627/115973516-a34e5880-a566-11eb-8966-10145ff9c1bf.png">

https://en.wikipedia.org/wiki/Naive_Bayes_classifier
