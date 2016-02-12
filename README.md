# Wiki_classifier

## Introduction

The task is to build a classifier to detect pages on disease from wikipedia pages and when a disease is detected to extract information about it (name, cause, features ...).

I decided to work first on this task in a ipython notebook because I find it more convenient for exploratory analysis and quick prototyping. I explain in the next paragraph my approach for solving this problem and provide the corresponding work in the notebook: 'Disease_classification.ipynb'. I also provide a ready to use pre-trained classifier in a python src file. Several libraries are required and then it is possible to classify a wikipedia page based on its url, and to retrieve relevant information about it if classified as a disease. The entire code to train the model is provided in the notebook but with the tight timeframe I decided only to pretrain a model and to provide a classification script. The outputs of the cell are provided as they contain meaningful information (accuracy of different model that I tried or I tuned).

Follow a part on how to use the classifier and one explaining my approach in details.

Thank you for reading me, I hope you will enjoy it and feel free to reach me with any questions or thoughts!

## Let's try it!

### Required environment

Python environment:

Python 2.7.11 :: Anaconda 2.4.1 (x86_64)

Libraries:

* re
* requests
* numpy
* pattern
* pyquery
* sklearn


### Example

The main.py script in the src contains the code to classify entry and retrieve information if needed. To use it you just need to call it from the command-line with your list of urls as argument. It will load the pre-trained model, classify your entry and print the results.

```
$ python main.py https://en.wikipedia.org/wiki/Contagion_\(film\) https://en.wikipedia.org/wiki/Paracetamol https://en.wikipedia.org/wiki/Appendicitis --model 1
```

Arguments to provide:

[required]: space separated urls to classify (unlimited number)

[optional]: --model [2] or 1, to chose the model (by default 2), 1 is the first version (without extension of the training set by drugs pages) and 2 the second version (with extension)

## Approach


I chose to divide my work pipeline into several phases:
* 1 scrapping
* 2 feature extraction
* 3 classification
* 4 specific information extraction

### 1 scrapping

I first scrapped the content of the paragraph in the wikipedia pages. I separately scraped the table of contents as they may have a specific pattern for the pages about disease or could be helpful to extract specific information (if a part is called 'cause' it may be interesting). But the first experiments in the classification task were not convincing so I kept it in my scrapping phase but do not use it after. This phase with the tagging part of the next phase take around 20 minutes to run.

To expand this phase, we could think on scrapping images or figures of the wikipedia pages to analyze them. We could also extract the external links or links to other wikipedia pages. This could provide a similarity mapping among the pages.

### 2 feature extraction

Once I have the list of words of a page, I decided to focus on the most relevant words. I applied a part-of-speech tagging and decided to keep only the nouns and the descriptive words (first separately but then realise their performance in the model were independent so I merged them). I also retrieved the Proper nouns but this distribution seems to be too sparse from one page to another to be really helpful. I also removed all the english stop words and all the different noisy symbols. The output is a list of standardized words tagged as nouns and adjectives. I then computed the occurrences of these words inside each document to build a words count vector. This first embedding was then used in two different ways.

First, I transformed it into frequency to make these vectors comparable from one document to another as their length varies a lot. These frequency vectors define a first feature vector I called occurrence.
Second, I use these counts to apply a topic modeling algorithm (I chose the LDA, from a package with an online variational approach to make easier the feature extraction on a new entry once the model has already been computed). This provides for each document a topic distribution which builds a second feature vector I called lda. This second approach needs more work as the lda contains one hyperparameter to tune: the number of topics. I applied my classification pipeline and tuned the parameter based on the mean accuracy on the test set (here used as a validation set). This approach was an easy and quick way to tune my parameter. However, to have a more accurate tuning we should digg deeper into the lda and use a tailored evaluation metric, one commonly used is the perplexity. This could also be a way to improve this work with more time.

One important thing is to keep an 'online' approach, i.e. make it easy to extract features for new entry once the model has been trained. For the first feature vector, we just have to count the occurrences of the words already in the vocabulary built during the training phase without considering the new words. For the lda, the online version makes it also easy to infer the topic distribution of the new entry without changing our model.

To expand this part, we could think of extending the word counts with ngrams counts, i.e. counting the co-occurrence of n consecutive words among the document. This could provide good results as the same kind of pattern is often repeated in a disease page (e.g. 'This syndrome is caused by...'). 

### 3 classification

With these two numerical feature vectors, I am able to apply traditional classifiers. 
First, I split the entire dataset in a train and test set using a proportion of 70-30 to test my models. I evaluated them with the mean accuracy measured on the test set. Then I defined a naive baseline (model predicting only non-disease) which provides a mean accuracy of about 73%.

First, I tried on the occurrence feature a naive bayes, a logistic regression (with the two penalizations l1 and l2) and a linear support vector machine.  The best result was given by the svm whith around 98% of accuracy on the test set. I applied the same approach on the lda feature, coming up (with a tuned number of topics) with a mean accuracy of about 95 %. My gut feeling was that the lda feature enables to take into account the context induced by co-occuring words so it classifies in a different way than the occurence feature. As a result, I decided to build an ensemble method, applying in parallel the two svms to each of my two feature vector and then applying another svm to the concatenation of the two output vectors provided by the first two svms (as probability vector). Nonetheless, the result was almost the same than the simple svm on the occurence feature (with sometimes a little improvement, around 0.1% but not convincing with regards to the overhead in time it adds, around 1 minute). At the end, I decided to stick with the first svm on the occurence feature as my final classifier because it was the fastest and roughly the most accurate.

Nonetheless, it was not classifying rightly the drugs article. To face this drug missclassification I thought about building a finer classifier while training it on the same training set expanded with negative pages really close to diseases (drug, general health related topic). The provided negative training set is in deed really different than the positive one so the model was only able to extract the global difference between for example a disease page and a art related page. This extension will forces the model to grasp a finer understanding of the different patterns involved and it should be better at discriminating drugs from diseases. One drawback is that it requires to identify pages very similar to disease pages manually. I scrapped from 4 general pages listing drugs about 700 urls of drug wikipedia pages. I built again the model on this new expanded training set and evaluate my result. First, I reached the same accuracy on the training and test set which is pretty impressive as it means that the negative drugs pages in the test set were most of the time well classified. I tested it then on the 4 same drugs where the first model failed and it was a success for all of them! (I checked that they were not in the extended training set to avoid a biaised evaluation).

To expand these part, we could tune the model used with a cross validation approach. I did not digg deeper into that part because of the tight timeframe and also because my model provided satisfying result with standard hyperparameters. Moreover, we could also think of deeper models able to take into account more information from the structure of the documents, e.g. neural networks. The overall stacked approach takes less a minute to train and to apply on new data.

### 4 specific information extraction

Lastly, for the pages classified as disease, we wanted to extract the name of the disease and as much information as possible about it. For the name of the disease, I retrieved the name of the wikipedia page as it seems to be a pretty good proxy. Then for the information I focused on specific key-words I manually defined:

['symptom', 'cause', 'prognosis', 'prevention', 'treatment', 'drug', 'susceptibility', 'feature', 'frequent']

I looked for the sentences containing these key words. First I tried to retrieve the global meaning of the sentence with only the nouns and adjectives after a part of speech tagging phase. But the results were pretty hard to understand so I just retrieve the whole sentence. This approach provides interesting information but it could be improved as the description may be pretty big for disease with long wikipedia page. Also it assumes we defined manually a list of key words but even if the wikipedia pages are following the same structure, some of them may use different words and we may miss relevant information.

To fix these problems we could consider the co-occurency of specific words. For instance, if we consider the most co-occuring words of the nouns in our list, we may retrieve either synonyms, either words which explain and define further the first word. This second category is exactly the one we are looking for. As a result, with more time I would try to retrieve for each word in my list the top (maybe 200 or more) most co-occuring words and look for them in each disease page. Then I could retrieve them and also their surroundings. This should provide more complete information. Otherwise, we could also scrap more information over the internet once we know the name of the disease, for example in an encyclopedia (https://www.nlm.nih.gov/medlineplus/).

As a conclusion, this first work provides really promising results for the classification task, but which could still be improved for the edge cases. Extracting a relevant and synthetic description of the disease found is then much harder. Anyway, I really enjoyed working on it as it maed me play with different tools and concepts commonly used in a data modeling pipeline, and especially because I was able at the end to try my classifier on any given wikipedia page!

2016, Nicolas Drizard
