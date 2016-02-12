import re
import requests
import numpy as np

from pattern.en import parse
from pyquery import PyQuery as pq
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.svm import SVC



###########
# Helper for text scrapping
###########

# Extract from the raw wikipedia page the text content
# of the paragraphs and the table of content as list of words


def get_text(data):
    # Extracting text content in paragraphs
    d_ = pq(data)
    d_p = pq(d_('#mw-content-text p'))
    paragraph = ''
    for r in d_p:
        paragraph += ' '+pq(r).text()

    # Extracing Table of Contents
    toc = []
    d_table = pq(d_('#toc li'))
    for c in d_table:
        t = pq(c).text()
        toc.append(' '.join(t.split(' ')[1:]))

    return paragraph, toc


# Return the title of the article, will be used to name the disease
def get_title(data):
    return pq(data)('#firstHeading').text()


# Check if string contains digit
def contains_digits(d):
    _digits = re.compile('\d')
    return bool(_digits.search(d))


# Return the common nouns in text (preprocessed) as a concatenated string
def get_words(thetext):
    stopwords = text.ENGLISH_STOP_WORDS
    punctuation = list('.,;:!?()[]{}`''\"@#$^&*+-|=~_')
    nouns = ''
    descriptives = ''
    proper = ''
    sentences = parse(thetext, tokenize=True, lemmata=True).split()
    for s in sentences:
        for token in s:
            if len(token[4]) > 1 and token[4] not in stopwords and token[4][0] not in punctuation and not contains_digits(token[4]):
                if token[1] in ['JJ', 'JJR', 'JJS']:
                    descriptives += ' ' + token[4]
                elif token[1] in ['NN', 'NNS']:
                    nouns += ' ' + token[4]
                elif token[1] in ['NNP', 'NNPS']:
                    proper += ' ' + token[4]
    return nouns, descriptives, proper


###########
# Helper for feature extraction and model learning
###########

# Train the lda model to extract feature in the list of nouns in noun list.
# Return the trained model (tuple (lda_ge, count_vect)) and the
# document_topics matrix from the list of nouns.
def train_lda_model(noun_list, num_topics):
    count_vect = CountVectorizer()
    X_counts = count_vect.fit_transform(noun_list)
    id2word = {k: v for v, k in count_vect.vocabulary_.iteritems()}
    corpus = build_corpus(X_counts)
    lda_ge = LdaModel(corpus, num_topics=num_topics, id2word=id2word)

    # Building the topics distribution for each document
    document_topics = np.zeros((len(corpus), num_topics))
    for i, bow in enumerate(corpus):
        dt = lda_ge.get_document_topics(bow)
        for t in dt:
            document_topics[i, t[0]] = t[1]

    return document_topics, lda_ge, count_vect


# Method to build the lda features for the document in the list of word
# counts vector X_counts based on a trained lda model.
def get_lda_features(X_counts, lda_ge):
    corpus = build_corpus(X_counts)
    infer = lda_ge.inference(corpus)[0]
    # Need to normalize the gamma to have the topic distribution
    document_topics = infer / np.sum(infer)

    return document_topics


# Build the corpus matrix in the required format to apply the lda
def build_corpus(X_counts):
    index = np.arange(X_counts.shape[1]).reshape((1, X_counts.shape[1]))
    corpus = []
    for i in xrange(X_counts.shape[0]):
        if (X_counts[i, :] > 0).sum():
            corpus.append([(w, c) for w, c in zip(index[np.where(
                X_counts[i, :].toarray())],
                np.array(X_counts[i, :][X_counts[i, :] > 0])[0])])
        else:
            corpus.append([])
    return corpus


# Function to build the occurence features and the pre-processing models
# from the list of nouns
def get_features(noun_list):
    # Building features
    count_vect = CountVectorizer()
    X_counts = count_vect.fit_transform(noun_list)
    tf_transformer = TfidfTransformer(use_idf=False).fit(X_counts)
    X_tf = tf_transformer.transform(X_counts)

    return X_tf, count_vect, tf_transformer


###########
# Helper for prediction
###########

# Predict probability response of the occurrences model
def predict_occ(X_counts, tf_transformer, svc):
    # Feature Extraction
    X_train_tf = tf_transformer.transform(X_counts)
    # Prediction
    y = svc.predict_proba(X_train_tf)[0][0]
    return y


# Predict probability response of the lda model
def predict_lda(X_counts, lda_ge, svc_lda):
    dt_new = get_lda_features(X_counts, lda_ge)
    y1 = svc_lda.predict_proba(dt_new)[0][0]

    return y1


# Retrieve the count matrix with the learned count_vect from the url
def get_counts(url, count_vect):
    t = requests.get(url)
    paragraph, toc = get_text(t.text)
    n, d, p = get_words(paragraph)
    X_counts = count_vect.transform([n + d])
    return X_counts


# Prediction with the stacked regression on new data from url
def predict_url(url, count_vect, tf_transformer, svc_occ, lda_ge, svc_lda,
                svc_stacked):
    X_counts = get_counts(url, count_vect)
    # Occurrence model
    y_occ = predict_occ(X_counts, tf_transformer, svc_occ)
    # Lda model
    y_lda = predict_lda(X_counts, lda_ge, svc_lda)
    # Stacked model
    y = svc_stacked.predict(np.array([[y_occ, y_lda]]))

    return y


###########
# Helper for information extraction
###########


def get_description_from_file(filename, path, kw_list):
    # identify if the class is negative or positive
    only_digit = re.compile('^[0-9]*$')
    if bool(only_digit.search(filename)):
        class_ = 'negative'
    else:
        class_ = 'positive'

    with open(path + class_ + '/' + filename) as f:
        data = f.read()
    return get_information(t.text, kw_list)


def get_description_from_url(url, kw_list):
    t = requests.get(url)
    title = get_title(t.text)
    return title, get_information(t.text, kw_list)


# Return sentences containing specific key-words
def get_information(data, kw_list):
    # Extracting Content
    paragraph, toc = get_text(data)
    # Result
    kw_to_info = {kw: u'' for kw in kw_list}

    # Removing the possible link
    sp_clean = [
        t for t in paragraph.split() if t not in ['[', ']'] and not contains_digits(t)]
    paragraph_clean = ' '.join(sp_clean)

    sentences = paragraph_clean.split('. ')
    for s in sentences:
        # Adding the content to corresponding kw if present
        # (by default the first one is chosen)
        for kw in kw_list:
            if kw in s:
                kw_to_info[kw] += s + u'. '
                break
    return kw_to_info


# Print in a pretty way the dictionary of disease description
def pretty_print_description(description):
    print u'Description of the disease: \n'
    for k, v in description.iteritems():
        if v:
            print(k.upper())
            print v + u' \n'


# Final function to classify a new entry given a pre-trained model and
# extract its information
def classify_from_url(url, count_vect, tf_transformer, svm_occ, kw_list):
    # Classification
    X_counts = get_counts(url, count_vect)
    y = predict_occ(X_counts, tf_transformer, svm_occ)
    res = bool(y < 0.5)
    # Extracting information if needed
    if res:
        title, description = get_description_from_url(url, kw_list)
        return res, title, description
    else:
        return res, None, None
