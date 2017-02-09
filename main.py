import time
import argparse

from sklearn.externals import joblib

# Local Module import
from helper import *

if __name__ == "__main__":
    kw_list = ['symptom', 'cause', 'prognosis', 'prevention',
               'treatment', 'drug', 'susceptibility', 'diagnosis']
    class_pred = {True: 'Disease', False: 'Non Disease'}

    # Parsing the arguments
    parser = argparse.ArgumentParser(description='Classify wikipedia pages')
    parser.add_argument('urls', metavar='U', nargs='*',
                        type=str, help='list of urls to classify')
    parser.add_argument('--model', dest='model',
                        type=int, default=2, help='choose the model version')
    parser.add_argument('--verbose', dest='verbose', type=bool, default=False,
                        help='To print the information for positive page')
    args = parser.parse_args()

    # Put name on each item
    names = [u.split('/')[-1] for u in args.urls]

    # Loading the pre-trained model
    svm_occ = joblib.load('model_v{}/svm_occ.pkl'.format(args.model))
    count_vect = joblib.load('model_v{}/count_vect.pkl'.format(args.model))
    tf_transformer = joblib.load(
        'model_v{}/tf_transformer.pkl'.format(args.model))

    for n, url in zip(names, args.urls):
        start = time.time()
        y, title, description = classify_from_url(url, count_vect,
                                                  tf_transformer, svm_occ,
                                                  kw_list)
        exec_t = time.time() - start
        print u'Prediction on {}: {} (in {} s) \n'.format(n, class_pred[y],
                                                          round(exec_t, 2))
        if args.verbose and title is not None:
            print u'Name of the disease: {} \n'.format(title)
            pretty_print_description(description)
