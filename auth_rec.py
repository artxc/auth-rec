from nltk import sent_tokenize, FreqDist
from nltk.corpus import stopwords
from os import path, walk, chdir
import re
import pymorphy2
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.feature_selection import SelectKBest
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.svm import LinearSVC
import numpy as np
import pandas as pd
import preprocessing
import warnings

warnings.filterwarnings("ignore")

SRC_DIR = path.dirname(path.realpath(__file__))
CORPUS_PATH = path.join(SRC_DIR, 'corpus')
PROCESSED_CORPUS_PATH = path.join(SRC_DIR, 'processed_corpus')
FEATURES_FILE = path.join(SRC_DIR, 'features.csv')
STOPWORDS = set(stopwords.words('russian'))
RANGE = 25
NOMINATIVE_PRONOUNS = {'я', 'ты', 'он', 'она', 'оно', 'они', 'мы', 'вы'}


def load_book_features(file_name):
    text = open(file_name).read()
    morph = pymorphy2.MorphAnalyzer()

    sentence_list = sent_tokenize(text)

    usual_book_words = []
    sentences_length_dist = []
    words_length_dist = []
    pron_dist = []
    conj_dist = []

    for sentence in sentence_list:
        if sentence != ".":
            pron_count = 0
            conj_count = 0
            sentence_words = re.findall(r"[\w]+", sentence)
            sentences_length_dist.append(len(sentence_words))

            for word in sentence_words:
                words_length_dist.append(len(word))
                if word in NOMINATIVE_PRONOUNS:
                    pron_count += 1
                if morph.parse(word)[0].tag.POS == 'CONJ':
                    conj_count += 1
                if word not in STOPWORDS:
                    usual_book_words.append(word)

            conj_dist.append(conj_count)
            pron_dist.append(pron_count)

    sentence_length_freq_dist = FreqDist(sentences_length_dist)
    sentences_length_dist = [sentence_length_freq_dist.freq(i) for i in range(1, RANGE + 1)]
    sentences_length_dist.append(1 - sum(sentences_length_dist))

    words_length_freq_dist = FreqDist(words_length_dist)
    words_length_dist = [words_length_freq_dist.freq(i) for i in range(1, RANGE + 1)]
    words_length_dist.append(1 - sum(words_length_dist))

    pron_freq_dist = FreqDist(pron_dist)
    pron_dist = [pron_freq_dist.freq(i) for i in range(0, RANGE + 1)]
    pron_dist.append(1 - sum(pron_dist))

    conj_freq_dist = FreqDist(conj_dist)
    conj_dist = [conj_freq_dist.freq(i) for i in range(0, RANGE + 1)]
    conj_dist.append(1 - sum(conj_dist))

    words_freq_dist = FreqDist(usual_book_words)

    num_unique_words = len(words_freq_dist.keys())
    num_total_words = len(usual_book_words)

    hapax = len(words_freq_dist.hapaxes()) / num_unique_words
    dis = len([item for item in words_freq_dist if words_freq_dist[item] == 2]) / num_unique_words
    richness = num_unique_words / num_total_words

    result = [hapax, dis, richness, *sentences_length_dist, *words_length_dist, *pron_dist, *conj_dist]
    return result


def extract_books_features_from_corpus():
    x = []
    y = []

    for (dir_path, dir_names, file_names) in walk(PROCESSED_CORPUS_PATH):
        if file_names:
            chdir(dir_path)
            for file_name in file_names:
                y.append(path.basename(dir_path))
                features = load_book_features(file_name)
                x.append(features)
    le = LabelEncoder().fit(np.array(y))
    return np.array(x), np.array(le.transform(y)), le


def save_book_features_to_file(x, y, le):
    chdir(SRC_DIR)
    df = pd.DataFrame(x)
    df['code'] = y
    df['author'] = le.inverse_transform(y)
    df.to_csv(FEATURES_FILE, encoding='cp1251')


def load_features_from_file():
    chdir(SRC_DIR)
    features = pd.read_csv(path.basename(FEATURES_FILE), encoding='cp1251')
    y = features.as_matrix(['code'])
    x = features.as_matrix(map(str, range(107)))
    return x, y


def get_ovr_estiomators_prediction(estimators, x_test):
    y_predict = []
    for estimator in estimators:
        y_predict.append(estimator.predict(x_test))
    return np.array(y_predict)


def get_ovo_estimators_prediction(estimators, classes, x_test):
    n_classes = classes.shape[0]
    votes = np.zeros((1, n_classes))

    k = 0
    for i in range(n_classes):
        for j in range(i + 1, n_classes):
            prediction = estimators[k].predict(x_test)
            votes[prediction == 0, i] += 1
            votes[prediction == 1, j] += 1
            k += 1

    maxima = votes == np.max(votes, axis=1)[:, np.newaxis]
    if np.any(maxima.sum(axis=1) > 1):
        return -1
    else:
        return classes[votes.argmax(axis=1)]


def hybrid_clasification_for_fold(x_train, y_train, x_test, y_test):
    scaler = MinMaxScaler(feature_range=(0, 1))
    x_train, x_test = scaler.fit_transform(x_train), scaler.fit_transform(x_test)

    fs = SelectKBest(k=73)
    x_train = fs.fit_transform(x_train, y_train)
    x_test = fs.transform(x_test)
    estimator = LinearSVC(random_state=0, tol=1e-8, penalty='l2', C=1.5)

    # PHASE 1
    ovr = OneVsRestClassifier(estimator, n_jobs=-1)

    ovr.fit(x_train, y_train)

    y_predict_ovr = get_ovr_estiomators_prediction(ovr.estimators_, x_test).T

    y_test_predict = np.ones(len(y_test)) * -1
    unclassified_indexes_tests_phase1 = []

    for index, prediction in enumerate(y_predict_ovr):
        if np.sum(prediction) == 1:
            y_test_predict[index] = ovr.classes_[np.nonzero(prediction)[0][0]]
        else:
            unclassified_indexes_tests_phase1.append(index)

    correct_after_phase1 = np.sum(y_test_predict == y_test.T) / len(y_test)
    unclassified_after_phase1 = np.sum(y_test_predict == -1) / len(y_test)
    incorrect_after_phase1 = 1 - (correct_after_phase1 + unclassified_after_phase1)

    # PHASE 2
    ovo = OneVsOneClassifier(estimator, n_jobs=-1)

    ovo.fit(x_train, y_train)

    for index in unclassified_indexes_tests_phase1:
        y_predict_ovo = get_ovo_estimators_prediction(ovo.estimators_, ovo.classes_, x_test[index])
        if y_predict_ovo != -1:
            y_test_predict[index] = y_predict_ovo

    correct_after_phase2 = np.sum(y_test_predict == y_test.T) / len(y_test)
    unclassified_after_phase2 = np.sum(y_test_predict == -1) / len(y_test)
    incorrect_after_phase2 = 1 - (correct_after_phase2 + unclassified_after_phase2)

    accuracy = metrics.accuracy_score(y_test_predict, y_test)

    return np.array(
        [accuracy, correct_after_phase1, correct_after_phase2, incorrect_after_phase1, incorrect_after_phase2,
         unclassified_after_phase1, unclassified_after_phase2])


def hybrid_classification(x, y):
    cv = StratifiedShuffleSplit(n_splits=10, test_size=.3)
    results = []

    for train_index, test_index in cv.split(x, y):
        results.append(hybrid_clasification_for_fold(x[train_index], y[train_index], x[test_index], y[test_index]))

    accuracy, correct_after_phase1, correct_after_phase2, incorrect_after_phase1, \
    incorrect_after_phase2, unclassified_after_phase1, unclassified_after_phase2 = np.transpose(results)

    print("With hybrid classification, average correct after phase 1: %f" % np.mean(correct_after_phase1))
    print("With hybrid classification, average correct after phase 2: %f" % np.mean(correct_after_phase2))
    print("With hybrid classification, average incorrect after phase 1: %f" % np.mean(incorrect_after_phase1))
    print("With hybrid classification, average incorrect after phase 2: %f" % np.mean(incorrect_after_phase2))
    print("With hybrid classification, average unclassified after phase 1: %f" % np.mean(unclassified_after_phase1))
    print("With hybrid classification, average unclassified after phase 2: %f" % np.mean(unclassified_after_phase2))
    print("With hybrid classification, average accuracy: %f" % np.mean(accuracy))


def run_classification():
    if not path.exists(PROCESSED_CORPUS_PATH):
        preprocessing.preprocess(SRC_DIR)

    if not path.exists(FEATURES_FILE):
        x, y, le = extract_books_features_from_corpus()
        save_book_features_to_file(x, y, le)
    else:
        x, y = load_features_from_file()

    hybrid_classification(x, y)


if __name__ == '__main__':
    run_classification()
