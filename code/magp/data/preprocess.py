#coding=utf-8

import re
import glob
import json
import copy
import datetime
import pickle
import gensim
import pandas as pd
import numpy as np
from operator import itemgetter
from collections import defaultdict

import scipy
from scipy.sparse import csr_matrix
from scipy.spatial.distance import cosine

from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
porter_stemmer = PorterStemmer()

from magp.common import *


############################crowdsourcing data files####################################
def preprocess_text(text):

    # remove symbols
    text = re.sub('[#$%&()*+-/<=>?@[\\]^_`{|}~]+', ' ', text)
    text = re.sub('\s+', ' ', text)
    return text


def format_cs2010_data(raw_data_file, crowdsourcing_corpus_path, output_corpus_path, output_file):
    """ Format cs2010 crowd sourcing data.
        Prepare documents."""

    df = pd.read_csv(raw_data_file, sep='\s+', names=['topicID', 'workerID', 'docID', 'gold', 'label'], header=0)

    # let label and gold >= 0
    df = df.loc[df["gold"] >= 0]
    df = df.loc[df["label"] >= 0]
    df['gold'] = df['gold'].map(lambda x: RELEVANT if x > 0 else NON_RELEVANT)
    df['label'] = df['label'].map(lambda x: RELEVANT if x > 0 else NON_RELEVANT)

    # check whether doc in df is also in crowd sourcing corpus
    if not os.path.exists(output_corpus_path):
        os.makedirs(output_corpus_path)

    existing_docs = []
    for doc_id in df.loc[:, 'docID'].unique():
        doc_id_in_corpus = False
        for file_path in glob.glob(crowdsourcing_corpus_path + '/*/*/*.txt'):
            if re.search(doc_id, file_path):
                doc_id_in_corpus = True

                with open(file_path, encoding='utf-8') as fr:
                    document = preprocess_text(fr.read())
                with open(os.path.join(output_corpus_path, doc_id), 'w', encoding='utf-8') as f:
                    f.write(document)

                break
            else:
                pass
        if doc_id_in_corpus:
            existing_docs.append(doc_id)

    # only keep these doc_ids
    df = df.loc[df['docID'].isin(existing_docs)]

    # write
    output_df = df[COLUMNS]
    output_df.to_csv(output_file, index=False)

    return


def format_cs2011_data(raw_data_file, crowdsourcing_corpus_path, output_corpus_path, output_file):
    """
    Format cs2011 crowd sourcing data.
    Prepare documents."""

    df = pd.read_csv(raw_data_file, sep='\s+',
                     names=['topicID', 'taskID', 'workerID', 'docID', 'gold', 'label'], header=None)

    # let label and gold >= 0
    df = df.loc[df["gold"] >= 0]
    df = df.loc[df["label"] >= 0]

    # check whether doc in df is also in crowd sourcing corpus
    if not os.path.exists(output_corpus_path):
        os.makedirs(output_corpus_path)

    existing_docs = []
    for doc_id in df.loc[:, 'docID'].unique():
        doc_id_in_corpus = False
        for file_path in glob.glob(crowdsourcing_corpus_path + '/*/*/*.txt'):
            if re.search(doc_id, file_path):
                doc_id_in_corpus = True

                with open(file_path, encoding='utf-8') as fr:
                    document = preprocess_text(fr.read())
                with open(os.path.join(output_corpus_path, doc_id), 'w', encoding='utf-8') as f:
                    f.write(document)

                break
            else:
                pass
        if doc_id_in_corpus:
            existing_docs.append(doc_id)

    # only keep these doc_ids
    df = df.loc[df['docID'].isin(existing_docs)]
    # write
    output_df = df[COLUMNS]
    output_df.to_csv(output_file, index=False)

    return


def parse_topic_badformat(fpath):  # Really bad! TREC topic files are not XML!!!
    queries = {}
    state = ''

    with open(fpath, 'r') as f:

        for line in f:
            if state == 'desc':
                query += line
                queries[qid] = query
                state = ''

            if line.startswith('<num>'):
                qid = line.replace('<num>', '').replace('Number:', '').strip()

            elif line.startswith('<title>'):
                query = line.replace('<title>', '')

            elif line.startswith('<desc>'):
                state = 'desc'
            else:
                pass

    return queries


def format_cs2010_2011_topic(crowdsourcing_data_file, topic_file, output_corpus_path):

    # topic_ids
    df = pd.read_csv(crowdsourcing_data_file)
    topic_ids = df.loc[:, 'topicID'].unique()

    # topic file
    queries = {}
    with open(topic_file, 'r', encoding='latin-1') as f:
        for line in f:
            qid = line.split(':')[0].strip()
            query = line.split(':')[2].strip()
            if int(qid) in topic_ids:
                queries[qid] = query

    # write
    if not os.path.exists(output_corpus_path):
        os.makedirs(output_corpus_path)

    for topic_id in topic_ids:
        topic_str = queries[str(topic_id)]
        topic_str = preprocess_text(topic_str)
        with open(os.path.join(output_corpus_path, str(topic_id)), 'w', encoding='utf-8') as f:
            f.write(topic_str)

    return


def format_run(crowdsourcing_file, raw_run_path, run_path):

    if not os.path.exists(run_path):
        os.makedirs(run_path)

    df = pd.read_csv(crowdsourcing_file)
    topics = df['topicID'].unique()
    docs = df['docID'].unique()

    for file in get_file_ids(raw_run_path):
        with open(os.path.join(raw_run_path, file)) as fr:
            dct = defaultdict(list)
            # only keep qid-did appearing in crowdsourcing_file
            for line in fr:
                qid, dummy, did, rank, score, team = line.split()

                if int(qid) in topics:
                    if did in docs:
                        dct[qid].append((qid, dummy, did, rank, score, team))
                    else:
                        pass
                else:
                    pass
            # make sure all rank number start from 1
            with open(os.path.join(run_path, file), 'w') as fw:
                for key in dct:
                    dct[key].sort(key=itemgetter(4), reverse=True)
                    for i, (qid, dummy, did, rank, score, team) in enumerate(dct[key]):
                        fw.write('{}\t{}\t{}\t{}\t{}\t{}\n'.format(qid, dummy, did, i+1, score, team))
    return


############################feature files####################################
def text_generater(corpus_path, stem=False, lower=False, stopword=False):
    doc_ids = get_file_ids(corpus_path)
    assert doc_ids != []
    for doc_id in doc_ids:
        with open(os.path.join(corpus_path, doc_id), 'r', encoding='utf8') as f:
            document = f.read()
            if stem:
                document = ' '.join(porter_stemmer.stem(token) for token in word_tokenize(document))
            if lower:
                document = ' '.join(token.lower() for token in word_tokenize(document))
            if stopword:
                document = ' '.join(token for token in word_tokenize(document) if token not in ENGLISH_STOP_WORDS)
            yield document


def id_text_generater(corpus_path, stem=False, lower=False, stopword=False):
    doc_ids = get_file_ids(corpus_path)
    assert doc_ids != []
    for doc_id in doc_ids:
        with open(os.path.join(corpus_path, doc_id), 'r', encoding='utf8') as f:
            document = f.read()
            if stem:
                document = ' '.join(porter_stemmer.stem(token) for token in word_tokenize(document))
            if lower:
                document = ' '.join(token.lower() for token in word_tokenize(document))
            if stopword:
                document = ' '.join(token for token in word_tokenize(document) if token not in ENGLISH_STOP_WORDS)
            yield doc_id, document


def tfidf(data_name, topic_or_document, enable_write=False):
    tf_dict = {}
    tfidf_dict = {}

    fit_corpus_path = PATH(data_name, 'document_corpus')
    if 'document' == topic_or_document:
        transform_corpus_path = PATH(data_name, 'document_corpus')
    elif 'topic' == topic_or_document:
        transform_corpus_path = PATH(data_name, 'topic_corpus')
    else:
        raise ValueError

    # fit model
    tf_model = CountVectorizer(min_df=3)
    tf_model.fit(text_generater(fit_corpus_path, stem=True, lower=True, stopword=True))

    # fit model
    tfidf_model = TfidfVectorizer(min_df=3)
    tfidf_model.fit(text_generater(fit_corpus_path, stem=True, lower=True, stopword=True))

    # transform
    transform_corpus_path = PATH(data_name, transform_corpus_path)
    tf_vecs = tf_model.transform(text_generater(transform_corpus_path))
    tfidf_vecs = tfidf_model.transform(text_generater(transform_corpus_path, stem=True, lower=True, stopword=True))

    for i, doc_id in enumerate(get_file_ids(transform_corpus_path)):
        tf_dict[doc_id] = tf_vecs[i]
        tfidf_dict[doc_id] = tfidf_vecs[i]

    # write
    if enable_write:
        feature_path = PATH(data_name, 'feature')
        with open(os.path.join(feature_path, 'tfidf.{}.pickle'.format(topic_or_document)), 'wb') as f:
            pickle.dump((tf_dict, tfidf_dict, tfidf_model), f)

    return tf_dict, tfidf_dict, tfidf_model


def word2vec_all(data_name, topic_or_document, enable_write=False):
    mdict = {}

    if 'document' == topic_or_document:
        corpus_path = PATH(data_name, 'document_corpus')
    elif 'topic' == topic_or_document:
        corpus_path = PATH(data_name, 'topic_corpus')
    else:
        corpus_path = ''
    # model
    model = gensim.models.KeyedVectors.load_word2vec_format(GOOGLE_WORD2VEC_MODEL_PATH, binary=True)
    word_vectors = model.wv
    del model

    # transform
    for doc_id, document in id_text_generater(corpus_path, lower=True, stopword=True):
        word_num = 0
        doc_vec = np.zeros(300)  # google model has 300 dimension
        for token in document.split():
            try:
                token_vec = word_vectors[token]
            except:
                print('word2vec_all: token {} not in word_vectors'.format(token))
                token_vec = np.zeros(300)
                pass
            doc_vec += token_vec
            word_num += 1
        vec = doc_vec/word_num
        vec = vec / np.linalg.norm(vec)
        mdict[doc_id] = vec

    # write
    if enable_write:
        feature_path = PATH(data_name, 'feature')
        with open(os.path.join(feature_path, 'word2vec_all.{}.pickle'.format(topic_or_document)), 'wb') as f:
            pickle.dump(mdict, f)

    return mdict


def word2vec_percentage(data_name, topic_or_document,
                        tfidf_dict, tfidf_model,
                        word_percentage=0.2, enable_write=False):
    mdict = {}

    feature_path = PATH(data_name, 'feature')

    if 'document' == topic_or_document:
        corpus_path = PATH(data_name, 'document_corpus')
    elif 'topic' == topic_or_document:
        corpus_path = PATH(data_name, 'topic_corpus')
    else:
        corpus_path = ''

    # word2vec model
    model = gensim.models.KeyedVectors.load_word2vec_format(GOOGLE_WORD2VEC_MODEL_PATH, binary=True)
    word_vectors = model.wv
    del model

    # transform
    for doc_id, document in id_text_generater(corpus_path, lower=True, stopword=True):

        doc_tfidf_vec = tfidf_dict[doc_id].tocoo()

        # row = doc_tfidf_vec.row
        col = doc_tfidf_vec.col
        data = doc_tfidf_vec.data
        valid_token_num = int(word_percentage*len(data))

        indices_weights = sorted(zip(col, data), key=itemgetter(1), reverse=True)[:valid_token_num]
        indices_weights_dict = dict(indices_weights)

        # average the word vector for top xx percent of the words in document
        total_weights = 1e-30
        doc_vec = np.zeros(300)  # google model has 300 dimension
        for token in word_tokenize(document):
            stemmed_token = porter_stemmer.stem(token)
            if tfidf_model.vocabulary_.get(stemmed_token):
                index = tfidf_model.vocabulary_[stemmed_token]
                if indices_weights_dict.get(index):
                    weight = indices_weights_dict.get(index, 0)
                    word_vec = word_vectors.get(token, lower=True)
                    doc_vec += word_vec * weight
                    total_weights += weight

        vec = doc_vec / total_weights
        vec = vec / np.linalg.norm(vec)
        mdict[doc_id] = vec

    # write
    if enable_write:
        with open(os.path.join(feature_path, 'word2vec_percentage{}.{}.pickle'.format(int(word_percentage*100),
                                                                                      topic_or_document)), 'wb') as f:
            pickle.dump(mdict, f)

    return mdict


def feature_tf(topic_tf_dict, doc_tf_dict, tid_did_list):
    x = []

    for topic_id, doc_id in tid_did_list:
        topic_vec_mask = topic_tf_dict[topic_id].astype(np.float)
        topic_vec_mask.data[:] = 1  # set 1 if the term is in topic (sparse matrix)
        doc_tf_vec = doc_tf_dict[doc_id].astype(np.float)
        summ = topic_vec_mask.dot(doc_tf_vec.T)[0, 0]  # inner product
        x.append(summ)

    x = np.array(x).reshape(-1, 1)
    print('{}: finish extracting feature_tf.'.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    return x


def feature_idf(topic_tf_dict, doc_tfidf_model, tid_did_list):
    x = []

    idf = csr_matrix(doc_tfidf_model.idf_, dtype=np.float)

    for topic_id, doc_id in tid_did_list:
        topic_vec_mask = topic_tf_dict[topic_id].astype(np.float)
        topic_vec_mask.data[:] = 1
        summ = topic_vec_mask.dot(idf.T)[0, 0]  # inner product
        x.append(summ)

    x = np.array(x).reshape(-1, 1)
    print('{}: finish extracting feature_idf.'.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    return x


def feature_tfidf(topic_tf_dict, doc_tfidf_dict, tid_did_list):
    x = []

    for topic_id, doc_id in tid_did_list:

        topic_vec_mask = topic_tf_dict[topic_id].astype(np.float)
        topic_vec_mask.data[:] = 1
        doc_tfidf_vec = doc_tfidf_dict[doc_id].astype(np.float)
        summ = topic_vec_mask.dot(doc_tfidf_vec.T)[0, 0]  # dot product
        x.append(summ)

    x = np.array(x).reshape(-1, 1)
    print('{}: finish extracting feature_tfidf.'.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    return x


def feature_tfidf_based_cosine_score(topic_tfidf_dict, doc_tfidf_dict, tid_did_list):
    x = []

    for topic_id, doc_id in tid_did_list:
        topic_vec = topic_tfidf_dict[topic_id]
        doc_vec = doc_tfidf_dict[doc_id]
        score = topic_vec.dot(doc_vec.T)[0, 0]  # dot product
        score /= scipy.sparse.linalg.norm(topic_vec) * scipy.sparse.linalg.norm(doc_vec)  # cosine similarity
        x.append(score)

    x = np.array(x).reshape(-1, 1)
    print('{}: finish extracting feature_tfidf_based_cosine_score.'.
          format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    return x


def feature_word2vec_based_cosine_score(topic_model, doc_model, tid_did_list):
    x = []

    for topic_id, doc_id in tid_did_list:
        topic_vec = topic_model[topic_id]
        doc_vec = doc_model[doc_id]
        if np.linalg.norm(topic_vec) == 0 or np.linalg.norm(doc_vec) == 0:
            score = 1  # if one of the vector is oov, the distance is big
        else:
            score = cosine(topic_vec, doc_vec)  # cosine distance
        assert score != np.nan
        x.append(score)

    x = np.array(x).reshape(-1, 1)
    print('{}: finish extracting feature_word2vec_based_cosine_score.'.
          format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    return x


def feature_word2vec_percentage20_based_cosine_score(topic_model, doc_model, tid_did_list):
    x = []

    for topic_id, doc_id in tid_did_list:
        topic_vec = topic_model[topic_id]
        doc_vec = doc_model[doc_id]
        if not np.isfinite(doc_vec).all() or not np.isfinite(topic_vec).all():
            print(topic_id, doc_id)
            print(doc_vec)
            print(topic_vec)
        if np.linalg.norm(topic_vec) == 0 or np.linalg.norm(doc_vec) == 0:
            score = 1
        else:
            score = cosine(topic_vec, doc_vec)  # cosine distance
        assert score != np.nan
        x.append(score)

    x = np.array(x).reshape(-1, 1)
    print('{}: finish extracting feature_word2vec_percentage20_based_cosine_score.'.
          format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    return x


def feature_bm25_score(document_corpus_path, topic_tf_dict, doc_tf_dict, doc_tfidf_model, tid_did_list):
    """
    bm25(q,d) = idf * tf * (k1+1) / tf + k1 (1 - b + b * |d|/avg_dl)
    Implemented based on https://en.wikipedia.org/wiki/Okapi_BM25
    :param instances:
    :return:
    """
    x = []

    k1 = 1.2
    b = 0.75

    # global variable
    dl_dict = {}
    for doc_id, doc in id_text_generater(document_corpus_path):
        doc_len = len(doc.split())
        dl_dict[doc_id] = doc_len

    # calculate avg_dl
    avg_dl = sum(dl_dict.values()) / len(dl_dict.keys())

    idf = csr_matrix(doc_tfidf_model.idf_, dtype=np.float)

    for topic_id, doc_id in tid_did_list:
        topic_vec_mask = topic_tf_dict[topic_id].astype(np.float)
        topic_vec_mask.data[:] = 1.0
        doc_tf_vec = doc_tf_dict[doc_id].astype(np.float)
        doc_tf_vec = topic_vec_mask.multiply(doc_tf_vec)  # tf for document
        # tf * (k1 + 1)
        temp0 = doc_tf_vec * (k1+1)
        # tf + k1 * (1 - b + b * |d|/avg_dl)
        temp1 = copy.deepcopy(doc_tf_vec)
        temp1.data[:] = k1 * (1 - b + b * (dl_dict[doc_id] / avg_dl))
        temp1 = doc_tf_vec + temp1
        temp2 = [i*1.0/j for i, j in zip(temp0.data, temp1.data)]
        temp3 = csr_matrix((temp2, temp1.indices, temp1.indptr), shape=temp1.shape)
        # bm25
        score = temp3.dot(idf.T)[0, 0]  # sum

        x.append(score)

    x = np.array(x).reshape(-1, 1)
    print('{}: finish extracting feature_bm25_score.'.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    return x


def feature_lm_jm_score(topic_tf_dict, doc_tf_dict, tid_did_list):
    """
    Implemented based on A Study of Smoothing Methods for Language Models Applied to Ad Hoc Information Retrieval
    :param instances:
    :return:
    """
    x = []
    lmd = 0.5

    # calculate
    corpus_tf_vec = None
    for doc_tf_vec in doc_tf_dict.values():
        if corpus_tf_vec is not None:
            corpus_tf_vec += doc_tf_vec
        else:
            corpus_tf_vec = doc_tf_vec

    for topic_id, doc_id in tid_did_list:
        topic_vec_mask = topic_tf_dict[topic_id].astype(np.float)
        topic_vec_mask.data[:] = 1.0
        doc_tf_vec = doc_tf_dict[doc_id].astype(np.float)

        doc_tf_vec = doc_tf_vec.multiply(topic_vec_mask.T)

        corpus_tf_vec_ = corpus_tf_vec.multiply(topic_vec_mask.T)

        # sum_i { log(lambda*p(wi|d) + (1-lambda)*p(wi|C)) }
        score = lmd*doc_tf_vec/doc_tf_vec.sum() + (1-lmd)*corpus_tf_vec_/corpus_tf_vec_.sum()
        score = score.log1p().sum()  # log1p(x) := log(1+x)

        x.append(score)

    x = np.array(x).reshape(-1, 1)
    print('{}: finish extracting feature_lm_jm_score.'.
          format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    return x


def lexical_feature(data_name):

    document_corpus_path = PATH(data_name, 'document_corpus')
    topic_corpus_path = PATH(data_name, 'topic_corpus')

    # preliminaries
    print('1')
    # topic_tf_dict, topic_tfidf_dict, topic_tfidf_model = tfidf(data_name, 'topic')
    # doc_tf_dict, doc_tfidf_dict, doc_tfidf_model = tfidf(data_name, 'document')
    print('2')
    topic_word2vec_model = word2vec_all(data_name, 'topic')
    doc_word2vec_model = word2vec_all(data_name, 'document')
    print('3')
    # topic_word2vecpct_model = word2vec_percentage(data_name, 'topic', topic_tfidf_dict, topic_tfidf_model)
    # doc_word2vecpct_model = word2vec_percentage(data_name, 'document', doc_tfidf_dict, doc_tfidf_model)
    print('4')
    # crowd sourcing data
    # df = pd.read_csv(os.path.join(DATA_DIR, data_name, 'crowdsourcing_data.csv'), sep=',', names=COLUMNS, header=0,
    #                  dtype=COLUMN_TYPES)
    # df = df[['topicID', 'docID']].drop_duplicates()
    # tid_did_list = df.values

    # extract feature
    # x = np.concatenate((
    #     feature_tf(topic_tf_dict, doc_tf_dict, tid_did_list),
    #     feature_idf(topic_tf_dict, doc_tfidf_model, tid_did_list),
    #     feature_tfidf(topic_tf_dict, doc_tfidf_dict, tid_did_list),
    #     feature_tfidf_based_cosine_score(topic_tfidf_dict, doc_tfidf_dict, tid_did_list),
    #     feature_word2vec_based_cosine_score(topic_word2vec_model, doc_word2vec_model, tid_did_list),
    #     feature_word2vec_percentage20_based_cosine_score(topic_word2vecpct_model, doc_word2vecpct_model, tid_did_list),
    #     feature_bm25_score(document_corpus_path, topic_tf_dict, doc_tf_dict, doc_tfidf_model, tid_did_list),
    #     feature_lm_jm_score(topic_tf_dict, doc_tf_dict, tid_did_list),
    # ), axis=1)
    # x = preprocessing.StandardScaler().fit_transform(x)
    #
    # # write
    # dct = {}
    # for (qid, did), feat in zip(df.values, x):
    #     dct[qid+did] = feat
    # with open(os.path.join(DATA_DIR, data_name, 'feature', 'lexical.pickle'), 'wb') as f:
    #     pickle.dump(dct, f)

    return


def system_run_rank_scaler(data_name):
    run_path = os.path.join(DATA_DIR, data_name, 'runs')
    dct_path = os.path.join(DATA_DIR, data_name, 'feature', 'rank.scaler.pickle')
    csd_file = os.path.join(DATA_DIR, data_name, 'crowdsourcing_data.csv')

    # crowd sourcing data
    csd_df = pd.read_csv(csd_file, sep=',', names=COLUMNS, header=0, dtype=COLUMN_TYPES)
    csd_df = csd_df[['topicID', 'docID']].drop_duplicates()

    qiddid_list = [qid.strip() + did.strip() for (qid, did) in csd_df.values]
    row_num = len(qiddid_list)
    run_list = get_file_ids(run_path)
    col_num = len(run_list)
    rank_one_hot_feature = np.zeros((row_num, col_num))

    for run in get_file_ids(run_path):
        with open(os.path.join(run_path, run)) as fr:
            for line in fr:
                qid, dummy, did, rank, score, team = line.split()
                if qid.strip() + did.strip() in qiddid_list:
                    i = qiddid_list.index(qid.strip() + did.strip())
                    j = run_list.index(run)
                    rank_one_hot_feature[i][j] = rank
                else:
                    pass

    rank_sum = rank_one_hot_feature.sum(axis=-1).reshape(-1, 1)
    scaled_feature = preprocessing.StandardScaler().fit_transform(rank_sum).reshape(-1, 1)
    dct = dict(zip(qiddid_list, scaled_feature))

    with open(dct_path, 'wb') as f:
        pickle.dump(dct, f)
    return


def system_run_rank_onehot(data_name):

    run_path = os.path.join(DATA_DIR, data_name, 'runs')
    dct_path = os.path.join(DATA_DIR, data_name, 'feature', 'rank.onehot.pickle')
    csd_file = os.path.join(DATA_DIR, data_name, 'crowdsourcing_data.csv')

    # crowd sourcing data
    csd_df = pd.read_csv(csd_file, sep=',', names=COLUMNS, header=0, dtype=COLUMN_TYPES)
    csd_df = csd_df[['topicID', 'docID']].drop_duplicates()

    qiddid_list = [qid.strip() + did.strip() for (qid, did) in csd_df.values]
    row_num = len(qiddid_list)
    run_list = get_file_ids(run_path)
    col_num = len(run_list)
    rank_one_hot_feature = np.zeros((row_num, col_num))

    for run in get_file_ids(run_path):
        with open(os.path.join(run_path, run)) as fr:
            for line in fr:
                qid, dummy, did, rank, score, team = line.split()
                if qid.strip()+did.strip() in qiddid_list:
                    i = qiddid_list.index(qid.strip()+did.strip())
                    j = run_list.index(run)
                    rank_one_hot_feature[i][j] = rank

                else:
                    pass
    print('piddid num', row_num, 'run num',col_num, 'slot num (piddid num * run num)', row_num * col_num, 'slot that has no rank', np.sum(rank_one_hot_feature == 0))

    max_rank = rank_one_hot_feature.max()

    # if there is no value from that run file, assume that run file ranks the qiddid at very bottom
    rank_one_hot_feature[rank_one_hot_feature == 0] = max_rank

    scaled_feature = preprocessing.StandardScaler().fit_transform(rank_one_hot_feature)

    dct = dict(zip(qiddid_list, scaled_feature))

    with open(dct_path, 'wb') as f:
        pickle.dump(dct, f)
    return


def clueweb_bert_vector(data_name):
    vector_file = os.path.join(CODE_DIR, 'external_resources', 'SIGIR19-BERT-IR-master/initial_document_rankings/temp/bert-models-cw-title-firstp-{}-fold1-test_results.json'.format(data_name))
    vector_dct = {}
    with open(vector_file, 'r') as f:
        for line in f:
            dct = json.loads(line)
            qid = dct['qid'].strip()
            did = dct['did'].strip()
            vector = dct['pooled_vector']
            vector_dct[qid+did] = vector

    pickle_file = os.path.join(DATA_DIR, data_name, 'feature', 'clueweb09.bert.pickle')
    with open(pickle_file, 'wb') as fw:
        pickle.dump(vector_dct, fw)

    return



########################################################################################################

def prepare_crowdsourcing_data(data_name):
    if data_name == 'cs2010':
        # prepare crowdsourcing data, and document corpus
        raw_crowdsourcing_file = os.path.join(DATA_DIR, 'raw_data/trec10-relevance-feedback/trec-rf10-crowd/trec-rf10-data.txt')
        raw_corpus_path = os.path.join(DATA_DIR, 'raw_data/treccrowd2011-v1.0-20110518')
        crowdsourcing_file = os.path.join(DATA_DIR, data_name, 'crowdsourcing_data.csv')
        document_corpus = os.path.join(DATA_DIR, data_name, 'document_corpus')
        format_cs2010_data(raw_crowdsourcing_file,
                           raw_corpus_path,
                           document_corpus,
                           crowdsourcing_file)

        # prepare topic corpus
        topic_file = os.path.join(DATA_DIR, 'raw_data/trec10-relevance-feedback/trec-rf10-crowd/09.mq.topics.20001-60000')
        topic_corpus_path = os.path.join(DATA_DIR, data_name, 'topic_corpus')
        format_cs2010_2011_topic(crowdsourcing_file,
                                 topic_file,
                                 topic_corpus_path)

        # prepare trec run files
        raw_run_path = os.path.join(DATA_DIR, 'raw_data/runs/Million Query track')
        run_path = os.path.join(DATA_DIR, data_name, 'runs')
        format_run(crowdsourcing_file,
                   raw_run_path,
                   run_path)

    elif data_name == 'cs2011':
        # prepare crowdsourcing data, and document corpus
        raw_crowdsourcing_file = os.path.join(DATA_DIR, 'raw_data/trec11-crowd-source/aggregation-dev/stage2.dev')
        raw_corpus_path = os.path.join(DATA_DIR, 'raw_data/treccrowd2011-v1.0-20110518')
        crowdsourcing_file = os.path.join(DATA_DIR, data_name, 'crowdsourcing_data.csv')
        document_corpus = os.path.join(DATA_DIR, data_name, 'document_corpus')
        format_cs2011_data(raw_crowdsourcing_file,
                           raw_corpus_path,
                           document_corpus,
                           crowdsourcing_file)

        # prepare topic corpus
        topic_file = os.path.join(DATA_DIR, 'raw_data/trec10-relevance-feedback/trec-rf10-crowd/09.mq.topics.20001-60000')
        topic_corpus_path = os.path.join(DATA_DIR, data_name, 'topic_corpus')
        format_cs2010_2011_topic(crowdsourcing_file,
                                 topic_file,
                                 topic_corpus_path)

        # prepare trec run files
        raw_run_path = os.path.join(DATA_DIR, 'raw_data/runs/Million Query track')
        run_path = os.path.join(DATA_DIR, data_name, 'runs')
        format_run(crowdsourcing_file,
                   raw_run_path,
                   run_path)

    return


def pretain_feature_data(data_name):
    feature_path = PATH(data_name, 'feature')
    if not os.path.exists(feature_path):
        os.makedirs(feature_path)

    # system_run_rank_onehot(data_name)
    # system_run_rank_scaler(data_name)
    # clueweb_bert_vector(data_name)
    lexical_feature(data_name)

    return


def split_crowdsourcing_data(data_name):
    csd_file = os.path.join(DATA_DIR, data_name, 'crowdsourcing_data.csv')
    csd_df = pd.read_csv(csd_file, sep=',', names=COLUMNS, header=0, dtype=COLUMN_TYPES)

    task_anno_num_df = csd_df.groupby(by=['topicID', 'docID'], as_index=False).agg({'label': 'count'})
    task_anno_num_df.rename(columns={'label': 'task_anno_num'}, inplace=True)

    worker_anno_num_df = csd_df.groupby(by=['workerID'], as_index=False).agg({'label': 'count'})
    worker_anno_num_df.rename(columns={'label': 'worker_anno_num'}, inplace=True)

    extended_csd_df = pd.merge(csd_df, task_anno_num_df, on=['topicID', 'docID'])
    extended_csd_df = pd.merge(extended_csd_df, worker_anno_num_df, on=['workerID'])

    tn_lst = [0, 5, 1e5]
    an_lst = [0, 5, 1e5]
    for i in range(len(tn_lst)-1):
        for j in range(len(an_lst)-1):
            output_file = os.path.join(DATA_DIR, data_name, 'crowdsourcing_data_t{}_a{}.csv'.format(i, j))
            output_df = extended_csd_df[(extended_csd_df['task_anno_num'] > tn_lst[i])&
                                    (extended_csd_df['task_anno_num'] <= tn_lst[i+1])&
                                    (extended_csd_df['worker_anno_num'] > an_lst[j])&
                                    (extended_csd_df['worker_anno_num'] <= an_lst[j+1])
                                    ]
            output_df = output_df[COLUMNS]
            output_df.to_csv(output_file, index=False)

    return


def prepare_baseline_data(data_name, csd_file='crowdsourcing_data.csv', output_prefix='crowdsourcing_data.yzheng_format.'):
    csd_file = os.path.join(DATA_DIR, data_name, csd_file)
    df = pd.read_csv(csd_file, sep=',', names=COLUMNS, header=0, dtype=COLUMN_TYPES)

    df['question'] = df['topicID'] + df['docID']
    df = df[['question', 'workerID', 'gold', 'label']]
    df = df.sort_values(by=['question', 'workerID', 'gold', 'label'])

    df_answer = df[['question', 'workerID', 'label']]
    df_answer = df_answer.rename(columns={'question': 'question', 'workerID': 'worker', 'label': 'answer'})
    df_answer = df_answer.drop_duplicates()
    df_answer.to_csv(output_prefix+'answer.csv', index=False)

    df_truth = df[['question', 'gold']]
    df_truth = df_truth.rename(columns={'question': 'question', 'gold': 'truth'})
    df_truth = df_truth.drop_duplicates()
    df_truth.to_csv(output_prefix+'truth.csv', index=False, columns=['question', 'truth'])

    return


if __name__ == '__main__':
    # prepare_crowdsourcing_data('cs2010')
    # prepare_crowdsourcing_data('cs2011')

    # pretain_feature_data('cs2010')
    # pretain_feature_data('cs2011')

    split_crowdsourcing_data('cs2010')

    pass