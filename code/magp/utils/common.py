#coding=utf-8

import os
import json
import numpy as np

# directories
PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, os.path.pardir))

CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
RET_DIR = os.path.join(PROJECT_DIR, 'ret')

GOOGLE_WORD2VEC_MODEL_PATH = os.path.join(CODE_DIR, 'external_resources', 'GoogleNews-vectors-negative300.bin')
MACE_JAR_PATH = os.path.join(CODE_DIR, 'external_resources', 'MACE-master')

def PATH(data_name, folder):
    mdir = os.path.join(DATA_DIR, data_name, folder)
    if not os.path.exists(mdir):
        os.makedirs(mdir)
    return mdir

NINF = 1e-30
JITTER = 1e-6
RELEVANT = 1
NON_RELEVANT = 0
NON_LABELLED = -999

GRADED_RELEVANCE_0 = 0
GRADED_RELEVANCE_1 = 1
GRADED_RELEVANCE_2 = 2

# crowd sourcing csv file format
COLUMNS = ['topicID', 'workerID', 'docID', 'gold', 'label']
COLUMN_NUM = len(COLUMNS)
COLUMN_TYPES = {'topicID': str, 'workerID': str, 'docID': str, 'gold': int, 'label': int}

# task feature types, pickle files are features generated in preprocess.py
FEATURE_INFO = {
    # one type
    'lexical': {'feature_files': ['lexical.normalized.pickle']},
    'rank.onehot': {'feature_files': ['rank.onehot.pickle']},
    'clueweb09.bert': {'feature_files': ['clueweb09.bert.pickle']},
    # two types
    'lexical.rank.onehot': {'feature_files': ['lexical.normalized.pickle', 'rank.onehot.pickle']},
    'lexical.clueweb09.bert': {'feature_files': ['lexical.normalized.pickle', 'clueweb09.bert.pickle']},
    'rank.onehot.clueweb09.bert': {'feature_files': ['rank.onehot.pickle', 'clueweb09.bert.pickle']},
    # three types
    'lexical.rank.onehot.clueweb09.bert': {'feature_files': ['lexical.normalized.pickle', 'rank.onehot.pickle', 'clueweb09.bert.pickle']},
}



CS2010_TOPICS = [
"20002",
"20012",
"20014",
"20016",
"20018",
"20022",
"20028",
"20030",
"20034",
"20040",
"20046",
"20004",
"20006",
"20008",
"20010",
"20024",
"20026",
"20032",
"20036",
"20038",
"20042",
"20044",
"20048",
"20050",
"20064",
"20076",
"20102",
"20112",
"20144",
"20152",
"20160",
"20170",
"20176",
"20180",
"20186",
"20196",
"20202",
"20206",
"20214",
"20224",
"20228",
"20232",
"20244",
"20254",
"20270",
"20290",
"20296",
"20298",
"20300",
"20308",
    "20330",
    "20332",
    "20340",
    "20352",
    "20374",
    "20384",
    "20398",
    "20412",
    "20416",
    "20420",
    "20424",
    "20440",
    "20446",
    "20480",
    "20484",
    "20488",
    "20508",
    "20528",
    "20530",
    "20542",
    "20584",
    "20598",
    "20636",
    "20642",
    "20644",
    "20686",
    "20690",
    "20694",
    "20696",
    "20704",
    "20714",
    "20764",
    "20766",
    "20778",
    "20956",
    "20962",
    "20986",
    "20996",
    "20780",
    "20812",
    "20814",
    "20832",
    "20910",
    "20916",
    "20922",
    "20932",
    "20958",
    "20972",
    "20976",
    "20984",
]


CS2011_TOPICS = [
"20002",
"20004",
"20006",
"20008",
"20010",
"20012",
"20014",
"20016",
"20018",
"20022",
"20024",
"20026",
    "20028",
    "20030",
    "20032",
    "20034",
    "20036",
    "20038",
    "20040",
    "20042",
    "20044",
    "20046",
    "20048",
    "20050",
    "20064",
]



def get_file_ids(path):
    """Get all file names in path."""
    file_ids = []
    for root, dirs, files in os.walk(path):
        file_ids.extend(files)
    file_ids = [f for f in file_ids if not f.startswith('.')]

    return file_ids

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def get_list_from_kwargs(**kwargs):
    lst = []
    for k, v in kwargs.items():
        lst.append(k)
        lst.append(v)
    return lst