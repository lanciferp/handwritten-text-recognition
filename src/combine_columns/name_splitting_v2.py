import argparse
import re
import time
import os
from difflib import SequenceMatcher, get_close_matches
import multiprocessing
from typing import Any
import numpy as np
import pandas as pd
from tqdm import tqdm
import unicodedata

DEBUG_MODE = False

ACTUAL_CALC_MULT = 100

RUN_COPY_DOWN = False
BUILD_COMBINED_WHEN_NO_COPY_DOWN = True

NEEDS_GIVEN_NAME = True
NEEDS_SURNAME = False

MAX_SURNAME_LEN = None



NON_WORD_CHARS = r'[^a-zA-ZÀ-ÿ \-\'\.\,]'

VERSION = 'forced_surname_4'


PATH_TO_NAME_DATA = './data/1880_2022_Given_Surname_Selection.tsv'
NAME_DATA_COL = 'name' #'surname'
NAME_DATA_RATIO_COL = 'surname_to_given_normalized' #'ratio'

NAME_DATA_SEP = '\t'
NAME_DATA_WORD_LEN_COL = 'word_len'
NAME_SEQ_MATCHER_COL = 'sequence_matcher'

SOURCE_DATASET = './data/Texas Divorce Husbands Names with 3 or more Spaces in Name.tsv'
SOURCE_SEP = '\t' #','

OUTPUT_DIR = './results/'
# OUTPUT_SEP = ','


MAX_INITIAL_CNT = 2

BLANK_COPY_DOWN_SIZE = 4 # Triggers if blanks in a row is greater than or equal

NON_INTERNAL_NAME_CHARS = r'[^A-Za-z]'
INTERNAL_TO_LOWER = True

def strip_accents(text):
    """
    Strip accents from input String.

    :param text: The input string.
    :type text: String.

    :returns: The processed String.
    :rtype: String.
    """
    try:
        text = unicode(text, 'utf-8')
    except (TypeError, NameError): # unicode is a default on python 3 
        pass
    text = unicodedata.normalize('NFD', text)
    text = text.encode('ascii', 'ignore')
    text = text.decode("utf-8")
    return str(text)


def internal_name_cleaning(names):
    def name_cleaner(name):
        name = strip_accents(name)
        name = re.sub(NON_INTERNAL_NAME_CHARS, '', name)
        if INTERNAL_TO_LOWER:
            name = name.lower()
        return name

    ret = None
    if isinstance(names, str):
        ret = name_cleaner(names)
    else:
        ret = [name_cleaner(name) for name in names]
    return ret

# print(internal_name_cleaning('Montréal, über, 12.89, Mère, Françoise, noël, 889'))

INITIAL_OTHERS = ['sr', 'jr', 'mr', 'mrs', 'ms', 'dr', 'rev']

# Types to classify names
TYPES = {
    'non_name': 'non_name',
    'given': 'given_name',
    'full': 'full_name',
    'unclear': 'unclear',
    'blank': 'blank'
}

FULL_NAME_MODEL_TYPES = {
    '<BLANK>' : TYPES['blank'],
    '<nln>' : TYPES['full'],
    '<san>' : TYPES['full'], # edgecase (2 examples, both were full)
    '<sab>' : TYPES['given'], # (partial name)
    '<sln>' : TYPES['given'] # (partial name) edgecase (2 examples, 1 was full, 1 was partial)
}

FULL_NAME_MODEL_CONF_THRESH = 0.6

# List of non-name words
NON_NAMES = [
    'blank', 'is', 'vacant', 'vecant', 'vacent', 'vocont', 'vacont', 'becant', 'bacent', 'zacant', 'hot', 'me',
    'for', 'ne', 'to', 'homes', 'noone', 'none', 'nonehome', 'name', 'see', 'sheet',
    'nobat', 'nothome', 'nokat', 'noxat', 'lome', 'nome', 'noat',
    'shut', 'line', 'no', 'one', 'at', 'horn', 'home', 'house', 'not', 'has',
    'husband', 'head', 'wife', 'son', 'daughter', 'mother', 'father', 'inlaw', 'notathome', 'vacont', 'nonviandenta', 'moxhome',
    'attone', 'nobt', 'norone', 'notat', 'noneat', 'nont', 'nontat', 'athome', 'notot', 'hame', 'hane', 'hone', 'natot', 'noneat'
    'jrane', 'mrone', 'oat', 'noncar', 'ihome', 'mconeat', 'alone', 'nrame', 'komie', 'arhrome', 'nox', 'alome', # last row are more questionable choices
]
NON_NAMES = {word.lower() for word in NON_NAMES} #[]

MIDDLE_SURNAME_WORDS = [
    'el', 'al', 'la', 'da', 'de', 'di', 'do', 'dos', 'das', 'del', 'san', 'bin', 'ben',  'of', # 'santa' 'santo', 'saint',
    'd', 'des', 'della', 'dela'#, ''
]

# Proportional and minimum length settings for non-names
NON_NAMES_PROPORTION = 0.5
NON_NAMES_MIN_LEN = 1

# Ratio Settings
TOTAL_LOWER_BOUND = 50
MAX_WORD_LEN_DIFF = 1
N_CLOSEST = 3

REMOVE_MULTI_WORD_NAMES = True
TYPING_ON_VALID_WORDS = True
CALC_IF_FOUND = False # True
SURNAME_EXPECTED_AT_START = True
REPLACE_NON_NAMES_WITH_EMPTY = False



# Confidence and ratio thresholds for surnames
# CONF_THRESHOLD_FOR_SURNAME = [0.9, 0.8, 0.7]
# RATIO_THRESHOLD_FOR_SURNAME = [0.6, 0.7, 0.8]

CONF_THRESHOLD_FOR_SURNAME = [0.9999, 0.9, 0.8, 0.7]
RATIO_THRESHOLD_FOR_SURNAME = [0.5001, 0.6, 0.7, 0.8]
RATIO_THRESHOLD_FOR_GIVEN_NAME = [1 - val for val in RATIO_THRESHOLD_FOR_SURNAME]
KEEP_CUT_OFF_COUNT = 2000
GIVEN_NAME_OVERRIDE_THRESH = 0.8


def non_name_classifier(text, non_names=NON_NAMES, non_names_proportion=NON_NAMES_PROPORTION, non_names_min_len=NON_NAMES_MIN_LEN):
    """
    Classify if an inout text is a non-name
    True = non-name
    False = a name
    """
    # If the text is empty
    if text is None:
        return True
    else:
        if isinstance(text, list):
            text = ' '.join(text)
        if not isinstance(text, str) or len(text) == 0:
            return True
        
        matches = 0
        all_words = text.strip().lower().split()
        matches = sum([word in non_names for word in all_words])
        return matches >= non_names_min_len and matches / len(all_words) >= non_names_proportion
    
# def clean_names(names, non_word_chars=NON_WORD_CHARS, to_title=False): #True):
def clean_names(names): #True):
    """
    Normalize names to a standard format
    Can either take in a single name or a list of names
    Will return in the same format as the input
    """
    def modify_name(name):
        # name = re.sub(non_word_chars, '', name)
        # name = re.sub(' *\. *', ' ', name)
        pts = name.strip().split()
        ret_pts = []
        for pt in pts:
            should_keep = True
            # Conditions to remove a name:
            modded_len = len(re.sub(NON_WORD_CHARS, '', pt))
            if (modded_len == 0) or pt == '' or pt is None:
                
                should_keep = False
                if len(pt) == 1 and pt.isnumeric(): # single number could be a suffix (ie. 2nd but without the 'nd') 
                    should_keep = True
            if should_keep:
                ret_pts.append(pt)
        return ' '.join(ret_pts)
    
    if isinstance(names, str):
        return modify_name(names)
    else:     

        return [modify_name(name) for name in names]
    

def load_and_preprocess_data(path, sep):    
    """
    Load and prepare the data to be modified
    """
    df = pd.read_csv(path, sep=sep)
    
    # df = df[:10]
    print('\tshape:', df.shape)
    print('\tcolumns:', df.columns)
    
    df['name'] = df['name'].ffill()
    df['name'] = df['name'].fillna('')
    df['name_cleaned'] = df['name'].str.replace('<BORDERLINE>', '')
    df['name_cleaned'] = df['name_cleaned'].apply(clean_names)

    if 'filename' in df.columns:
        df['state'] = df['filename'].str.extract(r'[0-9]+-([A-Za-z]+)-[0-9]+-[0-9]+_box_06_row_[0-9]+.jpg', expand=True) #str.findall(r'-([A-Za-z]+)-').str[0]
        df['image_id'] = df['filename'].str.extract(r'[0-9]+-[A-Za-z]+-[0-9]+-([0-9]+)_box_06_row_[0-9]+.jpg', expand=True) #str.findall(r'-([A-Za-z]+)-').str[0]
        df['row'] = df['filename'].str.extract(r'[0-9]+-[A-Za-z]+-[0-9]+-[0-9]+_box_06_row_([0-9]+).jpg', expand=True) #str.findall(r'-([A-Za-z]+)-').str[0]
    return df

def load_name_data(path=PATH_TO_NAME_DATA, sep=NAME_DATA_SEP):
    df = pd.read_csv(path, sep=sep)
    # df['surname'] = df['surname'].str.lower()
    df[NAME_DATA_COL] = df[NAME_DATA_COL].apply(internal_name_cleaning)
    if NAME_DATA_WORD_LEN_COL not in df.columns:
        # df[NAME_DATA_WORD_LEN_COL] = df[NAME_DATA_COL].apply(lambda x: len(x)) # x.split()
        df[NAME_DATA_WORD_LEN_COL] = df[NAME_DATA_COL].str.len()
    df = df.sort_values(NAME_DATA_WORD_LEN_COL, ascending=False)

    df[NAME_SEQ_MATCHER_COL] = [SequenceMatcher(None, '', x) for x in df[NAME_DATA_COL]]   
    return df


class Name_Ratio():
    """
    Class to store the confidence and ratio of a name
    (so that we dont recalculate it every time)
    """
    def __init__(self, name, confidence=None, ratio=None):
        self.name = name
        self.confidence = confidence
        self.ratio = ratio

    def __call__(self):
        return self.confidence, self.ratio
    
class Ratios():
    """
    Class to store the confidence and ratio of a name
    Will also remove the least used names if clean_up() is called
    """
    def __init__(self, df_ratios, names=[],
                 cut_off=KEEP_CUT_OFF_COUNT,
                 calc_if_found=CALC_IF_FOUND, max_word_len=MAX_WORD_LEN_DIFF, n_closest=N_CLOSEST):                        
        self.df_ratios = df_ratios
        # self.df_unique_list = set(self.df_ratios[NAME_DATA_COL].unique().tolist())
        self.df_inds = {name:index for index, name in zip(self.df_ratios.index, self.df_ratios[NAME_DATA_COL].tolist())}
        self.names = dict()
        self.counts = dict()
        self.cutoff_count = cut_off
        self.calc_if_found = calc_if_found
        self.max_word_len = max_word_len
        self.n_closest = n_closest
        if DEBUG_MODE:
            print('focus window:', self.n_closest * ACTUAL_CALC_MULT)
        
        self.sel_cutoff = 0.001
        self.min_len = 2
        
        self.selections = dict()
        upper_lim = self.df_ratios[NAME_DATA_WORD_LEN_COL].max()
        for i in range(1, upper_lim):
            upper = min(i + self.max_word_len, upper_lim)
            lower = max(i - self.max_word_len, 0)
            
            self.selections[i] = self.df_ratios[(self.df_ratios[NAME_DATA_WORD_LEN_COL] >= lower) & (self.df_ratios[NAME_DATA_WORD_LEN_COL] <= upper)]
        
        if not isinstance(names, list):
            names = [names]
            
        for name in names:
            self._add_name(name, 1)
        
        # if DEBUG_MODE:
        #     test_names = ['Jose', 'Josep', 'Joseph']
        #     for test_name in test_names:
        #         self._add_name(test_name, 1)
        #         print(self.names[test_name].name, self.names[test_name].ratio, self.names[test_name].confidence)
        
    # Returns the confidence and ratio of a name
    #  can take a string or a list of strings
    # @profile
    def __call__(self, names):
        should_ret_str = isinstance(names, str)
        if should_ret_str:
            names = [names]

        res_conf, res_ratio = [], []
        names = [internal_name_cleaning(name) for name in names]
        
        for name in names:
            conf, ratio = 0.0, 0.0
            if len(name) >= self.min_len:
                
                # if name not in self.names:
                self._add_name(name, 0)
                self.counts[name] += 1
                conf, ratio = self.names[name]() # retrieves the confidence and ratio
            
            res_conf.append(conf)
            res_ratio.append(ratio)
        if should_ret_str:
            return res_conf[0], res_ratio[0]

        return res_conf, res_ratio
    
    # Calculate the confidence and ratio of a name based on averaging the most similar known names
    # @profile
    def process_name(self, name):
        n_closest_inds = None
        n_closest_dists = None
        n_closest_ratios = None
        n_closest_names = None
        
        df_ratios = self.df_ratios
        
        # if name in self.df_unique_list and not self.calc_if_found:
        if name in self.df_inds and not self.calc_if_found:
        
            n_closest_dists = [1.0]
            n_closest_ratios = [self.df_ratios.at[self.df_inds[name], NAME_DATA_RATIO_COL]]

            
        # Do we want to recalculate this even if it is already calculated?  The results could change as other values get added,
        #    but that means that a previous version of itself could influence current versions.
        #    Though this probably isnt an issue since this is function is called in _add_name, which checks if it is already in the list
        else:
            sel_ind = len(name)
            if sel_ind not in self.selections:
                # print('len name:', len(name))
                # print('sel:', list(self.selections.keys()))
                sel_ind = min(list(self.selections.keys()), key=lambda i: abs(i-sel_ind))
                # print(sel_ind)
            df_sel = self.selections[sel_ind]
            
            matchers = df_sel[NAME_SEQ_MATCHER_COL].to_numpy()
            _ = [f.set_seq1(name) for f in matchers]

            edit_dist = [f.quick_ratio() for f in matchers]
            n_closest_inds = np.argsort(edit_dist)[-self.n_closest * ACTUAL_CALC_MULT:]
            # print(edit_dist[n_closest_inds[0]], edit_dist[n_closest_inds[-1]])
            
            # print(-len(edit_dist) // ACTUAL_CALC_DENOM)
            # print(-self.n_closest * ACTUAL_CALC_MULT)
            
            edit_dist = [matchers[i].ratio() for i in n_closest_inds]
            
            n_closest_inds = np.argsort(edit_dist)
            n_closest_inds = n_closest_inds[-self.n_closest:] if len(n_closest_inds) > self.n_closest else n_closest_inds
            
            n_closest_dists = [edit_dist[i] for i in n_closest_inds]
            n_closest_ratios = df_sel.iloc[n_closest_inds][NAME_DATA_RATIO_COL].to_numpy()

        avg_confidence = np.average(n_closest_dists)
        avg_ratio = np.average(n_closest_ratios) if sum(n_closest_dists) == 0 else np.average(n_closest_ratios, weights=n_closest_dists)
        # print(name, avg_confidence, n_closest_dists, avg_ratio, n_closest_ratios, n_closest_names)
        return avg_confidence, avg_ratio
    
    # Used to update the sequence matcher
    @staticmethod
    def matcher_update(matcher, name):
        matcher.set_seq1(name)
        return matcher.ratio()

    def _add_name(self, name, starting_count=1):
        name = internal_name_cleaning(name)
        if name not in self.names:
            conf, ratio = self.process_name(name)
            self.names[name] = Name_Ratio(name, conf, ratio)
            self.counts[name] = starting_count
            # print(self.names)

    # Frees up space by removing the names that are not as common
    def clean_up(self):
        if self.cutoff_count < len(self.names):
            to_keep = sorted(self.counts.values(), reverse=True)[:self.cutoff_count]
            min_to_keep = to_keep[-1]
            names_new = dict()
            kept_cnt = 0
            for name in self.names:
                if self.counts[name] >= min_to_keep:
                    names_new[name] = self.names[name]
                    
                    kept_cnt += 1
                if kept_cnt >= self.cutoff_count:
                    break
            self.names = names_new
        self.counts = {name: 1 for name in self.names}
        
def is_surname(confs, ratios, conf_thresh=CONF_THRESHOLD_FOR_SURNAME, ratio_thresh=RATIO_THRESHOLD_FOR_SURNAME, greater_than_ratio=True):
    """
    Determin if each word is likely to be a surname based on the confidence and ratio
    conf_thresh, ratio_thresh are lists of thresholds.  If a confidence is above a certain threshold, 
    then the ratio must be above a certain threshold, determined by the value in the associated index.
    This should probably be changed to be continuous, instead of checkpoints, but I was able to code it up quickly this way.
    Alternatively, other thresholds could be tested instead.
    ie. if conf_thresh = [0.9, 0.8] and ratio_thresh = [0.7, 0.8], and the confidence is 0.89, 
    then the ratio must be above 0.8 to be considered a surname
    """
    if not isinstance(confs, list):
        confs = [confs]
    if not isinstance(ratios, list):
        ratios = [ratios]
    res = [False for _ in confs]
    for i, (conf, ratio) in enumerate(zip(confs, ratios)):
        j = 0
        while j < len(conf_thresh) and conf <= conf_thresh[j]:
            j += 1
            
        word_res = j < len(conf_thresh)
        if greater_than_ratio:
            word_res = word_res and (ratio >= ratio_thresh[j])
        else:
            word_res = word_res and (ratio <= ratio_thresh[j])
        res[i] = word_res        
    return res
    
# @profile 
def process_name(name, full_name_type, type_conf, ratio_class, sur_expected_at_start=SURNAME_EXPECTED_AT_START):
    n_type = TYPES['unclear']
    given = name
    surname = ''
    surname_known = False

    words = name.split()
    initials = [word for word in words if len(word) == 1]
    initials.extend([word for word in words if word in INITIAL_OTHERS])
    non_initials = [word for word in words if word not in initials]
    # non_initials = [word for word in words if len(word) > 1]
    aggreement_type = None
    ratio_levels = None
    confidences = None

    # If it is believed to be a blank then set it to 'non_name'
    if (len(words) == 0) or (len(words) == 1 and words[0].lower() == 'blank') or full_name_type == TYPES['blank']:
        n_type = TYPES['non_name']
        given = ''
    # if either one name or one name and one initial, then assume it is a given name
    elif (len(words) == 1  or (len(words) == MAX_INITIAL_CNT and len(non_initials) == 1)) and (full_name_type == TYPES['given'] or full_name_type == TYPES['unclear']):
        n_type = TYPES['given']
    # cases when the full name model outputs a full name but word len is 2 with a initial
    elif (len(words) == 2  and (len(words) == MAX_INITIAL_CNT and len(non_initials) == 1)) and (full_name_type == TYPES['full']):
        n_type = TYPES['given']
    # if the full_name_type is a given name and above a threshold, assume it is a given name
    elif full_name_type == TYPES['given'] and type_conf >= FULL_NAME_MODEL_CONF_THRESH:
        n_type = TYPES['given']

    # otherwise, run the ratio and surname detection
    else:
        confs, ratios = ratio_class(words) #concat_dir
        confidences = confs
        ratio_levels = ratios
        last_name_results = is_surname(confs, ratios)
        original_results = last_name_results.copy()
        dir = {'pos':0, 'dir':1, 'max':len(words)} # expected direction of surname
        opposite = {'dir':-1, 'pos':-1, 'max':-len(words)}
        
        class movement:
            def __init__(self, pos, dir, max):
                self.pos = pos
                self.dir = dir
                self.max = max
                
            def __call__(self, *args: Any, **kwds: Any) -> Any:
                return self.pos
            
            def step(self):
                self.pos += self.dir
                return self.pos

            def can_step(self, is_next=False):
                # print('pos', self.pos, 'max', self.max, 'dir', self.dir, 'is_next', is_next)
                return self.pos != self.max and (not is_next or (self.pos + self.dir) != self.max)
        
        # is_surname_ls = [0 for _ in words]
        sur_dir = movement(0, 1, len(words))
        given_dir = movement(len(words)-1, -1, 0)
        
        if not sur_expected_at_start:
            sur_dir, given_dir = given_dir, sur_dir

        # print('word', words)
        # print('conf', confs)
        # print('rats', ratios)   
        # print('init', last_name_results)
        
        # TODO This might cause issues
        if NEEDS_SURNAME:
            given_dir.max -= given_dir.dir
            
        # Setting at least a value to be be the surname if the full name model says it is a full name and the confidence is above the threshold
        # Thus, if the first name is not believed to be a surname, but the second is, then both will be considered surnames
        # NOTE: This could be worse than better
        if NEEDS_SURNAME or (full_name_type == TYPES['full'] and type_conf >= FULL_NAME_MODEL_CONF_THRESH):
            # NOTE: If dont want single letter surnames would need to add some acception so that it continues
            last_name_results[sur_dir()] = True
        # print('needs surname', last_name_results)    
        
        surname_len = 0
        # # If find a name thats not alowed to be the end of a surname, set the next to True
        while sur_dir.can_step() and last_name_results[sur_dir()]: # not
            # print('ind', sur_dir())
            if words[sur_dir()].lower() in MIDDLE_SURNAME_WORDS and sur_dir.can_step(True):
                last_name_results[sur_dir() + 1] = True
            
            if words[sur_dir()].lower() in non_initials:
                surname_len += 1
            sur_dir.step()
        # print('updated_surname', last_name_results)    
        # print(last_name_results)
        
        # Set remainder to false
        # print(f'filling given from {sur_dir()} to {sur_dir.max} in dir {sur_dir.dir}')
        for i in range(sur_dir(), sur_dir.max, sur_dir.dir):
            # print(i)
            last_name_results[i] = False
        
        # print('filled given', last_name_results)
        
        if NEEDS_GIVEN_NAME:
            # print(initials)
            # print('next_can_step', given_dir.can_step(True))
            while given_dir.can_step() and (words[given_dir()] in initials or (given_dir.can_step(True) and words[given_dir() + given_dir.dir].lower() in MIDDLE_SURNAME_WORDS)):
                last_name_results[given_dir()] = False
                given_dir.step()
                # print('next_can_step', given_dir.can_step(True))
            if given_dir.can_step() or not original_results[given_dir()] or ratio_levels[given_dir()] <= GIVEN_NAME_OVERRIDE_THRESH:
                last_name_results[given_dir()] = False

        surname = ' '.join([words[i] for i in range(len(words)) if last_name_results[i]])
        surname_known = sum(last_name_results) > 0
        given = ' '.join([words[i] for i in range(len(words)) if not last_name_results[i]])
        # print('surname', surname, 'given', given)
        
        # NOTE: alternatively, if the number of words total is >= 3, then we can assume that it is "full"
        #  By this logic, we would also say that words of length 1 are "given" names
        #  And words of length 2 are "uncertain"
        if surname_known:
            n_type = TYPES['full']
        
        if n_type == TYPES['unclear']:
            
            given_name_results = is_surname(confs, ratios, ratio_thresh=RATIO_THRESHOLD_FOR_GIVEN_NAME, greater_than_ratio=False)
            concensus = True
            for word, res in zip(words, given_name_results):
                if word not in initials and not res:
                    # if non initials say that it is a givenname and the full name model says it is a full name, then it is concidered as a concensus
                    concensus = sum(given_name_results) > len(initials) and full_name_type == TYPES['given']
                    break
            if concensus:
                n_type = TYPES['given']
                if len(initials) > 0:
                    middle_initial = True
        
        aggreement_type = n_type == full_name_type if full_name_type != TYPES['unclear'] else None
        
    return n_type, surname, given, surname_known, ratio_levels, confidences, aggreement_type


def process_names(data_to_process, ratio_class):
    """
    NOTE: It is expected that names passed into this function are already preprocessed/normalized
    """
    is_non_names = None
    names = data_to_process['name_cleaned']
    full_name_types = []
    type_confidences = []
    if 'full_name_model' in data_to_process.columns:
        full_name_types = data_to_process['full_name_model']
        full_name_types = [FULL_NAME_MODEL_TYPES[typ] if typ in FULL_NAME_MODEL_TYPES else TYPES['unclear'] for typ in full_name_types]
    else:
        full_name_types = [TYPES['unclear'] for _ in names]
    
    if 'full_name_model_confidence' in data_to_process.columns:
        type_confidences = data_to_process['full_name_model_confidence']
    else:
        type_confidences = [0.0 for _ in names]

    should_ret_str = isinstance(names, str)
    if should_ret_str:
        names = [names]
        if is_non_names is not None:
            is_non_names = [is_non_names]
    if is_non_names is None:
        is_non_names = [non_name_classifier(name) for name in names]    
    
    raw_results = []
    name_types = []
    surnames = []
    given_names = []
    surname_known = []
    ratio_levels = []
    conf_levels = []
    aggreement_type = []
    names_len = len(names)
    for name, f_name_type, type_conf, is_non_name  in tqdm(zip(names, full_name_types, type_confidences, is_non_names), total=names_len):
        if name is None:
            name = ''
        if is_non_name or f_name_type == TYPES['blank'] or len(name) == 0: 
            name = '' if REPLACE_NON_NAMES_WITH_EMPTY else name
            name_types.append(TYPES['non_name'])
            surnames.append(None) 
            given_names.append(None) 
            surname_known.append(False)
            ratio_levels.append(None)
            conf_levels.append(None)
            aggreement_type.append(f_name_type == TYPES['blank']) 
            
        else:
            n_type, surname, given, is_sur_known, ratio_lvls, conf_lvls, agg_type  = process_name(name, f_name_type, type_conf, ratio_class) 
            
            name_types.append(n_type)
            surnames.append(surname)
            given_names.append(given)
            surname_known.append(is_sur_known)
            ratio_levels.append(ratio_lvls)
            conf_levels.append(conf_lvls)
            aggreement_type.append(agg_type)
            
        raw_results.append(name)

    data_to_process['name_type'] = name_types
    data_to_process['surname'] = surnames
    data_to_process['given_name'] = given_names
    data_to_process['surname_known'] = surname_known
    data_to_process['name_raw'] = raw_results
    data_to_process['is_non_name'] = is_non_names
    data_to_process['ratio_levels'] = ratio_levels
    data_to_process['conf_levels'] = conf_levels
    data_to_process['aggreement_type'] = aggreement_type
    # print(len(data_to_process))
    return data_to_process

def update_surname(df):
    df['surname'].fillna(method='ffill', inplace=True) # fill next with previous
    df.loc[(df['name_type'] == 'non_name') | (df['name_type'] == 'unclear') | (df['surname_mask']), 'surname'] = ''
     
    return df

def copy_down(df_orig):
    df_orig = df_orig.sort_values(by=['filename', 'image_id', 'row'])
    
    # Find locations that the surname should not be copied down (this triggers if ther are more than BLANK_COPY_DOWN_SIZE values in a row between names)
    df_orig['surname_mask'] = df_orig['is_non_name'].eq(True).rolling(BLANK_COPY_DOWN_SIZE).sum().ge(BLANK_COPY_DOWN_SIZE)
    df_orig['surname_mask'] = df_orig['surname_mask'].replace(False, None)
    df_orig.loc[df_orig['surname_known'], 'surname_mask'] = False
    df_orig['surname_mask'] = df_orig['surname_mask'].ffill()
    df_orig.loc[df_orig['is_non_name'], 'mask'] = False
    
    df = df_orig.loc[~df_orig['is_non_name']]
    inds = df.index
    df_sel = df.copy()
  
    df_sel['surname'] = df_sel['surname'].ffill()
    df_sel['surname'] = df_sel['surname'].fillna('')
    
    df_sel['given_name'] = df_sel['given_name'].fillna('')
    df_sel['surname'] = df_sel['surname'].replace('', np.nan)
    
    df_sel = update_surname(df_sel) # household names

    df_sel['combined_name'] = df_sel['given_name'] + ' ' + df_sel['surname']    
    df_sel['surname'] = df_sel['surname'].where((df_sel['given_name'].ne('')), '')
    df_sel['combined_name'] = df_sel['combined_name'].where((df_sel['given_name'].ne('')), '')

    df_orig.loc[inds, 'given_name'] = df_sel['given_name']
    df_orig.loc[inds, 'surname'] = df_sel['surname']
    df_orig.loc[inds, 'combined_name'] = df_sel['combined_name']
    df_orig.drop(columns=['surname_mask'], inplace=True)
    return df_orig


def parallel_processing(df, ratio_class, num_cores=6):
    #num_cores = multiprocessing.cpu_count()
    # Create a pool of worker processes
    pool = multiprocessing.Pool(num_cores)
    # Split the DataFrame into chunks based on the number of CPU cores
    chunks = np.array_split(df, num_cores)
    
    # Process each chunk in parallel
    # results = pool.map(process_names, [(chunks, ratio_class) for chunks in chunks])
    results = pool.starmap(process_names, [(chunks, ratio_class) for chunks in chunks])
    
    # Concatenate the processed chunks back into a single DataFrame
    processed_df = pd.concat(results)
    # Close the pool of worker processes
    pool.close()
    return processed_df

def bool_decode(s):
    return not s.lower() in ['false', 'f', 'no', 'n', '0']

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p','--distrib', type=bool_decode, default=True, help='If True, the dataframe with be splitted in n parts and processd in parallel')
    parser.add_argument('-n', '--cores', type=int, default=6, help='Number of cores for multiprocessing')
    parser.add_argument('-d', '--dataset', type=str, default=SOURCE_DATASET, help='Input dataset to be processed')
    parser.add_argument('--sep', type=str, default=SOURCE_SEP)
    parser.add_argument('--version', type=str, default=VERSION)
    
    args = parser.parse_args()

    distrib = args.distrib
    distrib = False if DEBUG_MODE else distrib
    n = args.cores

    PATH_TO_MODIFY = args.dataset
    MODIFY_SEP = args.sep
    # OUTPUT_PATH =  "results/Results_" + PATH_TO_MODIFY.split('/')[-1] #'./1950_Name_sample_results.csv'
    file_name = PATH_TO_MODIFY.split('/')[-1].split('.', 1)[0]
    version = args.version if args.version is not None and len(args.version) > 0 else None
    extension = PATH_TO_MODIFY.split('/')[-1].split('.', 1)[1]
    file_name = f'Results_{file_name}_{version}.{extension}' if version is not None else f'Results_{file_name}.{extension}'
    OUTPUT_PATH = os.path.join(OUTPUT_DIR, file_name)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f'Processing {PATH_TO_MODIFY} with {NAME_DATA_SEP} seperator using {n} processes/cores')
    print(f'Output file: {OUTPUT_PATH}')
    
    
    tic = time.time()
    progress = tqdm(total=4, position=0, leave=True, desc='Loading data to process')
    progress = tqdm(total=2, position=0, leave=False, desc='Preprocessing data')
    data_to_process = load_and_preprocess_data(PATH_TO_MODIFY, MODIFY_SEP)
    if DEBUG_MODE:
        # data_to_process = data_to_process[:10] if DEBUG_MODE else data_to_process
        print(f'Debug mode is on, df subset of shape {data_to_process.shape} selected')
    progress.update(1)
    
    df_ratios = load_name_data()
    ratio_class = Ratios(df_ratios)    

    progress.update(1)
    if distrib:
        data_to_process = parallel_processing(data_to_process, ratio_class)
    else:
        data_to_process = process_names(data_to_process, ratio_class)
    
    progress.update(1)
    progress.set_description('Copying down names')    
    if RUN_COPY_DOWN:
        data_to_process = copy_down(data_to_process)
    elif BUILD_COMBINED_WHEN_NO_COPY_DOWN:
        data_to_process['combined_name'] = data_to_process['given_name'] + ' ' + data_to_process['surname'] 
        data_to_process['combined_name'] = data_to_process['combined_name'].where((data_to_process['given_name'].ne('')), '')
    
    progress.update(1)
    progress.set_description('Saving data')
    data_to_process.to_csv(OUTPUT_PATH, sep=SOURCE_SEP, index=False)
    
    toc = time.time()
    print(f'Runtime: {toc-tic:.2f} seconds or {(toc-tic)/60:.2f} minutes')
    
