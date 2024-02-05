import argparse
import re
import time
import os
from difflib import SequenceMatcher, get_close_matches
import multiprocessing
import numpy as np
import pandas as pd
from tqdm import tqdm

# Name file path
PATH_TO_NAME_DATA = './data/1950_name_data.tsv'
NAME_DATA_SEP = '\t'
# SOURCE_DATASET = './data/1950_with_full_name_type.csv' #'./data/1950_Name_sample.csv'
SOURCE_DATASET = './data/1950_Name_sample_4_w_last_name.csv'
SOURCE_SEP = ','

VERSION = 'Test_1'

OUTPUT_DIR = './results/'
OUTPUT_SEP = ','

DEBUG_MODE = False

MAX_INITIAL_CNT = 2

BLANK_COPY_DOWN_SIZE = 4 # Triggers if blanks in a row is greater than or equal

INITIAL_OTHERS = ['Sr', 'Jr']

PROCESS_NAME_COLUMN = 'name_cleaned'

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

# Regular expression to remove non-alphabetical characters during normalization
NON_WORD_CHARS = r'[^a-zA-Z ]'

# List of non-name words
NON_NAMES = [
    'blank', 'is', 'vacant', 'vecant', 'vacent', 'becant', 'bacent', 'zacant', 'hot', 'me',
    'for', 'ne', 'to', 'homes', 'noone', 'none', 'nonehome', 'name', 'see', 'sheet', 'nome',
    'nobat', 'nothome', 'notathome', 'nooneathome', 'nokat', 'noxat', 'lome', 'nome', 'noat', 'nox', 'noxat', 'alome',
    'shut', 'line', 'no', 'one', 'at', 'horn', 'home', 'house', 'not', 'has',
    'husband', 'head', 'wife', 'son', 'daughter', 'mother', 'father', 'inlaw'
]
NON_NAMES = [word.lower() for word in NON_NAMES]

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
MAX_SURNAME_LEN = None
CAN_BE_ALL_SURNAME = False

# Confidence and ratio thresholds for surnames
CONF_THRESHOLD_FOR_SURNAME = [0.9, 0.8, 0.7]
RATIO_THRESHOLD_FOR_SURNAME = [0.6, 0.7, 0.8]
RATIO_THRESHOLD_FOR_GIVEN_NAME = [1 - val for val in RATIO_THRESHOLD_FOR_SURNAME]
KEEP_CUT_OFF_COUNT = 2000


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
        for word in all_words:
            if word in non_names:
                matches += 1
        return matches >= non_names_min_len and matches / len(all_words) >= non_names_proportion

def clean_names(names, non_word_chars=NON_WORD_CHARS, to_title=True):
    """
    Normalize names to a standard format
    Can either take in a single name or a list of names
    Will return in the same format as the input
    """
    should_ret_str = isinstance(names, str)
    if should_ret_str:
        names = [names]
    ret = []
    for name in names:
        name = re.sub(' *\. *', ' ', name)
        name = re.sub(non_word_chars, '', name)
        pts = name.strip().lower().split()
        
        # Capitalize the first letter of each word (note: does not produce the same result as .title())
        if to_title: 
            pts = [pt.capitalize() for pt in pts if pt is not None and pt != '']
        ret.append(' '.join(pts))
    return ret[0] if should_ret_str else ret

def load_and_preprocess_data(path, sep):    
    """
    Load and prepare the data to be modified
    """
    df = pd.read_csv(path, sep=sep)
    df['name'] = df['name'].ffill()
    df['name'] = df['name'].fillna('')
    df['name_cleaned'] = df['name'].str.replace('<BORDERLINE>', '')
    df['name_cleaned'] = df['name_cleaned'].apply(clean_names)

    df['state'] = df['filename'].str.extract(r'[0-9]+-([A-Za-z]+)-[0-9]+-[0-9]+_box_06_row_[0-9]+.jpg', expand=True) #str.findall(r'-([A-Za-z]+)-').str[0]
    df['image_id'] = df['filename'].str.extract(r'[0-9]+-[A-Za-z]+-[0-9]+-([0-9]+)_box_06_row_[0-9]+.jpg', expand=True) #str.findall(r'-([A-Za-z]+)-').str[0]
    df['row'] = df['filename'].str.extract(r'[0-9]+-[A-Za-z]+-[0-9]+-[0-9]+_box_06_row_([0-9]+).jpg', expand=True) #str.findall(r'-([A-Za-z]+)-').str[0]
    return df

def load_name_data(path=PATH_TO_NAME_DATA, sep=NAME_DATA_SEP):
    df = pd.read_csv(path, sep=sep)
    # df['surname'] = df['surname'].str.lower()
    df['sequence_matcher'] = df['surname'].apply(lambda x: SequenceMatcher(None, '', x))    
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
        # self.count = 0
        
    def __call__(self):
        # self.count += 1
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
        self.df_unique_list = self.df_ratios['surname'].unique().tolist()
        self.names = dict()
        self.counts = dict()
        self.cutoff_count = cut_off
        self.calc_if_found = calc_if_found
        self.max_word_len = max_word_len
        self.n_closest = n_closest
        self.sel_cutoff = 0.001
        self.min_len = 2
        
        
        
        if not isinstance(names, list):
            names = [names]

        # names.extend(self.df_unique_list)
        
        # print(names)
        
        for name in names:
            self._add_name(name, 1)
        
        if DEBUG_MODE:
            test_names = ['Jose', 'Josep', 'Joseph']
            for test_name in test_names:
                self._add_name(test_name, 1)
                print(self.names[test_name].name, self.names[test_name].ratio, self.names[test_name].confidence)
        
        #     if name == test_name:
        #         print(test_name)
        #         print(self.names[test_name])
        # print(self.names['failedswe'])
                
        # print(self.names[test_name]['name'], self.names[test_name]['confidence'], self.names[test_name]['ratio'])
    
    # Returns the confidence and ratio of a name
    #  can take a string or a list of strings
    def __call__(self, names):
        should_ret_str = isinstance(names, str)
        if should_ret_str:
            names = [names]

        res_conf, res_ratio = [], []
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
    @profile
    def process_name(self, name):
        n_closest_inds = None
        n_closest_dists = None
        n_closest_ratios = None
        n_closest_names = None
        
        if name in self.df_unique_list and not self.calc_if_found:
            n_closest_dists = [1.0]
            n_closest_ratios = self.df_ratios[self.df_ratios['surname'] == name]['ratio'].tolist()
            n_closest_ratios = [n_closest_ratios[0]]
            
            
        # Do we want to recalculate this even if it is already calculated?  The results could change as other values get added,
        #    but that means that a previous version of itself could influence current versions.
        #    Though this probably isnt an issue since this is function is called in _add_name, which checks if it is already in the list
        else:  
            upper = len(name) + self.max_word_len
            lower = max(len(name) - self.max_word_len, 0)
            
            
            df_sel = self.df_ratios[(self.df_ratios['word_len'] >= lower) & (self.df_ratios['word_len'] <= upper)]
            
            # ## Option 2 (Comment out the next 3 lines to use Option 1)
            # sel_names = df_sel['surname'].tolist()
            # sel_names = get_close_matches(name, sel_names, self.n_closest * 2, self.sel_cutoff)
            # df_sel = df_sel[df_sel['surname'].isin(sel_names)]
            
            
            sel_index = df_sel.index.tolist()
            edit_dist = df_sel['sequence_matcher'].apply(lambda x: Ratios.matcher_update(x, name)).tolist()
            n_closest_inds = np.argsort(edit_dist)
            n_closest_inds = n_closest_inds[-self.n_closest:] if len(n_closest_inds) > self.n_closest else n_closest_inds
            
            n_closest_dists = [edit_dist[i] for i in n_closest_inds]
            n_closest_ratios = df_sel.iloc[n_closest_inds]['ratio'].tolist()
            n_closest_names = df_sel.iloc[n_closest_inds]['surname'].tolist()

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
        if name not in self.names:
            conf, ratio = self.process_name(name)
            self.names[name] = Name_Ratio(name, conf, ratio)
            self.counts[name] = starting_count

    # Adds a list of names to the list of names to be stored
    def populate(self, names):
        for name in names:
            
            self._add_name(name)

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
        
def process_name(name, full_name_type, type_conf, is_non_name, ratio_class, sur_expected_at_start=SURNAME_EXPECTED_AT_START):
# def process_name(row, ratio_class, sur_expected_at_start=SURNAME_EXPECTED_AT_START):
    # name, full_name_type, type_conf, is_non_name = row[PROCESS_NAME_COLUMN], row['full_name_model'], row['full_name_model_confidence'], row['is_non_name']
    name = '' if name is None else name
    
    
    n_type = TYPES['unclear']
    surname = ''
    given = name
    surname_known = False
    raw_name = name
    ratio_levels = None
    confidence_levels = None
    aggreement_type = None
    # return n_type, surname, given, surname_known, name_raw, ratio_levels, confidence_levels, aggreement_type
    
    if is_non_name or full_name_type == TYPES['blank'] or len(name) == 0:
        name = '' if REPLACE_NON_NAMES_WITH_EMPTY else name
        n_type = TYPES['non_name']
        surname = None
        given = None
        aggreement_type = (full_name_type == TYPES['blank'])
    else:
    
        
        middle_initial = False

        words = name.split()
        # name_len = [len(w) for w in words]
        initials = [word for word in words if len(word) == 1]
        initials.extend([word for word in words if word in INITIAL_OTHERS])
        non_initials = [word for word in words if word not in initials]
        # non_initials = [word for word in words if len(word) > 1]
        
        # non_initial_count = sum([n_len > 1 for n_len in name_len])
        # is_initial = [n_len == 1 for n_len in words]
        
        # If it is believed to be a blank then set it to 'non_name'
        if (len(words) == 1 and words[0].lower() == 'blank') or full_name_type == TYPES['blank']:
            n_type = TYPES['non_name']
            given = ''
        
        # if either one name or one name and one initial, then assume it is a given name
        elif (len(words) == 1 or (len(words) == MAX_INITIAL_CNT and len(non_initials) == 1)) and (full_name_type == TYPES['given'] or full_name_type == TYPES['unclear']):
            n_type = TYPES['given']
            middle_initial = len(words) == 2
        # if the full_name_type is a given name and above a threshold, assume it is a given name
        elif full_name_type == TYPES['given'] and type_conf >= FULL_NAME_MODEL_CONF_THRESH:
            n_type = TYPES['given']
            middle_initial = len(words[-1]) == 1
        # otherwise, run the ratio and surname detection
        else:
            confs, ratios = ratio_class(words)
            confidence_levels = confs
            ratio_levels = ratios
            last_name_results = is_surname(confs, ratios)
            dir = {'pos':0, 'dir':1} # expected direction of surname
            opposite = {'dir':-1, 'pos':-1}
            
            # if last name expected at end, reverse the "main" direction
            if not sur_expected_at_start:
                dir, opposite = opposite, dir
                
            res_surname = []
            surname_inds = []
            max_surname_len = MAX_SURNAME_LEN if MAX_SURNAME_LEN is not None else len(words)
            # If the whole name cannot be surname, make sure there is at least some given name.
            # NOTE: This INCLUDES initials/single letter words!!!! Could be better to only count names longer than one letter
            if not CAN_BE_ALL_SURNAME:
                # max_surname_len = min(max_surname_len, len(words) - 1)
                max_surname_len = min(max_surname_len, len(non_initials) - 1)
                
            
            # Setting at least a value to be be the surname if the full name model says it is a full name and the confidence is above the threshold
            # Thus, if the first name is not believed to be a surname, but the second is, then both will be considered surnames
            # NOTE: This could be worse than better
            if full_name_type == TYPES['full'] and type_conf >= FULL_NAME_MODEL_CONF_THRESH:
                last_name_results[dir['pos']] = True      
                
            # try to collect the words that are part of the surname
            while dir['pos'] < len(words) and last_name_results[dir['pos']]:
                surname_inds.append(dir['pos'])
                res_surname.append(words[dir['pos']])
                dir['pos'] += dir['dir']
                
                if len(res_surname) >= max_surname_len:
                    break
                
            # Reverse the surname order if they were collected backwards
            if dir['dir'] < 0:
                res_surname.reverse()
            
            surname = ' '.join(res_surname)
            surname_known = len(res_surname) > 0
            
            given = ' '.join([w for i, w in enumerate(words) if i not in surname_inds])
            
            # NOTE: alternatively, if the number of words total is >= 3, then we can assume that it is "full"
            #  By this logic, we would also say that words of length 1 are "given" names
            #  And words of length 2 are "uncertain"
            if surname_known: # or len(words) >= 3:
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
        
    return n_type, surname, given, surname_known, raw_name, ratio_levels, confidence_levels, aggreement_type

@profile
def process_names(data_to_process, ratio_class):
    """
    NOTE: It is expected that names passed into this function are already preprocessed/normalized
    """
    
    df = data_to_process
    
    full_name_included = 'full_name_model' in df.columns
    full_name_conf_included = 'full_name_model_confidence' in df.columns
    if not full_name_included:
        # df['full_name_model'] = [TYPES['unclear'] for _ in df['name_cleaned']]
        df['full_name_model'] = TYPES['unclear']
    if not full_name_conf_included:
        df['full_name_model_confidence'] = 0.0
    df['is_non_name'] = df.apply(lambda row: non_name_classifier(row[PROCESS_NAME_COLUMN]), axis=1)
    df['is_non_name'] = df['is_non_name'].fillna(False)
    # df[['name_type', 'surname', 'given_name',\
    #     'surname_known', 'name_raw', 'ratio_levels',\
    #         'conf_levels', 'aggreement_type']] = 
        # n_type, surname, given, surname_known, ratio_levels, confidence_levels, aggreement_type
    
    # row[PROCESS_NAME_COLUMN], row['full_name_model'], row['full_name_model_confidence'], row['is_non_name']
    
    
    
    is_non_names = None
    # names = data_to_process['name_cleaned']
    # full_name_types = []
    # type_confidences = []
    
    
    
    
    # if 'full_name_model' in data_to_process.columns:
    #     full_name_types = data_to_process['full_name_model']
    #     full_name_types = [FULL_NAME_MODEL_TYPES[typ] if typ in FULL_NAME_MODEL_TYPES else TYPES['unclear'] for typ in full_name_types]
    # else:
    #     full_name_types = [TYPES['unclear'] for _ in names]
    
    # if 'full_name_model_confidence' in data_to_process.columns:
    #     type_confidences = data_to_process['full_name_model_confidence']
    # else:
    #     type_confidences = [0.0 for _ in names]

    # This is probably useless...
    # should_ret_str = isinstance(names, str)
    # if should_ret_str:
    #     names = [names]
    #     if is_non_names is not None:
    #         is_non_names = [is_non_names]
    # if is_non_names is None:
    #     is_non_names = [non_name_classifier(name) for name in names]  
    # is_non_names = data_to_process['is_non_name']  
        
    
    raw_results = []
    name_types = []
    surnames = []
    given_names = []
    surname_known = []
    ratio_levels = []
    conf_levels = []
    aggreement_type = []
    
    
    
    for name, f_name_type, type_conf, is_non_name  in tqdm(zip(data_to_process['name_cleaned'], df['full_name_model'], df['full_name_model_confidence'], data_to_process['is_non_name'] )):
        # if name is None:
        #     name = ''
        # if is_non_name or f_name_type == TYPES['blank'] or len(name) == 0: 
        #     name = '' if REPLACE_NON_NAMES_WITH_EMPTY else name
        #     name_types.append(TYPES['non_name'])
        #     surnames.append(None) 
        #     given_names.append(None) 
        #     surname_known.append(False)
        #     raw_results.append(name)
        #     ratio_levels.append(None)
        #     conf_levels.append(None)
        #     aggreement_type.append(f_name_type == TYPES['blank']) 
            
        # else:
            # n_type, surname, given, is_sur_known = process_name(name, f_name_type, ratio_class)
 
        n_type, surname, given, is_sur_known, raw_name, ratio_lvls, conf_lvls, agg_type  = process_name(name, f_name_type, type_conf, is_non_name, ratio_class) 
        
        name_types.append(n_type)
        surnames.append(surname)
        given_names.append(given)
        surname_known.append(is_sur_known)
        raw_results.append(raw_name)
        ratio_levels.append(ratio_lvls)
        conf_levels.append(conf_lvls)
        aggreement_type.append(agg_type)
            
        # raw_results.append(name)
    data_to_process['name_type'] = name_types
    data_to_process['surname'] = surnames
    data_to_process['given_name'] = given_names
    data_to_process['surname_known'] = surname_known
    data_to_process['name_raw'] = raw_results
    data_to_process['ratio_levels'] = ratio_levels
    data_to_process['conf_levels'] = conf_levels
    data_to_process['aggreement_type'] = aggreement_type
    # data_to_process['is_non_name'] = is_non_names
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
    # Alternative where if '-p' is included, it will be false, else it will be true
    # parser.add_argument('-p','--distrib', action='store_false', help='If True, the dataframe with be splitted in n parts and processd in parallel')
    
    # If not in the list 'false', 'f', 'no', 'n', '0', then it will be True
    parser.add_argument('-p','--distrib', type=bool_decode, default=True, help='If True, the dataframe with be splitted in n parts and processd in parallel')
    parser.add_argument('-n', '--cores', type=int, default=6, help='Number of cores for multiprocessing')
    parser.add_argument('-d', '--dataset', type=str, default=SOURCE_DATASET, help='Input dataset to be processed')
    parser.add_argument('--sep', type=str, default=SOURCE_SEP)
    parser.add_argument('--version', type=str, default=VERSION)
    
    args = parser.parse_args()

    distrib = args.distrib
    distrib = False if DEBUG_MODE else distrib
    # print(distrib)
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
    
    tic = time.time()
    progress = tqdm(total=4, position=0, leave=True, desc='Loading data to process')
    data_to_process = load_and_preprocess_data(PATH_TO_MODIFY, MODIFY_SEP)
    progress.update(1)
    progress.set_description('Loading name data')
    
    df_ratios = load_name_data()
    ratio_class = Ratios(df_ratios)    

    progress.update(1)
    progress.set_description('Processing names')
    if distrib:
        data_to_process = parallel_processing(data_to_process, ratio_class)
    else:
        data_to_process = process_names(data_to_process, ratio_class)
    
    progress.update(1)
    progress.set_description('Copying down names')    
    data_to_process = copy_down(data_to_process)
    
    progress.update(1)
    progress.set_description('Saving data')
    data_to_process.to_csv(OUTPUT_PATH, sep=OUTPUT_SEP, index=False)

    toc = time.time()
    print(f'Runtime: {toc-tic:.2f} seconds or {(toc-tic)/60:.2f} minutes')
    