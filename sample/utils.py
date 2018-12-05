'''
    utils.py

    Implement helper functions for sampling from corpus.
'''

##################################################################################
##  FILE I/O AND DATA STRUCTURES
##################################################################################

def read_from_archive(path_to_archive, file_to_read=None):
    # quick and dirty method to get type of archive
    archive_type = path_to_archive.split('.')[-1]
    if archive_type == 'zip':
        import zipfile
        zip = zipfile.ZipFile(path_to_archive, 'r')
        data = zip.read(file_to_read)
    elif archive_type == 'tar':
        import tarfile
        tar = tarfile.open(path_to_archive, 'r')
        f = tar.extractfile(file_to_read)
        data = f.read()
        tar.close()
    elif archive_type == 'bz2':
        import bz2
        bz_file = bz2.BZ2File(path_to_archive)
        data = bz_file.readlines() # 100000000
    else:
        raise NameError('Only .zip, .tar, and .bz2 formats supported')
    return data

def write_to_json(data_dict, json_path):
    import json
    with open(json_path, 'w') as f:
        json.dump(data_dict, f, indent=4, sort_keys=True)

def flatten(l):
    return [item for sublist in l for item in sublist]

##################################################################################
##  TOKENIZING AND TEXT PROCESSING
##################################################################################

import nltk

def not_proper(pos):
    return pos not in ['NNP', 'NNPS', 'FW', 'CD']

def word_tokenize(s):
    return nltk.word_tokenize(s.decode('utf8'))

def sent_tokenize(t):
    sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
    return sent_detector.tokenize(t.strip())

def texts_to_sentences(texts):
    sentences = []
    for t in texts:
        try:
            sentences.append(sent_tokenize(t))
        # ignore encoding errors for now
        except UnicodeDecodeError:
            pass
    sentences = flatten(sentences)
    return sentences

def sentences_of_length_n(n, tokens_dict):
    sentences_n = [s for s, tokens in tokens_dict.items()
                     if len(tokens) == n and
                        all([not_proper(pos) for token, pos
                                             in nltk.pos_tag(tokens)])]
    return sentences_n
