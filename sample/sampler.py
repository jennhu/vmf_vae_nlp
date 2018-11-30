'''
    sampler.py

    Implement class for generating sample of sentences from the corpus
    to feed into the network. Uses 200 GloVe clusters.
'''

import pandas as pd
import nltk

def read_from_zip(path_to_zipfile, file_to_read):
    import zipfile
    archive = zipfile.ZipFile(path_to_zipfile, 'r')
    data = archive.read(file_to_read)
    return data

class Sampler():
    def __init__(self, n_word_list=[5,10], corpus='ptb',
                 cluster_path='IARPA_200clusters.csv', strict_stop=False):
        self.n_word_list = n_word_list
        self.corpus = corpus
        self.corpus_path = '../data/{}.zip'.format(corpus)
        self.cluster_path = cluster_path
        self.strict_stop = strict_stop
        self.make_cluster_dict()

    def make_cluster_dict(self):
        '''
        Returns dictionary where each key is a target word, and its value
        is a list of 20 related words.
        '''
        df = pd.read_csv(self.cluster_path, skipfooter=1, engine='python',
                         usecols=['target word', '20 related words'])
        cluster_dict = df.set_index('target word').T.to_dict('records')[0]
        cluster_dict = {t : w.split(' ')  for t,w in cluster_dict.iteritems()}
        self.clusters = cluster_dict

    def read_corpus(self, delim='\n'):
        '''
        Returns list of sentences, delimited by <delim>,
        from self.corpus_path.
        '''
        data = read_from_zip(self.corpus_path, '{}/test.txt'.format(self.corpus))
        sentences = data.split(delim)
        return sentences

    def pick_sentence(self, target, n):
        '''
        Returns a sentence (str) from self.corpus with length <n>
        that contains one of the 20 words related to <target>.
        '''
        # get list of 20 related words
        rel_words = self.clusters[target]

        # get sentences of specified length (number of NLTK tokens)
        sentences_n = [s for s,ts in self.tokens.iteritems() if len(ts) == n]

        # get sentence of specified length that contains one of the related words
        sentences_iter = iter([s for s in sentences_n
                               if any([w in self.tokens[s] for w in rel_words])])

        # if self.strict_stop, then stop if no sentence is found
        if self.strict_stop:
            try:
                sentence = next(sentences_iter)
            except StopIteration:
                err_str = 'No sentence found: n={}, target={}.'.format(n, target)
                raise NameError(err_str)
        # otherwise, insert None and move on
        else:
            sentence = next(sentences_iter, None)
        return sentence

    def get_sample(self):
        '''
        Returns nested dictionary where <targets> are the keys. The values are
        dictionaries keyed by lengths from <n_words_list>, with sentences
        as values.
        '''
        print('Reading corpus at {}'.format(self.corpus_path))
        sentences = self.read_corpus()
        print('Tokenizing...')
        self.tokens = {
            s : nltk.word_tokenize(s) for s in sentences
        }
        print('Generating sample...')
        sample = {
            t : {n : self.pick_sentence(t, n) for n in self.n_word_list}
                for t in self.clusters.keys()
        }
        return sample
