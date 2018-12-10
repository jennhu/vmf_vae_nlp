'''
    sampler.py

    Implement class for generating sample of sentences from the corpus
    to feed into the network. Uses 200 GloVe clusters.
'''

from utils import *

class Sampler():
    def __init__(self, n_words=[5,10], corpus='ptb',
                 cluster_path='IARPA_200clusters.csv', strict_stop=False,
                 sample_path='samples/sample_data.json'):
        self.n_words = n_words
        self.corpus = corpus
        self.cluster_path = cluster_path
        self.strict_stop = strict_stop
        self.sample_path = sample_path

        if corpus == 'ptb':
            self.corpus_path = '../data/ptb.zip'
        elif corpus == 'wiki':
            self.corpus_path = '../data/WestburyLab.Wikipedia.Corpus.txt.bz2'
        else:
            raise NameError('Only ptb and wiki are currently supported')

        self.make_cluster_dict()

    def make_cluster_dict(self):
        '''
        Returns dictionary where each key is a target word, and its value
        is a list of 20 related words.
        '''
        import pandas as pd
        df = pd.read_csv(self.cluster_path,
                         usecols=['target word', '20 related words'])
        cluster_dict = df.set_index('target word').T.to_dict('records')[0]
        cluster_dict = {t : w.split(' ')  for t,w in cluster_dict.items()}
        self.clusters = cluster_dict

    def read_corpus(self):
        '''
        Returns list of sentences, delimited by <delim>,
        from self.corpus_path. The test.txt file is loaded by default.
        '''
        if self.corpus == 'ptb':
            delim = '\n'
            fname = 'test.txt'
            data = read_from_archive(self.corpus_path,
                                     '{}/{}'.format(self.corpus, fname))
            sentences = data.split(delim)
        elif self.corpus == 'wiki':
            data = read_from_archive(self.corpus_path)
            sentences = texts_to_sentences(data)
            print(len(sentences))
        return sentences

    def pick_sentence(self, target, n, strict_stop):
        '''
        Returns a sentence (str) from self.corpus with length <n>
        that contains one of the 20 words related to <target>.
        '''
        # get list of 20 related words
        rel_words = self.clusters[target]

        # get sentence of specified length that contains one of the related words
        sentences_iter = iter([s for s in self.sentences_n[n]
                               if any([w in self.tokens[s] for w in rel_words])])

        # if strict_stop, then raise StopIteraction if no sentence is found
        if strict_stop:
            try:
                sentence = next(sentences_iter)
            except StopIteration:
                err_str = 'No sentence found: n={}, target={}.'.format(n, target)
                raise NameError(err_str)
        # otherwise, return None
        else:
            sentence = next(sentences_iter, None)
        return sentence

    def get_sample(self):
        '''
        Returns nested dictionary where <targets> are the keys. The values are
        dictionaries keyed by lengths from <n_words_list>, with sentences
        as values.
        '''
        print('Reading corpus at {}\nThis may take a while...'.format(self.corpus_path))
        sentences = self.read_corpus()

        print('Tokenizing...')
        self.tokens = {
            s : word_tokenize(s) for s in sentences
        }

        print('Getting sentences of length {}...'.format(self.n_words))
        # get sentences of specified length (number of NLTK tokens)
        self.sentences_n = {
            n : sentences_of_length_n(n, self.tokens) for n in self.n_words
        }

        print('Generating sample...')
        sample = {
            t : {n : self.pick_sentence(t, n, self.strict_stop)
                     for n in self.n_words}
                for t in self.clusters.keys()
        }
        return sample

    def write_sample(self, sample):
        '''
        Writes sample dictionary to .json file.
        '''
        data_dict = {
            'corpus_path' : self.corpus_path,
            'cluster_path' : self.cluster_path,
            'sample': sample
        }
        write_to_json(data_dict, self.sample_path)
        print('Wrote sample to {}'.format(self.sample_path))
