'''
    sample.py

    Implement class for generating sample of sentences from the corpus
    to feed into the network. Uses 200 GloVe clusters.
'''

import pandas as pd

def unzip(path_to_zip, dir_to_write):
    import zipfile
    with zipfile.ZipFile(path_to_zip, 'r') as zip_ref:
        zip_ref.extractall(dir_to_write)

def read_from_zip(path_to_zipfile, file_to_read):
    import zipfile
    archive = zipfile.ZipFile(path_to_zipfile, 'r')
    data = archive.read(file_to_read)
    return data

class Sampler():
    def __init__(self, corpus, n_word_list=[4,5,8],
                 cluster_path='IARPA_200clusters.csv',
		 corpus='ptb'):
        self.corpus = corpus
        self.n_word_list = n_word_list
        self.cluster_path = cluster_path
	self.data_path = '../data/{}.zip'
        self.make_cluster_dict()

    def make_cluster_dict(self):
        '''
        Returns dictionary where each key is a target word, and its value
        is a list of 20 related words.
        '''
        df = pd.read_csv(self.cluster_path, skipfooter=1,
                         usecols=['target word', '20 related words'])
        cluster_dict = df.set_index('target word').T.to_dict('records')[0]
        cluster_dict = {t : w.split(' ')  for t,w in cluster_dict.iteritems()}
        self.clusters = cluster_dict

    def pick_sentence(self, target, n):
        '''
        Returns a sentence (str) from self.corpus with length <n>
        that contains one of the 20 words related to <target>.
        '''
        related_words = self.clusters[target]
        # TODO: pick sentence from corpus -- rejection sample? filter by length?
	# in PTB (WSJ), sentences are delimited with \n
	data = read_from_zip(self.data_path)

    def get_sample(self):
        '''
        Returns dictionary where <targets> are the keys, and the values are
        a list of sampled sentences with lengths from <n_words_list>.
        '''
        sample = [self.pick_sentence(t, n) for t in self.clusters.keys()
                                           for n in self.n_word_list]
        return sample
