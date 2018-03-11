import os
import pickle

import torch

TRAIN_BATCH_SZ = 20


def preprocess_data(path):
    # Read all data
    train = read_txt_file(os.path.join(path, 'train.txt'))
    valid = read_txt_file(os.path.join(path, 'valid.txt'))
    test = read_txt_file(os.path.join(path, 'test.txt'))

    # Build dict
    tmp_dict = {}
    tmp_dict = count(tmp_dict, train)
    tmp_dict = count(tmp_dict, valid)
    tmp_dict = count(tmp_dict, test)
    word_order = [k for k in sorted(tmp_dict, key=tmp_dict.get, reverse=True)]
    word_order.insert(0, '<pad>')

    os.chdir(path)

    # Batch the data
    batch_data(word_order, test, 'test', 1)
    batch_data(word_order, valid, 'valid', 1)
    batch_data(word_order, train, 'train', TRAIN_BATCH_SZ)

    # Store dict
    with open(os.path.join(path, 'vocab.dict'), 'wb') as fd:
        pickle.dump(word_order, fd)


def pack_a_batch(dict, bag):
    max_len = len(bag[0])
    batch_sz = len(bag)
    txt_mat = torch.zeros(batch_sz, max_len)
    mask = [len(x) for x in bag]
    for bidx, batch in enumerate(bag):
        for widx, w in enumerate(batch):
            id = dict.index(w)
            txt_mat[bidx][widx] = id
    return [txt_mat, mask]


def batch_data(dict, data, name, batch_sz):
    data = sorted(data, key=lambda x: len(x), reverse=True)
    data_patch = []
    buff = 0
    tmp_bag = []
    for d in data:
        tmp_bag.append(d)
        buff += 1
        if buff >= batch_sz:
            data_patch.append(pack_a_batch(dict, tmp_bag))
            buff = 0
            tmp_bag = []
    if tmp_bag != []:
        data_patch.append(pack_a_batch(dict, tmp_bag))

    os.chdir(name + 's')
    for idx, dp in enumerate(data_patch):
        with open(name + '-' + str(idx) + '.bin', 'wb') as wfd:
            pickle.dump(dp, wfd)

    os.chdir('..')
    print('Writing finished: %s' % name)


def count(dict, inp):
    for s in inp:
        for w in s:
            if w in dict:
                dict[w] += 1
            else:
                dict[w] = 1
    return dict


def read_txt_file(path):
    """
    Read txt file and return splited words in list.
    :param path: complete file path
    :return: [[s1w1, s1w2, ...], [], ...]
    """
    with open(path, 'r') as f:
        lists = f.read().splitlines()
        lists = [s.strip().split(' ') for s in lists]
    return lists


if __name__ == "__main__":
    path = '/home/jcxu/vae_txt/data/ptb'
    preprocess_data(path)
