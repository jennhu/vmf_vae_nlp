from sampler import Sampler
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_word_list', type=int, nargs='+', default=[5,10])
    parser.add_argument('--corpus', type=str, choices=['ptb'], default='ptb')
    parser.add_argument('--cluster_path', type=str, default='IARPA_200clusters.csv')
    parser.add_argument('--strict_stop', action='store_true', default=False)
    args = parser.parse_args()

    # NOTE: yelp corpus not supported yet
    s = Sampler(**vars(args))
    sample = s.get_sample()
    print(sample['food'])
    print(sample['money'])
