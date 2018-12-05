from sampler import Sampler
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_words', type=int, nargs='+', default=[5,10])
    parser.add_argument('--corpus', type=str, choices=['ptb','wiki'], default='ptb')
    parser.add_argument('--cluster_path', type=str, default='IARPA_200clusters.csv')
    parser.add_argument('--strict_stop', action='store_true', default=False)
    parser.add_argument('--sample_path', type=str, default='samples/sample_data.json')
    args = parser.parse_args()

    # NOTE: yelp corpus not supported yet
    s = Sampler(**vars(args))

    sample = s.get_sample()
    s.write_sample(sample)
