from __future__ import print_function
import tensorflow as tf

import argparse
import os
import pickle

from model import Model

from six import text_type


def main():
    parser = argparse.ArgumentParser(
                       formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--save_dir', type=str, default='save',
                        help='model directory to store checkpointed models')
    parser.add_argument('-n', type=int, default=5,
                        help='number of characters to sample')
    parser.add_argument('--prime', type=text_type, default=u'0.0 0.0 0.0 2887.0 2711.0 8399.0 7869.0 6630.0 7454.0 2357.0 2315.0 1541.0 4547.0 4426.0 6441.0 7061.0 5322.0 1572.0 571.0 0.0',
                        help='prime text')
    parser.add_argument('--sample', type=int, default=1,
                        help='0 to use max at each timestep, 1 to sample at '
                             'each timestep, 2 to sample on spaces')
    parser.add_argument('--word_rnn', type=bool, default=True,
                        help='是否每次生成一个词')
    parser.add_argument('--seperator', type=str, default=None,
                        help='词分隔符, 默认通过空白分隔')
    args = parser.parse_args()
    sample(args)


def sample(args):
    with open(os.path.join(args.save_dir, 'config.pkl'), 'rb') as f:
        saved_args = pickle.load(f)
    with open(os.path.join(args.save_dir, 'chars_vocab.pkl'), 'rb') as f:
        chars, vocab = pickle.load(f)
    model = Model(saved_args, training=False)
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        saver = tf.train.Saver(tf.global_variables())
        ckpt = tf.train.get_checkpoint_state(args.save_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            if args.word_rnn:
                if not args.seperator:
                    args.prime = args.prime.split()
                else:
                    args.prime = args.prime.split(args.prime.seperator)
            result = model.sample(sess, chars, vocab, args.n, args.prime,
                               args.sample, args.word_rnn)
            if args.word_rnn:
                print(' '.join(result))
            else:
                print(result.encode('utf-8').decode())

if __name__ == '__main__':
    main()
