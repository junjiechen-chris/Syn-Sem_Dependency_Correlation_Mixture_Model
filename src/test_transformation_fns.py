from functools import partial
from random import random, randint

import tensorflow as tf
import numpy as np
import output_fns
from src import attention_fns
import constants
import nn_utils
import output_fns
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

class OutputFnTests(tf.test.TestCase):

  # correct solution:
  def softmax(self, x):
    """Compute softmax values for each set of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


  def test_transformation(self):
    # raise NotImplemented
    with self.test_session() as sess:

      # code = size *12 + offset
      for i in range(1):
        s = tf.constant(0.)
        div = tf.math.sigmoid(tf.constant(-20.))
        zero_div = s/div
        print(zero_div.eval())
        # k = randint(4, 15)
        # block_list = tf.constant(np.random.rand(20, k, k), dtype=tf.float32)
        head_ind = tf.constant([
          [1, 1, 3, 1, 5, 1, 5, 5],
          [1, 1, 3, 1, 5, 1, -1, -1]
        ], dtype=tf.int32)
        label_ind = tf.constant([
          [32, 0, 4, 6, 19, 3, 5, 2],
          [0, 1, 2, 1, 1, 3, -1, -1]
        ], dtype=tf.int32)
        # label_ind = tf.constant([
        #   [0, 0, 0, 0, 0, 0, 0, 0]
        # ], dtype=tf.int32)
        test_score = tf.constant(np.random.rand(3, 4, 4), dtype=tf.float32)
        test_score_label = tf.constant(np.random.rand(3, 4, 5), dtype=tf.float32)
        random_input = tf.constant(np.random.randint(0, 2, (2, 100)), dtype=tf.int32)
        random_label = tf.constant(np.random.randint(0, 44, (2, 100)), dtype=tf.int32)
        all_on_mask = tf.ones((2, 60), dtype=tf.int32)
        # print(test_score)
        score_token_to_keep = tf.constant([
          [1, 1, 1, 1],
          [1, 1, 1, 0],
          [1, 1, 0, 0],
        ], dtype=tf.int32)

        # block_list = tf.constant([[0, 2, 0, 5, 5, 2, 5], [0, 2, 2, 4, 2, -1, -1]], dtype=tf.int32)
        # block_list_label = tf.constant([[0, 2, 1, 3, 0, 4, 4], [0, 1, 1, 4, 4, -1, -1]], dtype=tf.int32)
        tokens_to_keep = tf.constant([
          [1, 1, 1, 1, 1, 1, 1, 1],
          [1, 1, 1, 1, 1, 1, 0, 0]
        ], dtype=tf.int32)
        # output = transformation_fn.get_adjacent_mtx_2(test_score, ['u2', 'd1'], tokens_to_keep, normalization = "none", using_log_prob=True, allow_intermediate_nodes = True)
        predicate_mlp = tf.random.uniform([2, 8, 4])
        role_mlp = tf.random.uniform([2, 8, 4])
        down_mask, hiddens_l, hiddens_r = output_fns.get_dep_transition_kup_mtx_collect_dep_path(head_ind, predicate_mlp,
                                                                                                 role_mlp, transpose=False,
                                                                                                 tokens_to_keep=tokens_to_keep,
                                                                                                 parse_labels=None,
                                                                                                 k=1)
        down_mask = tf.squeeze(down_mask, axis=-1)
        # output = output_fns.get_dep_transition_kup1down_mtx(parse_gold=test_score, parse_labels=test_score_label,
        #                                                     tokens_to_keep=score_token_to_keep,
        #                                                     k=1, pow_norm=True)
        # output = output_fns.get_dep_transition_mtx(parse_gold=head_ind, parse_labels=label_ind,
        #                                                     tokens_to_keep=tokens_to_keep
        #                                                   )
        # output = output_fns.get_dep_transition_mtx(parse_gold=test_score, parse_labels=test_score_label,
        #                                            tokens_to_keep=score_token_to_keep
        #                                            )
        sess.run(tf.global_variables_initializer())
        print(down_mask.eval())
        # print(u.eval())
        # print(d.eval())
        # print(debug.eval())
        # print([i.eval() for i in _] )
        # print(output_dist.eval())
      # dep_1_mtx_cmp = tf.nn.softmax(transformation_fn.one_hot(block_list))
      # print(dep_1_mtx_cmp.eval())
      # dep_1_mtx = tf.nn.softmax(transformation_fn.get_k_adjacent_mtx(block_list, ['up']))
      # print(block_list.eval())
      # assert_op = tf.debugging.assert_equal(dep_1_mtx, block_list)
      # with tf.control_dependencies([tf.debugging.assert_equal(dep_1_mtx, block_list)]):
      # 	continue
      # print(dep_1_mtx_cmp.eval())
      # print(dep_1_mtx.eval())
    # weight = tf.constant([0.2, 0.3, 0.5])
    # dependency_list_weight_pair = (dependency_list, weight)
    # attention = attention_fns.attention_to_aggregated(mode=tf.estimator.ModeKeys.TRAIN, train_attention_to_aggregated=dependency_list_weight_pair, eval_attention_to_aggregated=None)
    # print(attention.eval())


if __name__ == '__main__':

  tf.test.main()
