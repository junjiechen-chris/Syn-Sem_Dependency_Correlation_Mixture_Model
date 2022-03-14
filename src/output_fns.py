import heapq
from collections import OrderedDict

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.estimator import ModeKeys

import constants
import nn_utils
import tf_utils
import train_utils
import transformation_fn

feature_idx_to_step = {i * 10 + j: (i, j) for i in range(0, 10) for j in range(0, 10)}
step_to_feature_idx = {(i, j): i * 10 + j for i in range(0, 10) for j in range(0, 10)}


def softmax_classifier(mode, hparams, model_config, inputs, targets, num_labels, tokens_to_keep, transition_params):
  with tf.name_scope('softmax_classifier'):
    # # todo pass this as initial proj dim (which is optional)
    # projection_dim = model_config['predicate_pred_mlp_size']
    #
    # with tf.variable_scope('MLP'):
    #   mlp = nn_utils.MLP(inputs, projection_dim, keep_prob=hparams.mlp_dropout, n_splits=1)
    with tf.variable_scope('Classifier'):
      logits = nn_utils.MLP(inputs, num_labels, keep_prob=hparams.mlp_dropout, n_splits=1)

    # todo implement this
    if transition_params is not None:
      print('Transition params not yet supported in softmax_classifier')
      exit(1)

    # cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=targets)
    targets_onehot = tf.one_hot(indices=targets, depth=num_labels, axis=-1)
    loss = tf.losses.softmax_cross_entropy(logits=tf.reshape(logits, [-1, num_labels]),
                                           onehot_labels=tf.reshape(targets_onehot, [-1, num_labels]),
                                           weights=tf.reshape(tokens_to_keep, [-1]),
                                           label_smoothing=hparams.label_smoothing,
                                           reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS)
    # loss = tf.Print(loss, [loss], ' softmax classifier')
    #
    # loss = 0.
    predictions = tf.cast(tf.argmax(logits, axis=-1), tf.int32)
    # loss = tf.zeros([])
    output = {
      'loss': loss,
      'predictions': predictions,
      'scores': logits,
      'probabilities': tf.nn.softmax(logits, -1)
    }

  return output


def softmax_classifier_2(mode, hparams, model_config, inputs, targets, num_labels, tokens_to_keep, transition_params):
  with tf.name_scope('softmax_classifier_2'):
    # # todo pass this as initial proj dim (which is optional)
    # projection_dim = model_config['predicate_pred_mlp_size']
    #
    # with tf.variable_scope('MLP'):
    #   mlp = nn_utils.MLP(inputs, projection_dim, keep_prob=hparams.mlp_dropout, n_splits=1)
    with tf.variable_scope('SM2_Classifier'):
      logits = nn_utils.MLP(inputs, num_labels, keep_prob=hparams.mlp_dropout, n_splits=1)

    # todo implement this
    if transition_params is not None:
      print('Transition params not yet supported in softmax_classifier')
      exit(1)

    # cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=targets)
    targets_onehot = tf.one_hot(indices=targets, depth=num_labels, axis=-1)
    loss = tf.losses.softmax_cross_entropy(logits=tf.reshape(logits, [-1, num_labels]),
                                           onehot_labels=tf.reshape(targets_onehot, [-1, num_labels]),
                                           weights=tf.reshape(tokens_to_keep, [-1]),
                                           label_smoothing=hparams.label_smoothing,
                                           reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS)
    # loss = tf.Print(loss, [loss], ' softmax classifier')
    #
    # loss = 0.
    predictions = tf.cast(tf.argmax(logits, axis=-1), tf.int32)
    # loss = tf.zeros([])
    output = {
      'loss': loss,
      'predictions': predictions,
      'scores': logits,
      'probabilities': tf.nn.softmax(logits, -1)
    }

  return output


def get_separate_scores_preds_from_joint(joint_outputs, joint_maps, joint_num_labels):
  predictions = joint_outputs['predictions']
  scores = joint_outputs['scores']
  output_shape = tf.shape(predictions)
  batch_size = output_shape[0]
  batch_seq_len = output_shape[1]
  sep_outputs = {}
  for map_name, label_comp_map in joint_maps.items():
    # print("debug <joint map>: name {}, comp_map {}".format(map_name, label_comp_map))
    short_map_name = map_name.split('_to_')[-1]
    # label_comp_predictions = tf.nn.embedding_lookup(label_comp_map, predictions)
    # sep_outputs["%s_predictions" % short_map_name] = tf.squeeze(label_comp_predictions, -1)

    # marginalize out probabilities for this task
    task_num_labels = tf.shape(tf.unique(tf.reshape(label_comp_map, [-1]))[0])[0]
    joint_probabilities = tf.nn.softmax(scores)
    joint_probabilities_flat = tf.reshape(joint_probabilities, [-1, joint_num_labels])
    # print("debug <computing segment_ids>! ")
    # tf.Print("debug <@get_separate_scores_preds_from_joint>:", tf.nn.embedding_lookup(label_comp_map, tf.range(joint_num_labels)))
    segment_ids = tf.squeeze(tf.nn.embedding_lookup(label_comp_map, tf.range(joint_num_labels)), -1)
    segment_scores = tf.unsorted_segment_sum(tf.transpose(joint_probabilities_flat), segment_ids, task_num_labels)
    segment_scores = tf.reshape(tf.transpose(segment_scores), [batch_size, batch_seq_len, task_num_labels])
    sep_outputs["%s_probabilities" % short_map_name] = segment_scores

    # use marginalized probabilities to get predictions
    sep_outputs["%s_predictions" % short_map_name] = tf.argmax(segment_scores, -1)
  # print("debug <sep output>: ", sep_outputs)
  return sep_outputs


def joint_softmax_classifier(mode, hparams, model_config, inputs, targets, num_labels, tokens_to_keep, joint_maps,
                             transition_params):
  with tf.name_scope('joint_softmax_classifier'):
    # todo pass this as initial proj dim (which is optional)
    projection_dim = model_config['predicate_pred_mlp_size']

    with tf.variable_scope('MLP'):
      mlp = nn_utils.MLP(inputs, projection_dim, keep_prob=hparams.mlp_dropout, n_splits=1)
    with tf.variable_scope('Classifier'):
      logits = nn_utils.MLP(mlp, num_labels, keep_prob=hparams.mlp_dropout, n_splits=1)

    # todo implement this
    if transition_params is not None:
      print('Transition params not yet supported in joint_softmax_classifier')
      exit(1)

    # print("debug <shape of joint softmax classifier logit vs. target>", logits, " ", targets)

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=targets)

    # print("debug <shape of joint softmax classifier ce vs. tokens_to_keep>", cross_entropy, " ", tokens_to_keep)
    cross_entropy *= tokens_to_keep
    loss = tf.reduce_sum(cross_entropy) / tf.reduce_sum(tokens_to_keep)
    # loss = 0.
    # loss = tf.Print(loss, [loss], ' joint softmax classifier')

    predictions = tf.cast(tf.argmax(logits, axis=-1), tf.int32)
    # loss = tf.Print(loss, [loss], "JOINT LOSS")
    if mode != ModeKeys.TRAIN:
      loss = tf.zeros([1])

    output = {
      'loss': loss,
      'predictions': predictions,
      'scores': logits,
      'probabilities': tf.nn.softmax(logits, -1),
      'hidden': mlp
    }

    # now get separate-task scores and predictions for each of the maps we've passed through
    separate_output = get_separate_scores_preds_from_joint(output, joint_maps, num_labels)
    combined_output = OrderedDict({**output, **separate_output})

    return combined_output


def joint_softmax_classifier_wsm(mode, hparams, model_config, inputs, targets, num_labels, tokens_to_keep, joint_maps,
                                 transition_params):
  with tf.name_scope('joint_softmax_classifier'):
    # todo pass this as initial proj dim (which is optional)
    projection_dim = model_config['predicate_pred_mlp_size']

    with tf.variable_scope('MLP'):
      mlp = nn_utils.MLP(inputs, projection_dim, keep_prob=hparams.mlp_dropout, n_splits=1)
    with tf.variable_scope('Classifier'):
      logits = nn_utils.MLP(mlp, num_labels, keep_prob=hparams.mlp_dropout, n_splits=1)

    # todo implement this
    if transition_params is not None:
      print('Transition params not yet supported in joint_softmax_classifier')
      exit(1)

    # print("debug <shape of joint softmax classifier logit vs. target>", logits, " ", targets)
    targets = tf.one_hot(targets, num_labels, axis=-1)
    cross_entropy = tf.reduce_sum(
      tf.nn.weighted_cross_entropy_with_logits(logits=logits, targets=targets, pos_weight=0.8), axis=-1)

    # print("debug <shape of joint softmax classifier ce vs. tokens_to_keep>", cross_entropy, " ", tokens_to_keep)
    cross_entropy *= tokens_to_keep
    loss = tf.reduce_sum(cross_entropy) / tf.reduce_sum(tokens_to_keep)
    # loss = 0.
    # loss = tf.Print(loss, [loss], ' joint softmax classifier')

    predictions = tf.cast(tf.argmax(logits, axis=-1), tf.int32)
    # loss = tf.Print(loss, [loss], "JOINT LOSS")

    output = {
      'loss': loss,
      'predictions': predictions,
      'scores': logits,
      'probabilities': tf.nn.softmax(logits, -1)
    }

    # now get separate-task scores and predictions for each of the maps we've passed through
    separate_output = get_separate_scores_preds_from_joint(output, joint_maps, num_labels)
    combined_output = OrderedDict({**output, **separate_output})

    return combined_output


def joint_softmax_classifier_ls(mode, hparams, model_config, inputs, targets, num_labels, tokens_to_keep, joint_maps,
                                transition_params):
  with tf.name_scope('joint_softmax_classifier'):
    # todo pass this as initial proj dim (which is optional)
    projection_dim = model_config['predicate_pred_mlp_size']

    with tf.variable_scope('MLP'):
      mlp = nn_utils.MLP(inputs, projection_dim, keep_prob=hparams.mlp_dropout, n_splits=1)
    with tf.variable_scope('Classifier'):
      logits = nn_utils.MLP(mlp, num_labels, keep_prob=hparams.mlp_dropout, n_splits=1)

    # todo implement this
    if transition_params is not None:
      print('Transition params not yet supported in joint_softmax_classifier')
      exit(1)

    # print("debug <shape of joint softmax classifier logit vs. target>", logits, " ", targets)

    targets_onehot = tf.one_hot(indices=targets, depth=num_labels, axis=-1)
    loss = tf.losses.softmax_cross_entropy(logits=tf.reshape(logits, [-1, num_labels]),
                                           onehot_labels=tf.reshape(targets_onehot, [-1, num_labels]),
                                           weights=tf.reshape(tokens_to_keep, [-1]),
                                           label_smoothing=hparams.label_smoothing,
                                           reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS)

    # cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=targets)

    # print("debug <shape of joint softmax classifier ce vs. tokens_to_keep>", cross_entropy, " ", tokens_to_keep)
    # cross_entropy *= tokens_to_keep
    # loss = tf.reduce_sum(cross_entropy) / tf.reduce_sum(tokens_to_keep)
    # loss = 0.
    # loss = tf.Print(loss, [loss], ' joint softmax classifier')

    predictions = tf.cast(tf.argmax(logits, axis=-1), tf.int32)

    output = {
      'loss': loss,
      'predictions': predictions,
      'scores': logits,
      'probabilities': tf.nn.softmax(logits, -1)
    }

    # now get separate-task scores and predictions for each of the maps we've passed through
    separate_output = get_separate_scores_preds_from_joint(output, joint_maps, num_labels)
    combined_output = OrderedDict({**output, **separate_output})

    return combined_output


def parse_bilinear_with_decedents(mode, hparams, model_config, inputs, targets, num_labels, tokens_to_keep,
                                  transition_params):
  class_mlp_size = model_config['class_mlp_size']
  attn_mlp_size = model_config['attn_mlp_size']

  if transition_params is not None:
    print('Transition params not supported in parse_bilinear')
    exit(1)

  with tf.variable_scope('parse_bilinear_with_decedents'):
    with tf.variable_scope('MLP'):
      dep_mlp, head_mlp = nn_utils.MLP(inputs, class_mlp_size + attn_mlp_size, n_splits=2,
                                       keep_prob=hparams.mlp_dropout)
      dep_arc_mlp, dep_rel_mlp = dep_mlp[:, :, :attn_mlp_size], dep_mlp[:, :, attn_mlp_size:]
      head_arc_mlp, head_rel_mlp = head_mlp[:, :, :attn_mlp_size], head_mlp[:, :, attn_mlp_size:]

    with tf.variable_scope('Arcs'):
      # batch_size x batch_seq_len x batch_seq_len
      arc_logits = nn_utils.bilinear_classifier(dep_arc_mlp, head_arc_mlp, hparams.bilinear_dropout)
      # mean, variance = tf.nn.moments(arc_logits, [-1], keep_dims=True)
      # with tf.variable_scope('BatchNorm'):
      #   beta = tf.get_variable('offset', [1, heads])
      #   gamma = tf.get_variable('scale', [1])
      # arc_logits = tf.nn.batch_normalization(arc_logits, mean, variance, offset=beta, scale=gamma, variance_epsilon=1e-6)

    num_tokens = tf.reduce_sum(tokens_to_keep)

    predictions = tf.argmax(arc_logits, -1)
    probabilities = tf.nn.softmax(arc_logits)

    # arc_logits = tf.Print(arc_logits, [arc_logits], 'parse_bilinear_arc_logit')
    # targets = tf.Print(targets, [])
    # targets = tf.Print(targets, [targets, tf.reduce_sum(targets)], 'parse_bilinear_targets, checking whether the target vector is passed correctly')

    targets_decedent = transformation_fn.get_decedent_mtx(targets)

    cross_entropy_heads = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=arc_logits, labels=targets)
    # cross_entropy = tf.Print(cross_entropy, [cross_entropy], 'parse_ce_loss')
    loss_head = tf.reduce_sum((cross_entropy_heads * tokens_to_keep)) / num_tokens

    with tf.variable_scope('bias'):
      beta = tf.get_variable('offset', [])
      cross_entropy_decedent = tf.nn.sigmoid_cross_entropy_with_logits(logits=arc_logits + beta,
                                                                       labels=targets_decedent)
    # cross_entropy = tf.Print(cross_entropy, [cross_entropy], 'parse_ce_loss')
    seq_len = tf.shape(tokens_to_keep)[-1]
    mask_decedents_x = tf.tile(tf.expand_dims(tokens_to_keep, -1), [1, 1, seq_len])
    mask_decedents_y = tf.tile(tf.expand_dims(tokens_to_keep, 1), [1, seq_len, 1])
    mask_decedents = tf.where(
      tf.logical_and(tf.cast(mask_decedents_x, dtype=tf.bool), tf.cast(mask_decedents_y, dtype=tf.bool)),
      tf.ones_like(mask_decedents_x), tf.zeros_like(mask_decedents_x))
    loss_decedent = tf.reduce_sum((cross_entropy_decedent * mask_decedents)) / tf.reduce_sum(mask_decedents)
    loss = loss_head + loss_decedent
    # loss = 0.
    # loss = tf.zeros([])
    # loss = tf.Print(loss, [loss], 'parse_bilinear')
    output = {
      'loss': loss,
      'loss_head': loss_head,
      'loss_decedent': loss_decedent,
      'predictions': predictions,
      'probabilities': probabilities,
      'scores': arc_logits,
      'dep_rel_mlp': dep_rel_mlp,
      'head_rel_mlp': head_rel_mlp
    }

  return output


def parse_bilinear(mode, hparams, model_config, inputs, targets, num_labels, tokens_to_keep, transition_params):
  class_mlp_size = model_config['class_mlp_size']
  attn_mlp_size = model_config['attn_mlp_size']

  if transition_params is not None:
    print('Transition params not supported in parse_bilinear')
    exit(1)

  with tf.variable_scope('parse_bilinear'):
    with tf.variable_scope('MLP'):
      dep_mlp, head_mlp = nn_utils.MLP(inputs, class_mlp_size + attn_mlp_size, n_splits=2,
                                       keep_prob=hparams.mlp_dropout)
      dep_arc_mlp, dep_rel_mlp = dep_mlp[:, :, :attn_mlp_size], dep_mlp[:, :, attn_mlp_size:]
      head_arc_mlp, head_rel_mlp = head_mlp[:, :, :attn_mlp_size], head_mlp[:, :, attn_mlp_size:]

    with tf.variable_scope('Arcs'):
      # batch_size x batch_seq_len x batch_seq_len
      arc_logits = nn_utils.bilinear_classifier(dep_arc_mlp, head_arc_mlp, hparams.bilinear_dropout)

    num_tokens = tf.reduce_sum(tokens_to_keep)

    predictions = tf.argmax(arc_logits, -1)
    probabilities = tf.nn.softmax(arc_logits)
    # targets = tf.Print(targets, [targets, arc_logits], "TARGETS")
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=arc_logits, labels=targets)
    # cross_entropy = tf.Print(cross_entropy, [cross_entropy], "CE", summarize=80)

    # cross_entropy = tf.Print(cross_entropy, [cross_entropy], 'parse_ce_loss')
    loss = tf.reduce_sum((cross_entropy * tokens_to_keep)) / num_tokens
    if mode != ModeKeys.TRAIN:
      loss -= loss
    # loss = tf.Print(loss, [loss], "PARSE HEAD LOSS")
    # loss = 0.
    # loss = tf.zeros([])
    # loss = tf.Print(loss, [loss], 'parse_bilinear')
    output = {
      'loss': loss,
      'predictions': predictions,
      'probabilities': probabilities,
      'scores': arc_logits,
      'dep_rel_mlp': dep_rel_mlp,
      'head_rel_mlp': head_rel_mlp
    }

  return output


def get_lprior_up_mtx(parse_gold, parse_label, num_labels, tokens_to_keep=None, extreme_value=False,
                      layer_norm_to_heads=False, transpose=False, memory_efficient=False, joint_par_srl_training=False,
                      relu_imp=False, pos_tag=None):
  """
    heads: head-dependent distribution of shape (B, seq_len, seq_len)
    labels: label distribution for each head-dependent choice, shape: (B, seq_len, labels)
  """

  def prod(l, r):
    # l, r are of (B, S, S, num_labels)
    l_t = tf.stack(tf.unstack(l, axis=-1), axis=0)
    r_t = tf.stack(tf.unstack(r, axis=-1), axis=0)
    tmp_t = tf.linalg.matmul(l_t, r_t)
    tmp = tf.stack(tf.unstack(tmp_t, axis=0), axis=-1)
    return tmp

  heads = parse_gold
  labels = parse_label
  # assert  head_gen_fn_train == nn_utils.generating_prior_mtx_for_srl
  # Note that row mask is applied for log-scale and col mask is applied for normal scale
  token_mask_row = tf.expand_dims(tf.cast(1 - tokens_to_keep, tf.float32) * constants.VERY_SMALL, 1)
  token_mask_col = tf.expand_dims(tf.cast(tokens_to_keep, tf.float32), 2)

  tf.logging.log(tf.logging.INFO, "Generating labeled-prior for SRL")

  if len(heads.get_shape()) < 3:
    if extreme_value:
      on_value = constants.VERY_LARGE
      off_value = constants.VERY_SMALL
      off_value_label = constants.VERY_SMALL
    else:
      on_value = 10.
      off_value = -10.
      off_value_label = -10.

    heads = tf.one_hot(heads, tf.shape(heads)[-1], off_value=off_value, on_value=on_value)
    labels = tf.one_hot(labels, 69, off_value=off_value_label, on_value=on_value)  # hard code num labels for srl 09
    if pos_tag is not None:
      pos_tag = tf.one_hot(pos_tag, 48, off_value=off_value_label, on_value=on_value)
    heads = heads + token_mask_row

  else:

    heads = heads + token_mask_row
    if not joint_par_srl_training:
      heads = tf.stop_gradient(heads)
      labels = tf.stop_gradient(labels)
    # heads = tf.concat([heads[:, :1, :] + 100, heads[:, 1:, :]], axis=1)

  if not relu_imp:
    masked_heads = nn_utils.generating_prior_mtx_for_srl(heads, labels, num_labels, chain=['u1'],
                                                         tokens_to_keep=tokens_to_keep,
                                                         memory_efficient=memory_efficient)[0]
  else:
    masked_heads = nn_utils.generating_prior_mtx_for_srl_relu(heads, labels, num_labels, chain=['u1'],
                                                              tokens_to_keep=tokens_to_keep,
                                                              memory_efficient=memory_efficient)[0]
  masked_heads *= tf.expand_dims(token_mask_col, axis=-1)
  if transpose:
    masked_heads = tf.transpose(masked_heads, perm=[0, 2, 1, 3])
  # Get predicate POS into play
  if pos_tag is not None:
    init_head = nn_utils.generating_prior_mtx_with_pos(pos_tag, num_labels)
    masked_heads = prod(init_head, masked_heads)

  masked_heads = tf.Print(masked_heads, [masked_heads[:, :, :, :5]], "priors inferred from dependency graph",
                          summarize=40)
  return masked_heads


def get_lprior_kup1down_mtx(parse_gold, parse_label, num_labels, tokens_to_keep=None, extreme_value=False,
                            layer_norm_to_heads=False, k=-1, memory_efficient=False, joint_par_srl_training=False,
                            relu_imp=False):
  assert k > 0
  heads = parse_gold
  labels = parse_label
  token_mask_row = tf.expand_dims(tf.cast(1 - tokens_to_keep, tf.float32) * constants.VERY_SMALL, 1)
  token_mask_col = tf.expand_dims(tf.cast(tokens_to_keep, tf.float32), 2)
  # token_mask_col = tf.expand_dims(tf.cast(1 - tokens_to_keep, tf.float32) * constants.VERY_SMALL, 2)

  tf.logging.log(tf.logging.INFO, "Using simple {}up1down direction, removing all prob to ROOT".format(k))

  def prod(l, r):
    # l, r are of (B, S, S, num_labels)
    l_t = tf.stack(tf.unstack(l, axis=-1), axis=0)
    r_t = tf.stack(tf.unstack(r, axis=-1), axis=0)
    tmp_t = tf.linalg.matmul(l_t, r_t)
    tmp = tf.stack(tf.unstack(tmp_t, axis=0), axis=-1)
    return tmp

  if len(heads.get_shape()) < 3:
    # max_seq_len = tf.cast(tf.shape(heads)[-1], dtype=tf.float32)

    if extreme_value:
      on_value = constants.VERY_LARGE
      off_value = constants.VERY_SMALL
      off_value_label = constants.VERY_SMALL
    else:
      on_value = 10.
      off_value = -10.
      off_value_label = -10.

    heads = tf.one_hot(heads, tf.shape(heads)[-1], off_value=off_value, on_value=on_value)
    labels = tf.one_hot(labels, 69, off_value=off_value_label, on_value=on_value)
    heads = heads + token_mask_row
  else:

    heads = heads + token_mask_row
    if not joint_par_srl_training:
      heads = tf.stop_gradient(heads)
      labels = tf.stop_gradient(labels)

  print("heads, labels, ", heads, labels)
  if relu_imp:
    masked_heads = nn_utils.generating_prior_mtx_for_srl_relu(heads, labels, num_labels, chain=['u{}'.format(k + 1)],
                                                              tokens_to_keep=tokens_to_keep,
                                                              memory_efficient=memory_efficient)
  else:
    masked_heads = nn_utils.generating_prior_mtx_for_srl(heads, labels, num_labels, chain=['u{}'.format(k + 1)],
                                                         tokens_to_keep=tokens_to_keep,
                                                         memory_efficient=memory_efficient)
  # Applying column-wise masking
  masked_heads = [masked_heads[idx] * tf.expand_dims(token_mask_col, axis=-1) for idx in range(k + 1)]
  # masked_heads = tf.map_fn(lambda mtx: mtx+token_mask_col, masked_heads, dtype=tf.float32)[0]
  # First, we make a up/down matrix, then remove prob on root node
  up, down = masked_heads[-2:]
  # Prevent back-looping
  updown = prod(up, tf.transpose(down, perm=[0, 2, 1, 3])) * tf.expand_dims(
    1 - tf.eye(tf.shape(heads)[1], batch_shape=[tf.shape(heads)[0]]), -1)
  tmp = updown
  for idx in reversed(range(k - 1)):
    tmp = prod(masked_heads[idx], tmp)
  tmp = tf.Print(tmp, [tmp[:, :, :, :5]], "merged masked_head")
  return tmp


def parse_bilinear_msm(mode, hparams, model_config, inputs, targets, num_labels, tokens_to_keep, transition_params):
  class_mlp_size = model_config['class_mlp_size']
  attn_mlp_size = model_config['attn_mlp_size']

  if transition_params is not None:
    print('Transition params not supported in parse_bilinear')
    exit(1)

  with tf.variable_scope('parse_bilinear'):
    with tf.variable_scope('MLP'):
      dep_mlp, head_mlp = nn_utils.MLP(inputs, class_mlp_size + attn_mlp_size, n_splits=2,
                                       keep_prob=hparams.mlp_dropout)
      dep_arc_mlp, dep_rel_mlp = dep_mlp[:, :, :attn_mlp_size], dep_mlp[:, :, attn_mlp_size:]
      head_arc_mlp, head_rel_mlp = head_mlp[:, :, :attn_mlp_size], head_mlp[:, :, attn_mlp_size:]

    with tf.variable_scope('Arcs'):
      # batch_size x batch_seq_len x batch_seq_len
      arc_logits = nn_utils.bilinear_classifier(dep_arc_mlp, head_arc_mlp, hparams.bilinear_dropout)

    num_tokens = tf.reduce_sum(tokens_to_keep)

    predictions = tf.argmax(arc_logits, -1)
    probabilities = tf.nn.softmax(arc_logits)

    # arc_logits = tf.Print(arc_logits, [arc_logits], 'parse_bilinear_arc_logit')
    # targets = tf.Print(targets, [])
    # targets = tf.Print(targets, [targets, tf.reduce_sum(targets)], 'parse_bilinear_targets, checking whether the target vector is passed correctly')
    num_labels = tf.shape(targets)[-1]
    parse_target_onehot = tf.one_hot(indices=targets, depth=tf.shape(targets)[-1], axis=-1)
    loss = tf.losses.softmax_cross_entropy(logits=tf.reshape(arc_logits, [-1, num_labels]),
                                           onehot_labels=tf.reshape(parse_target_onehot, [-1, num_labels]),
                                           # tf.reshape(srl_targets_onehot, [-1, num_labels]),
                                           weights=tf.reshape(tokens_to_keep, [-1]),
                                           reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS)

    # cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=arc_logits, labels=targets)
    # cross_entropy = tf.Print(cross_entropy, [cross_entropy], 'parse_ce_loss')
    # loss = tf.reduce_sum((cross_entropy * tokens_to_keep))/ num_tokens
    # loss = 0.
    # loss = tf.zeros([])
    # loss = tf.Print(loss, [loss], 'parse_bilinear')
    if mode != ModeKeys.TRAIN:
      loss -= loss
    output = {
      'loss': loss,
      'predictions': predictions,
      'probabilities': probabilities,
      'scores': arc_logits,
      'dep_rel_mlp': dep_rel_mlp,
      'head_rel_mlp': head_rel_mlp
    }

  return output


def parse_bilinear_ls(mode, hparams, model_config, inputs, targets, num_labels, tokens_to_keep, transition_params):
  class_mlp_size = model_config['class_mlp_size']
  attn_mlp_size = model_config['attn_mlp_size']

  if transition_params is not None:
    print('Transition params not supported in parse_bilinear')
    exit(1)

  with tf.variable_scope('parse_bilinear'):
    with tf.variable_scope('MLP'):
      dep_mlp, head_mlp = nn_utils.MLP(inputs, class_mlp_size + attn_mlp_size, n_splits=2,
                                       keep_prob=hparams.mlp_dropout)
      dep_arc_mlp, dep_rel_mlp = dep_mlp[:, :, :attn_mlp_size], dep_mlp[:, :, attn_mlp_size:]
      head_arc_mlp, head_rel_mlp = head_mlp[:, :, :attn_mlp_size], head_mlp[:, :, attn_mlp_size:]

    with tf.variable_scope('Arcs'):
      # batch_size x batch_seq_len x batch_seq_len
      arc_logits = nn_utils.bilinear_classifier(dep_arc_mlp, head_arc_mlp, hparams.bilinear_dropout)

    num_tokens = tf.reduce_sum(tokens_to_keep)

    predictions = tf.argmax(arc_logits, -1)
    probabilities = tf.nn.softmax(arc_logits)

    # arc_logits = tf.Print(arc_logits, [arc_logits], 'parse_bilinear_arc_logit')
    # targets = tf.Print(targets, [])
    # targets = tf.Print(targets, [targets, tf.reduce_sum(targets)], 'parse_bilinear_targets, checking whether the target vector is passed correctly')
    num_labels = tf.shape(targets)[-1]
    parse_target_onehot = tf.one_hot(indices=targets, depth=tf.shape(targets)[-1], axis=-1)
    loss = tf.losses.softmax_cross_entropy(logits=tf.reshape(arc_logits, [-1, num_labels]),
                                           onehot_labels=tf.reshape(parse_target_onehot, [-1, num_labels]),
                                           # tf.reshape(srl_targets_onehot, [-1, num_labels]),
                                           weights=tf.reshape(tokens_to_keep, [-1]),
                                           label_smoothing=hparams.label_smoothing,
                                           reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS)

    # cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=arc_logits, labels=targets)
    # cross_entropy = tf.Print(cross_entropy, [cross_entropy], 'parse_ce_loss')
    # loss = tf.reduce_sum((cross_entropy * tokens_to_keep))/ num_tokens
    # loss = 0.
    # loss = tf.zeros([])
    # loss = tf.Print(loss, [loss], 'parse_bilinear')
    if mode != ModeKeys.TRAIN:
      loss -= loss
    output = {
      'loss': loss,
      'predictions': predictions,
      'probabilities': probabilities,
      'scores': arc_logits,
      'dep_rel_mlp': dep_rel_mlp,
      'head_rel_mlp': head_rel_mlp
    }

  return output


def parse_bilinear_sigmoid(mode, hparams, model_config, inputs, targets, num_labels, tokens_to_keep, transition_params):
  print("using transformed bilinear parser")
  # targets = tf.Print(targets, [targets], "transformed target")
  # print(targets)

  class_mlp_size = model_config['class_mlp_size']
  attn_mlp_size = model_config['attn_mlp_size']

  if transition_params is not None:
    print('Transition params not supported in parse_bilinear')
    exit(1)

  with tf.variable_scope('parse_bilinear_sigmoid'):
    with tf.variable_scope('MLP'):
      dep_mlp, head_mlp = nn_utils.MLP(inputs, class_mlp_size + attn_mlp_size, n_splits=2,
                                       keep_prob=hparams.mlp_dropout)
      dep_arc_mlp, dep_rel_mlp = dep_mlp[:, :, :attn_mlp_size], dep_mlp[:, :, attn_mlp_size:]
      head_arc_mlp, head_rel_mlp = head_mlp[:, :, :attn_mlp_size], head_mlp[:, :, attn_mlp_size:]

    with tf.variable_scope('Arcs'):
      # batch_size x batch_seq_len x batch_seq_len
      arc_logits = nn_utils.bilinear_classifier(dep_arc_mlp, head_arc_mlp, hparams.bilinear_dropout)

    num_tokens = tf.reduce_sum(tokens_to_keep)

    predictions = tf.argmax(arc_logits, -1)
    probabilities = tf.nn.sigmoid(arc_logits)

    # arc_logits = tf.Print(arc_logits, [arc_logits], 'parse_bilinear_arc_logit')
    # targets = tf.Print(targets, [])
    # targets = tf.Print(targets, [targets, tf.reduce_sum(targets)], 'parse_bilinear_targets, checking whether the target vector is passed correctly')
    targets = transformation_fn.one_hot(targets)

    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=arc_logits, labels=targets)
    # cross_entropy = tf.Print(cross_entropy, [cross_entropy], 'parse_ce_loss')
    seq_len = tf.shape(tokens_to_keep)[-1]
    mask_decedents_x = tf.tile(tf.expand_dims(tokens_to_keep, -1), [1, 1, seq_len])
    mask_decedents_y = tf.tile(tf.expand_dims(tokens_to_keep, 1), [1, seq_len, 1])
    mask_decedents = tf.where(
      tf.logical_and(tf.cast(mask_decedents_x, dtype=tf.bool), tf.cast(mask_decedents_y, dtype=tf.bool)),
      tf.ones_like(mask_decedents_x), tf.zeros_like(mask_decedents_x))

    loss = tf.reduce_sum((cross_entropy * mask_decedents)) / num_tokens
    # loss = 0.
    # loss = tf.zeros([])
    # loss = tf.Print(loss, [loss], 'parse_bilinear')
    output = {
      'loss': loss,
      'predictions': predictions,
      'probabilities': probabilities,
      'scores': arc_logits,
      'dep_rel_mlp': dep_rel_mlp,
      'head_rel_mlp': head_rel_mlp
    }

  return output


def parse_aggregation(mode, hparams, model_config, inputs, targets, num_labels, tokens_to_keep, transition_params):
  class_mlp_size = model_config['class_mlp_size']
  attn_mlp_size = model_config['attn_mlp_size']

  if transition_params is not None:
    print('Transition params not supported in parse_aggregation')
    exit(1)

  with tf.variable_scope('parse_bilinear'):
    with tf.variable_scope('MLP'):
      dep_mlp, head_mlp = nn_utils.MLP(inputs, class_mlp_size + attn_mlp_size, n_splits=2,
                                       keep_prob=hparams.mlp_dropout)
      dep_arc_mlp, dep_rel_mlp = dep_mlp[:, :, :attn_mlp_size], dep_mlp[:, :, attn_mlp_size:]
      head_arc_mlp, head_rel_mlp = head_mlp[:, :, :attn_mlp_size], head_mlp[:, :, attn_mlp_size:]

    with tf.variable_scope('Arcs'):
      # batch_size x batch_seq_len x batch_seq_len
      arc_logits = nn_utils.bilinear_classifier(dep_arc_mlp, head_arc_mlp, hparams.bilinear_dropout)

    num_tokens = tf.reduce_sum(tokens_to_keep)

    predictions = tf.argmax(arc_logits, -1)
    probabilities = tf.nn.softmax(arc_logits)

    # arc_logits = tf.Print(arc_logits, [arc_logits], 'parse_bilinear_arc_logit')
    # targets = tf.Print(targets, [])
    # targets = tf.Print(targets, [targets, tf.reduce_sum(targets)], 'parse_bilinear_targets, checking whether the target vector is passed correctly')

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=arc_logits, labels=targets)
    # cross_entropy = tf.Print(cross_entropy, [cross_entropy], 'parse_ce_loss')
    loss = tf.reduce_sum((cross_entropy * tokens_to_keep)) / num_tokens
    # loss = 0.
    # loss = tf.zeros([])
    # loss = tf.Print(loss, [loss], 'parse_bilinear')
    output = {
      'loss': loss,
      'predictions': predictions,
      'probabilities': probabilities,
      'scores': arc_logits,
      'dep_rel_mlp': dep_rel_mlp,
      'head_rel_mlp': head_rel_mlp
    }

  return output


def conditional_bilinear(mode, hparams, model_config, inputs, targets, num_labels, tokens_to_keep, transition_params,
                         dep_rel_mlp, head_rel_mlp, parse_preds_train, parse_preds_eval):
  parse_preds = parse_preds_train if mode == ModeKeys.TRAIN else parse_preds_eval
  with tf.variable_scope('conditional_bilin'):
    logits, per_head_logits = nn_utils.conditional_bilinear_classifier(dep_rel_mlp, head_rel_mlp, num_labels,
                                                                       parse_preds, hparams.bilinear_dropout)

  predictions = tf.argmax(logits, -1)
  probabilities = tf.nn.softmax(logits)

  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=targets)

  n_tokens = tf.reduce_sum(tokens_to_keep)
  loss = tf.reduce_sum(cross_entropy * tokens_to_keep) / n_tokens
  # loss = tf.Print(loss, [loss], "PARSE LABEL LOSS")
  # loss = tf.zeros([])
  output = {
    'loss': loss,
    'scores': logits,
    'predictions': predictions,
    'probabilities': probabilities,
    'per_head_scores': per_head_logits
  }

  return output


def conditional_bilinear_ls(mode, hparams, model_config, inputs, targets, num_labels, tokens_to_keep, transition_params,
                            dep_rel_mlp, head_rel_mlp, parse_preds_train, parse_preds_eval):
  parse_preds = parse_preds_train if mode == ModeKeys.TRAIN else parse_preds_eval
  with tf.variable_scope('conditional_bilin'):
    logits, per_head_logits = nn_utils.conditional_bilinear_classifier(dep_rel_mlp, head_rel_mlp, num_labels,
                                                                       parse_preds, hparams.bilinear_dropout)

  predictions = tf.argmax(logits, -1)
  probabilities = tf.nn.softmax(logits)

  label_target_onehot = tf.one_hot(indices=targets, depth=num_labels, axis=-1)
  loss = tf.losses.softmax_cross_entropy(logits=tf.reshape(logits, [-1, num_labels]),
                                         onehot_labels=tf.reshape(label_target_onehot, [-1, num_labels]),
                                         # tf.reshape(srl_targets_onehot, [-1, num_labels]),
                                         weights=tf.reshape(tokens_to_keep, [-1]),
                                         reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS)
  # loss = tf.Print(loss, [loss], "PARSE LABEL LOSS")
  # cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=targets)

  # n_tokens = tf.reduce_sum(tokens_to_keep)
  # loss = tf.reduce_sum(cross_entropy * tokens_to_keep) / n_tokens
  # loss = tf.zeros([])
  output = {
    'loss': loss,
    'scores': logits,
    'predictions': predictions,
    'probabilities': probabilities,
    'per_head_scores': per_head_logits
  }

  return output


def mean_sentential_feature(mode, hparams, model_config, inputs, tokens_to_keep):
  # suppose input is of form (B, Seq, H) and tokens to keep is of form (B, Seq)
  masked_input = input * tf.expand_dims(tokens_to_keep, -1)
  sent_embedding = tf.reduce_sum(masked_input, axis=1)
  return {
    'embedding': sent_embedding
  }
  raise NotImplementedError


def srl_bilinear(mode, hparams, model_config, inputs, targets, num_labels, tokens_to_keep, predicate_preds_train,
                 predicate_preds_eval, predicate_targets, transition_params):
  '''

    :param input: Tensor with dims: [batch_size, batch_seq_len, hidden_size]
    :param predicate_preds: Tensor of predictions from predicates layer with dims: [batch_size, batch_seq_len]
    :param targets: Tensor of SRL labels with dims: [batch_size, batch_seq_len, batch_num_predicates]
    :param tokens_to_keep:
    :param predictions:
    :param transition_params: [num_labels x num_labels] transition parameters, if doing Viterbi decoding
    :return:
    '''
  # inputs = tf.Print(inputs, [tf.shape(inputs)], "bilinear input size")
  with tf.name_scope('srl_bilinear'):

    def bool_mask_where_predicates(predicates_tensor):
      return tf.logical_and(tf.not_equal(predicates_tensor, predicate_outside_idx), tf.cast(tokens_to_keep, tf.bool))

    input_shape = tf.shape(inputs)
    batch_size = input_shape[0]
    batch_seq_len = input_shape[1]

    predicate_mlp_size = model_config['predicate_mlp_size']
    role_mlp_size = model_config['role_mlp_size']

    # TODO this should really be passed in, not assumed...
    predicate_outside_idx = 0

    predicate_preds = predicate_preds_train if mode == tf.estimator.ModeKeys.TRAIN else predicate_preds_eval
    # predicate_preds = tf.Print(predicate_preds, [predicate_preds], "srl-bilinear predicate prediction")
    # predicate_preds = tf.Print(predicate_preds, [predicate_preds], "predicate_prds", summarize=60)
    predicate_gather_indices = tf.where(bool_mask_where_predicates(predicate_preds))

    # (1) project into predicate, role representations
    if hparams.bilstm:
      with tf.variable_scope("BILSTM"):
        forward_layer = tf.keras.layers.LSTM(400, return_sequences=True, dropout=0.1, recurrent_dropout=0.1)
        bidirectional = tf.keras.layers.Bidirectional(forward_layer, merge_mode='concat', swap_memory=True)
        inputs = bidirectional(inputs, mask=tokens_to_keep)
        tf.logging.log(tf.logging.INFO, "bidirectional output {}".format(inputs))

    with tf.variable_scope('MLP'):
      predicate_role_mlp = nn_utils.MLP(inputs, predicate_mlp_size + role_mlp_size, keep_prob=hparams.mlp_dropout)
      predicate_mlp, role_mlp = predicate_role_mlp[:, :, :predicate_mlp_size], \
                                predicate_role_mlp[:, :, predicate_mlp_size:]

    # (2) feed through bilinear to obtain scores
    with tf.variable_scope('Bilinear'):
      # gather just the predicates
      # gathered_predicates: num_predicates_in_batch x 1 x predicate_mlp_size
      # role mlp: batch x seq_len x role_mlp_size
      # gathered roles: need a (batch_seq_len x role_mlp_size) role representation for each predicate,
      # i.e. a (num_predicates_in_batch x batch_seq_len x role_mlp_size) tensor
      gathered_predicates = tf.expand_dims(tf.gather_nd(predicate_mlp, predicate_gather_indices), 1)
      tiled_roles = tf.reshape(tf.tile(role_mlp, [1, batch_seq_len, 1]),
                               [batch_size, batch_seq_len, batch_seq_len, role_mlp_size])
      gathered_roles = tf.gather_nd(tiled_roles, predicate_gather_indices)
      # now multiply them together to get (num_predicates_in_batch x batch_seq_len x num_srl_classes) tensor of scores
      srl_logits = nn_utils.bilinear_classifier_nary(gathered_predicates, gathered_roles, num_labels,
                                                     hparams.bilinear_dropout)
      srl_logits_transposed = tf.transpose(srl_logits, [0, 2, 1])

    # (3) compute loss

    # need to repeat each of these once for each target in the sentence
    mask_tiled = tf.reshape(tf.tile(tokens_to_keep, [1, batch_seq_len]), [batch_size, batch_seq_len, batch_seq_len])
    mask = tf.gather_nd(mask_tiled, predicate_gather_indices)

    # now we have k sets of targets for the k frames
    # (p1) f1 f2 f3
    # (p2) f1 f2 f3

    # get all the tags for each token (which is the predicate for a frame), structuring
    # targets as follows (assuming p1 and p2 are predicates for f1 and f3, respectively):
    # (p1) f1 f1 f1
    # (p2) f3 f3 f3
    srl_targets_transposed = tf.transpose(targets, [0, 2, 1])

    gold_predicate_counts = tf.reduce_sum(tf.cast(bool_mask_where_predicates(predicate_targets), tf.int32), -1)
    srl_targets_indices = tf.where(tf.sequence_mask(tf.reshape(gold_predicate_counts, [-1])))

    # num_predicates_in_batch x seq_len
    srl_targets_gold_predicates = tf.gather_nd(srl_targets_transposed, srl_targets_indices)

    predicted_predicate_counts = tf.reduce_sum(tf.cast(bool_mask_where_predicates(predicate_preds), tf.int32), -1)
    srl_targets_pred_indices = tf.where(tf.sequence_mask(tf.reshape(predicted_predicate_counts, [-1])))
    srl_targets_predicted_predicates = tf.gather_nd(srl_targets_transposed, srl_targets_pred_indices)

    # num_predicates_in_batch x seq_len
    predictions = tf.cast(tf.argmax(srl_logits_transposed, axis=-1), tf.int32)

    seq_lens = tf.cast(tf.reduce_sum(mask, 1), tf.int32)

    if transition_params is not None and (mode == ModeKeys.PREDICT or mode == ModeKeys.EVAL):
      predictions, score = tf.contrib.crf.crf_decode(srl_logits_transposed, transition_params, seq_lens)
      log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(tf.stop_gradient(srl_logits_transposed),
                                                                            srl_targets_predicted_predicates,
                                                                            seq_lens,
                                                                            tf.stop_gradient(transition_params))
      loss = tf.reduce_mean(-log_likelihood)

    if transition_params is not None and mode == ModeKeys.TRAIN:  # and tf_utils.is_trainable(transition_params):
      # flat_seq_lens = tf.reshape(tf.tile(seq_lens, [1, bucket_size]), tf.stack([batch_size * bucket_size]))
      if tf_utils.is_trainable(transition_params):
        tf.logging.log(tf.logging.INFO, "Using CRF + CE LOSS")
        srl_targets_onehot = tf.one_hot(indices=srl_targets_predicted_predicates, depth=num_labels, axis=-1)
        loss_emission = tf.losses.softmax_cross_entropy(logits=tf.reshape(srl_logits_transposed, [-1, num_labels]),
                                                        onehot_labels=tf.reshape(srl_targets_onehot,
                                                                                 [-1, num_labels]),
                                                        weights=tf.reshape(mask, [-1]),
                                                        label_smoothing=hparams.label_smoothing,
                                                        reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS)
        log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(tf.stop_gradient(srl_logits_transposed),
                                                                              srl_targets_predicted_predicates,
                                                                              seq_lens, transition_params)

        loss = loss_emission - tf.reduce_mean(log_likelihood)
      else:
        tf.logging.log(tf.logging.INFO, "Using CE LOSS")
        srl_targets_onehot = tf.one_hot(indices=srl_targets_predicted_predicates, depth=num_labels, axis=-1)
        loss_emission = tf.losses.softmax_cross_entropy(logits=tf.reshape(srl_logits_transposed, [-1, num_labels]),
                                                        onehot_labels=tf.reshape(srl_targets_onehot, [-1, num_labels]),
                                                        weights=tf.reshape(mask, [-1]),
                                                        label_smoothing=hparams.label_smoothing,
                                                        reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS)
        # log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(tf.stop_gradient(srl_logits_transposed),
        #                                                                       srl_targets_predicted_predicates,
        #                                                                       seq_lens, transition_params)
        loss = loss_emission  # - tf.reduce_mean(log_likelihood)
    if transition_params is None and mode == ModeKeys.TRAIN:
      tf.logging.log(tf.logging.INFO, "Using CE LOSS, with no transition param, Training mode")
      srl_targets_onehot = tf.one_hot(indices=srl_targets_predicted_predicates, depth=num_labels, axis=-1)
      loss_emission = tf.losses.softmax_cross_entropy(logits=tf.reshape(srl_logits_transposed, [-1, num_labels]),
                                                      onehot_labels=tf.reshape(srl_targets_onehot, [-1, num_labels]),
                                                      weights=tf.reshape(mask, [-1]),
                                                      label_smoothing=hparams.label_smoothing,
                                                      reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS)
      loss = loss_emission  # - tf.reduce_mean(log_likelihood)\
      # loss = tf.Print(loss, [loss], "CE LOSS")
    if transition_params is None and (mode == ModeKeys.PREDICT or mode == ModeKeys.EVAL):
      tf.logging.log(tf.logging.INFO, "Using CE LOSS, with no transition param, Pred mode")
      srl_targets_onehot = tf.one_hot(indices=srl_targets_predicted_predicates, depth=num_labels, axis=-1)
      loss_emission = tf.losses.softmax_cross_entropy(logits=tf.reshape(srl_logits_transposed, [-1, num_labels]),
                                                      onehot_labels=tf.reshape(srl_targets_onehot, [-1, num_labels]),
                                                      weights=tf.reshape(mask, [-1]),
                                                      # label_smoothing=hparams.label_smoothing,
                                                      reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS)
      loss = loss_emission  # - tf.reduce_mean(log_likelihood)
      predictions = tf.argmax(srl_logits_transposed, axis=-1)

    # if transition_params is not None and mode == ModeKeys.TRAIN: #and hparams.train_with_crf:
    # flat_seq_lens = tf.reshape(tf.tile(seq_lens, [1, bucket_size]), tf.stack([batch_size * bucket_size]))

    # loss = tf.reduce_mean(-log_likelihood)
    # else:

    output = {
      'loss': loss,
      'predictions': predictions,
      'scores': srl_logits_transposed,
      'targets': srl_targets_gold_predicates,
      'probabilities': tf.nn.softmax(srl_logits_transposed, -1)
    }

    return output


def srl_bilinear_sm(mode, hparams, model_config, inputs, targets, num_labels, tokens_to_keep, predicate_preds_train,
                    predicate_preds_eval, predicate_targets, transition_params):
  '''

  :param input: Tensor with dims: [batch_size, batch_seq_len, hidden_size]
  :param predicate_preds: Tensor of predictions from predicates layer with dims: [batch_size, batch_seq_len]
  :param targets: Tensor of SRL labels with dims: [batch_size, batch_seq_len, batch_num_predicates]
  :param tokens_to_keep:
  :param predictions:
  :param transition_params: [num_labels x num_labels] transition parameters, if doing Viterbi decoding
  :return:
  '''
  # inputs = tf.Print(inputs, [tf.shape(inputs)], "bilinear input size")
  with tf.name_scope('srl_bilinear'):

    def bool_mask_where_predicates(predicates_tensor):
      return tf.logical_and(tf.not_equal(predicates_tensor, predicate_outside_idx), tf.cast(tokens_to_keep, tf.bool))

    input_shape = tf.shape(inputs)
    batch_size = input_shape[0]
    batch_seq_len = input_shape[1]

    predicate_mlp_size = model_config['predicate_mlp_size']
    role_mlp_size = model_config['role_mlp_size']

    # TODO this should really be passed in, not assumed...
    predicate_outside_idx = 0

    predicate_preds = predicate_preds_train if mode == tf.estimator.ModeKeys.TRAIN else predicate_preds_eval
    # predicate_preds = tf.Print(predicate_preds, [predicate_preds], "srl-bilinear predicate prediction")
    # predicate_preds = tf.Print(predicate_preds, [predicate_preds], "predicate_prds", summarize=60)
    predicate_gather_indices = tf.where(bool_mask_where_predicates(predicate_preds))

    # (1) project into predicate, role representations

    with tf.variable_scope('MLP'):
      predicate_role_mlp = nn_utils.MLP(inputs, predicate_mlp_size + role_mlp_size, keep_prob=hparams.mlp_dropout)
      predicate_mlp, role_mlp = predicate_role_mlp[:, :, :predicate_mlp_size], \
                                predicate_role_mlp[:, :, predicate_mlp_size:]

    # (2) feed through bilinear to obtain scores
    with tf.variable_scope('Bilinear'):
      # gather just the predicates
      # gathered_predicates: num_predicates_in_batch x 1 x predicate_mlp_size
      # role mlp: batch x seq_len x role_mlp_size
      # gathered roles: need a (batch_seq_len x role_mlp_size) role representation for each predicate,
      # i.e. a (num_predicates_in_batch x batch_seq_len x role_mlp_size) tensor
      gathered_predicates = tf.expand_dims(tf.gather_nd(predicate_mlp, predicate_gather_indices), 1)
      tiled_roles = tf.reshape(tf.tile(role_mlp, [1, batch_seq_len, 1]),
                               [batch_size, batch_seq_len, batch_seq_len, role_mlp_size])
      gathered_roles = tf.gather_nd(tiled_roles, predicate_gather_indices)
      # now multiply them together to get (num_predicates_in_batch x batch_seq_len x num_srl_classes) tensor of scores
      srl_logits = nn_utils.bilinear_classifier_nary(gathered_predicates, gathered_roles, num_labels,
                                                     hparams.bilinear_dropout)
      srl_logits_transposed = tf.transpose(srl_logits, [0, 2, 1])

    # (3) compute loss

    # need to repeat each of these once for each target in the sentence
    mask_tiled = tf.reshape(tf.tile(tokens_to_keep, [1, batch_seq_len]), [batch_size, batch_seq_len, batch_seq_len])
    mask = tf.gather_nd(mask_tiled, predicate_gather_indices)

    # now we have k sets of targets for the k frames
    # (p1) f1 f2 f3
    # (p2) f1 f2 f3

    # get all the tags for each token (which is the predicate for a frame), structuring
    # targets as follows (assuming p1 and p2 are predicates for f1 and f3, respectively):
    # (p1) f1 f1 f1
    # (p2) f3 f3 f3
    srl_targets_transposed = tf.transpose(targets, [0, 2, 1])

    gold_predicate_counts = tf.reduce_sum(tf.cast(bool_mask_where_predicates(predicate_targets), tf.int32), -1)
    srl_targets_indices = tf.where(tf.sequence_mask(tf.reshape(gold_predicate_counts, [-1])))

    # num_predicates_in_batch x seq_len
    srl_targets_gold_predicates = tf.gather_nd(srl_targets_transposed, srl_targets_indices)

    predicted_predicate_counts = tf.reduce_sum(tf.cast(bool_mask_where_predicates(predicate_preds), tf.int32), -1)
    srl_targets_pred_indices = tf.where(tf.sequence_mask(tf.reshape(predicted_predicate_counts, [-1])))
    srl_targets_predicted_predicates = tf.gather_nd(srl_targets_transposed, srl_targets_pred_indices)

    # num_predicates_in_batch x seq_len
    predictions = tf.cast(tf.argmax(srl_logits_transposed, axis=-1), tf.int32)

    seq_lens = tf.cast(tf.reduce_sum(mask, 1), tf.int32)

    if mode == ModeKeys.PREDICT or mode == ModeKeys.EVAL:
      srl_targets_onehot = tf.one_hot(indices=srl_targets_predicted_predicates, depth=num_labels, axis=-1)
      loss_emission = tf.losses.softmax_cross_entropy(logits=tf.reshape(srl_logits_transposed, [-1, num_labels]),
                                                      onehot_labels=tf.reshape(srl_targets_onehot, [-1, num_labels]),
                                                      weights=tf.reshape(mask, [-1]),
                                                      label_smoothing=hparams.label_smoothing,
                                                      reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS)
      predictions = tf.argmax(srl_logits_transposed, axis=-1)
      loss = loss_emission
    else:
      assert mode == ModeKeys.TRAIN
      srl_targets_onehot = tf.one_hot(indices=srl_targets_predicted_predicates, depth=num_labels, axis=-1)
      loss_emission = tf.losses.softmax_cross_entropy(logits=tf.reshape(srl_logits_transposed, [-1, num_labels]),
                                                      onehot_labels=tf.reshape(srl_targets_onehot, [-1, num_labels]),
                                                      weights=tf.reshape(mask, [-1]),
                                                      label_smoothing=hparams.label_smoothing,
                                                      reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS)
      loss = loss_emission

    output = {
      'loss': loss,
      'predictions': predictions,
      'scores': srl_logits_transposed,
      'targets': srl_targets_gold_predicates,
      'probabilities': tf.nn.softmax(srl_logits_transposed, -1)
    }

    return output


def srl_bilinear_dep_prior(mode, hparams, model_config, inputs, targets, num_labels, tokens_to_keep,
                           predicate_preds_train,
                           predicate_preds_eval, predicate_targets, parse_label_predictions, parse_label_targets,
                           parse_head_predictions, parse_head_targets, transition_params):
  '''

  :param input: Tensor with dims: [batch_size, batch_seq_len, hidden_size]
  :param predicate_preds: Tensor of predictions from predicates layer with dims: [batch_size, batch_seq_len]
  :param targets: Tensor of SRL labels with dims: [batch_size, batch_seq_len, batch_num_predicates]
  :param tokens_to_keep:
  :param predictions:
  :param transition_params: [num_labels x num_labels] transition parameters, if doing Viterbi decoding
  :return:
  '''
  # This function includes priors inferred from dependency graph

  if mode == ModeKeys.TRAIN:
    parse_gold = parse_head_targets
    parse_label = parse_label_targets
  else:
    parse_gold = parse_head_predictions
    parse_label = parse_label_predictions
  with tf.name_scope('srl_bilinear'):

    def bool_mask_where_predicates(predicates_tensor):
      return tf.logical_and(tf.not_equal(predicates_tensor, predicate_outside_idx), tf.cast(tokens_to_keep, tf.bool))

    input_shape = tf.shape(inputs)
    batch_size = input_shape[0]
    batch_seq_len = input_shape[1]

    predicate_mlp_size = model_config['predicate_mlp_size']
    role_mlp_size = model_config['role_mlp_size']

    # TODO this should really be passed in, not assumed...
    predicate_outside_idx = 0

    predicate_preds = predicate_preds_train if mode == tf.estimator.ModeKeys.TRAIN else predicate_preds_eval
    # predicate_preds = tf.Print(predicate_preds, [predicate_preds], "srl-bilinear predicate prediction")
    # predicate_preds = tf.Print(predicate_preds, [predicate_preds], "predicate_prds", summarize=60)
    predicate_gather_indices = tf.where(bool_mask_where_predicates(predicate_preds))

    # (1) project into predicate, role representations
    with tf.variable_scope('MLP'):
      predicate_role_mlp = nn_utils.MLP(inputs, predicate_mlp_size + role_mlp_size, keep_prob=hparams.mlp_dropout)
      predicate_mlp, role_mlp = predicate_role_mlp[:, :, :predicate_mlp_size], \
                                predicate_role_mlp[:, :, predicate_mlp_size:]

    # (2) feed through bilinear to obtain scores
    with tf.variable_scope('Bilinear'):
      # gather just the predicates
      # gathered_predicates: num_predicates_in_batch x 1 x predicate_mlp_size
      # role mlp: batch x seq_len x role_mlp_size
      # gathered roles: need a (batch_seq_len x role_mlp_size) role representation for each predicate,
      # i.e. a (num_predicates_in_batch x batch_seq_len x role_mlp_size) tensor
      gathered_predicates = tf.expand_dims(tf.gather_nd(predicate_mlp, predicate_gather_indices), 1)
      tiled_roles = tf.reshape(tf.tile(role_mlp, [1, batch_seq_len, 1]),
                               [batch_size, batch_seq_len, batch_seq_len, role_mlp_size])
      gathered_roles = tf.gather_nd(tiled_roles, predicate_gather_indices)
      # now multiply them together to get (num_predicates_in_batch x batch_seq_len x num_srl_classes) tensor of scores
      srl_logits = nn_utils.bilinear_classifier_nary(gathered_predicates, gathered_roles, num_labels,
                                                     hparams.bilinear_dropout)
      srl_logits_transposed = tf.transpose(srl_logits, [0, 2, 1])
      if hparams.srl_layernorm:
        with tf.variable_scope('LayerNorm'):
          srl_logits_transposed = tf.contrib.layers.layer_norm(srl_logits_transposed)

      # Here gather (num_predicates_in_batch, seq_len, num_classes) dep prior

      # Do not include _ tag because we are doing addressing
      prior_list = []
      if hparams.one_down:
        with tf.variable_scope("1down_addressing"):
          if not hparams.relu_implementation:
            if hparams.elevate_underline:
              dep_prior = get_lprior_up_mtx(parse_gold, parse_label, num_labels, tokens_to_keep=tokens_to_keep,
                                            extreme_value=False, layer_norm_to_heads=hparams.layer_norm_to_heads,
                                            transpose=True, memory_efficient=hparams.memory_efficient,
                                            joint_par_srl_training=hparams.joint_par_srl_training, relu_imp=False)
            else:
              dep_prior = get_lprior_up_mtx(parse_gold, parse_label, num_labels - 1, tokens_to_keep=tokens_to_keep,
                                            extreme_value=False, layer_norm_to_heads=hparams.layer_norm_to_heads,
                                            transpose=True, memory_efficient=hparams.memory_efficient,
                                            joint_par_srl_training=hparams.joint_par_srl_training, relu_imp=False)
              sdp = tf.shape(dep_prior)
              dep_prior_empty_filler = tf.fill([sdp[0], sdp[1], sdp[2], 1], 0.)
              dep_prior = tf.concat([dep_prior_empty_filler, dep_prior], axis=-1)
            dep_prior_gathered = tf.gather_nd(dep_prior, predicate_gather_indices)
          else:
            dep_prior = get_lprior_up_mtx(parse_gold, parse_label, num_labels, tokens_to_keep=tokens_to_keep,
                                          extreme_value=False, layer_norm_to_heads=hparams.layer_norm_to_heads,
                                          transpose=True, memory_efficient=hparams.memory_efficient,
                                          joint_par_srl_training=hparams.joint_par_srl_training, relu_imp=True)
            dep_prior_gathered = tf.gather_nd(dep_prior, predicate_gather_indices)
          if hparams.srl_layernorm:
            with tf.variable_scope('LayerNorm'):
              dep_prior_gathered = tf.contrib.layers.layer_norm(dep_prior_gathered)
          prior_list.append(dep_prior_gathered)

      if hparams.one_up:
        with tf.variable_scope("1up_addressing"):
          if not hparams.relu_implementation:
            if hparams.elevate_underline:
              dep_prior = get_lprior_up_mtx(parse_gold, parse_label, num_labels, tokens_to_keep=tokens_to_keep,
                                            extreme_value=False, layer_norm_to_heads=hparams.layer_norm_to_heads,
                                            transpose=False, memory_efficient=hparams.memory_efficient,
                                            joint_par_srl_training=hparams.joint_par_srl_training, relu_imp=False)
            else:
              dep_prior = get_lprior_up_mtx(parse_gold, parse_label, num_labels - 1, tokens_to_keep=tokens_to_keep,
                                            extreme_value=False, layer_norm_to_heads=hparams.layer_norm_to_heads,
                                            transpose=False, memory_efficient=hparams.memory_efficient,
                                            joint_par_srl_training=hparams.joint_par_srl_training, relu_imp=False)
              sdp = tf.shape(dep_prior)
              dep_prior_empty_filler = tf.fill([sdp[0], sdp[1], sdp[2], 1], 0.)
              dep_prior = tf.concat([dep_prior_empty_filler, dep_prior], axis=-1)
            dep_prior_gathered = tf.gather_nd(dep_prior, predicate_gather_indices)
          else:
            dep_prior = get_lprior_up_mtx(parse_gold, parse_label, num_labels, tokens_to_keep=tokens_to_keep,
                                          extreme_value=False, layer_norm_to_heads=hparams.layer_norm_to_heads,
                                          transpose=False, memory_efficient=hparams.memory_efficient,
                                          joint_par_srl_training=hparams.joint_par_srl_training, relu_imp=True)
            dep_prior_gathered = tf.gather_nd(dep_prior, predicate_gather_indices)
          if hparams.srl_layernorm:
            with tf.variable_scope('LayerNorm'):
              dep_prior_gathered = tf.contrib.layers.layer_norm(dep_prior_gathered)
          prior_list.append(dep_prior_gathered)
      # Now we get a tensor of shape (batch, seq, seq, srl_labels)

      if hparams.kup1down_up_to > 0:
        for k in range(hparams.kup1down_up_to):
          with tf.variable_scope("{}up1down_addressing".format(k + 1)):
            if not hparams.relu_implementation:
              if hparams.elevate_underline:
                dep_prior_ku1d = get_lprior_kup1down_mtx(parse_gold, parse_label, num_labels,
                                                         tokens_to_keep=tokens_to_keep,
                                                         extreme_value=False,
                                                         layer_norm_to_heads=hparams.layer_norm_to_heads, k=k + 1,
                                                         memory_efficient=hparams.memory_efficient,
                                                         joint_par_srl_training=hparams.joint_par_srl_training,
                                                         relu_imp=False)
              else:
                dep_prior_ku1d = get_lprior_kup1down_mtx(parse_gold, parse_label, num_labels - 1,
                                                         tokens_to_keep=tokens_to_keep,
                                                         extreme_value=False,
                                                         layer_norm_to_heads=hparams.layer_norm_to_heads, k=k + 1,
                                                         memory_efficient=hparams.memory_efficient,
                                                         joint_par_srl_training=hparams.joint_par_srl_training,
                                                         relu_imp=False)
                sdp = tf.shape(dep_prior_ku1d)
                dep_prior_ku1d_empty_filler = tf.fill([sdp[0], sdp[1], sdp[2], 1], 0.)
                dep_prior_ku1d = tf.concat([dep_prior_ku1d_empty_filler, dep_prior_ku1d], axis=-1)
              dep_prior_ku1d_gathered = tf.gather_nd(dep_prior_ku1d, predicate_gather_indices)
            else:
              dep_prior_ku1d = get_lprior_kup1down_mtx(parse_gold, parse_label, num_labels,
                                                       tokens_to_keep=tokens_to_keep,
                                                       extreme_value=False,
                                                       layer_norm_to_heads=hparams.layer_norm_to_heads, k=k + 1,
                                                       memory_efficient=hparams.memory_efficient,
                                                       joint_par_srl_training=hparams.joint_par_srl_training,
                                                       relu_imp=True)
              dep_prior_ku1d_gathered = tf.gather_nd(dep_prior_ku1d, predicate_gather_indices)
            if hparams.srl_layernorm:
              with tf.variable_scope('LayerNorm'):
                dep_prior_ku1d_gathered = tf.contrib.layers.layer_norm(dep_prior_ku1d_gathered)
            prior_list.append(dep_prior_ku1d_gathered)

      tf.logging.log(tf.logging.INFO, "dep prior: {}".format(dep_prior))

      tf.logging.log(tf.logging.INFO, "gathered dep prior gather indices: {}".format(predicate_gather_indices))
      tf.logging.log(tf.logging.INFO, "gathered dep prior: {}".format(dep_prior_gathered))
      if not hparams.weight_per_label:
        lbda = tf.get_variable("dep_prior_weight", [len(prior_list), 1, 1, 1], initializer=tf.ones_initializer,
                               trainable=hparams.dep_prior_trainable)
      else:
        lbda = tf.get_variable("dep_prior_weight", [len(prior_list), 1, 1, num_labels],
                               initializer=tf.ones_initializer,
                               trainable=hparams.dep_prior_trainable)
      # NOTE:!! The print op here was lbda * dep_prior_gathered -> mis-use of boardcasting
      lbda = tf.Print(lbda, [lbda, (lbda * tf.stack(prior_list, axis=0))[:, :, :, :5]], "dep_prior_weight",
                      summarize=40)

      srl_logits_transposed += tf.reduce_sum(lbda * tf.stack(prior_list, axis=0), axis=0)
      srl_logits_transposed = tf.Print(srl_logits_transposed, [srl_logits_transposed[:, :, :5]],
                                       "srl_logits_transposed", summarize=40)

    # (3) compute loss

    # need to repeat each of these once for each target in the sentence
    mask_tiled = tf.reshape(tf.tile(tokens_to_keep, [1, batch_seq_len]), [batch_size, batch_seq_len, batch_seq_len])
    mask = tf.gather_nd(mask_tiled, predicate_gather_indices)

    # now we have k sets of targets for the k frames
    # (p1) f1 f2 f3
    # (p2) f1 f2 f3

    # get all the tags for each token (which is the predicate for a frame), structuring
    # targets as follows (assuming p1 and p2 are predicates for f1 and f3, respectively):
    # (p1) f1 f1 f1
    # (p2) f3 f3 f3
    srl_targets_transposed = tf.transpose(targets, [0, 2, 1])

    gold_predicate_counts = tf.reduce_sum(tf.cast(bool_mask_where_predicates(predicate_targets), tf.int32), -1)
    srl_targets_indices = tf.where(tf.sequence_mask(tf.reshape(gold_predicate_counts, [-1])))

    # num_predicates_in_batch x seq_len
    srl_targets_gold_predicates = tf.gather_nd(srl_targets_transposed, srl_targets_indices)

    predicted_predicate_counts = tf.reduce_sum(tf.cast(bool_mask_where_predicates(predicate_preds), tf.int32), -1)
    srl_targets_pred_indices = tf.where(tf.sequence_mask(tf.reshape(predicted_predicate_counts, [-1])))
    srl_targets_predicted_predicates = tf.gather_nd(srl_targets_transposed, srl_targets_pred_indices)

    # num_predicates_in_batch x seq_len
    predictions = tf.cast(tf.argmax(srl_logits_transposed, axis=-1), tf.int32)

    seq_lens = tf.cast(tf.reduce_sum(mask, 1), tf.int32)

    if (mode == ModeKeys.PREDICT or mode == ModeKeys.EVAL):
      srl_targets_onehot = tf.one_hot(indices=srl_targets_predicted_predicates, depth=num_labels, axis=-1)
      loss_emission = tf.losses.softmax_cross_entropy(logits=tf.reshape(srl_logits_transposed, [-1, num_labels]),
                                                      onehot_labels=tf.reshape(srl_targets_onehot, [-1, num_labels]),
                                                      weights=tf.reshape(mask, [-1]),
                                                      label_smoothing=hparams.label_smoothing,
                                                      reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS)

      loss = loss_emission
      predictions = tf.argmax(srl_logits_transposed, axis=-1)

    if mode == ModeKeys.TRAIN:  # and tf_utils.is_trainable(transition_params):
      # flat_seq_lens = tf.reshape(tf.tile(seq_lens, [1, bucket_size]), tf.stack([batch_size * bucket_size]))
      srl_targets_onehot = tf.one_hot(indices=srl_targets_predicted_predicates, depth=num_labels, axis=-1)
      loss_emission = tf.losses.softmax_cross_entropy(logits=tf.reshape(srl_logits_transposed, [-1, num_labels]),
                                                      onehot_labels=tf.reshape(srl_targets_onehot, [-1, num_labels]),
                                                      weights=tf.reshape(mask, [-1]),
                                                      label_smoothing=hparams.label_smoothing,
                                                      reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS)

      loss = loss_emission

    output = {
      'loss': loss,
      'predictions': predictions,
      'scores': srl_logits_transposed,
      'targets': srl_targets_gold_predicates,
      'probabilities': tf.nn.softmax(srl_logits_transposed, -1)
    }

    return output


def srl_bilinear_dep_prior_pos(mode, hparams, model_config, inputs, targets, num_labels, tokens_to_keep,
                               predicate_preds_train,
                               predicate_preds_eval, predicate_targets, parse_label_predictions, parse_label_targets,
                               parse_head_predictions, parse_head_targets, pos_predictions, pos_targets,
                               transition_params):
  '''

  :param input: Tensor with dims: [batch_size, batch_seq_len, hidden_size]
  :param predicate_preds: Tensor of predictions from predicates layer with dims: [batch_size, batch_seq_len]
  :param targets: Tensor of SRL labels with dims: [batch_size, batch_seq_len, batch_num_predicates]
  :param tokens_to_keep:
  :param predictions:
  :param transition_params: [num_labels x num_labels] transition parameters, if doing Viterbi decoding
  :return:
  '''
  # This function includes priors inferred from dependency graph

  if mode == ModeKeys.TRAIN:
    parse_gold = parse_head_targets
    parse_label = parse_label_targets
    pos_tag = pos_targets
  else:
    parse_gold = parse_head_predictions
    parse_label = parse_label_predictions
    pos_tag = pos_predictions
  with tf.name_scope('srl_bilinear'):

    def bool_mask_where_predicates(predicates_tensor):
      return tf.logical_and(tf.not_equal(predicates_tensor, predicate_outside_idx), tf.cast(tokens_to_keep, tf.bool))

    input_shape = tf.shape(inputs)
    batch_size = input_shape[0]
    batch_seq_len = input_shape[1]

    predicate_mlp_size = model_config['predicate_mlp_size']
    role_mlp_size = model_config['role_mlp_size']

    # TODO this should really be passed in, not assumed...
    predicate_outside_idx = 0

    predicate_preds = predicate_preds_train if mode == tf.estimator.ModeKeys.TRAIN else predicate_preds_eval
    # predicate_preds = tf.Print(predicate_preds, [predicate_preds], "srl-bilinear predicate prediction")
    # predicate_preds = tf.Print(predicate_preds, [predicate_preds], "predicate_prds", summarize=60)
    predicate_gather_indices = tf.where(bool_mask_where_predicates(predicate_preds))

    # (1) project into predicate, role representations
    with tf.variable_scope('MLP'):
      predicate_role_mlp = nn_utils.MLP(inputs, predicate_mlp_size + role_mlp_size, keep_prob=hparams.mlp_dropout)
      predicate_mlp, role_mlp = predicate_role_mlp[:, :, :predicate_mlp_size], \
                                predicate_role_mlp[:, :, predicate_mlp_size:]

    # (2) feed through bilinear to obtain scores
    with tf.variable_scope('Bilinear'):
      # gather just the predicates
      # gathered_predicates: num_predicates_in_batch x 1 x predicate_mlp_size
      # role mlp: batch x seq_len x role_mlp_size
      # gathered roles: need a (batch_seq_len x role_mlp_size) role representation for each predicate,
      # i.e. a (num_predicates_in_batch x batch_seq_len x role_mlp_size) tensor
      gathered_predicates = tf.expand_dims(tf.gather_nd(predicate_mlp, predicate_gather_indices), 1)
      tiled_roles = tf.reshape(tf.tile(role_mlp, [1, batch_seq_len, 1]),
                               [batch_size, batch_seq_len, batch_seq_len, role_mlp_size])
      gathered_roles = tf.gather_nd(tiled_roles, predicate_gather_indices)
      # now multiply them together to get (num_predicates_in_batch x batch_seq_len x num_srl_classes) tensor of scores
      srl_logits = nn_utils.bilinear_classifier_nary(gathered_predicates, gathered_roles, num_labels,
                                                     hparams.bilinear_dropout)
      srl_logits_transposed = tf.transpose(srl_logits, [0, 2, 1])
      if hparams.srl_layernorm:
        with tf.variable_scope('LayerNorm'):
          srl_logits_transposed = tf.contrib.layers.layer_norm(srl_logits_transposed)

      # Here gather (num_predicates_in_batch, seq_len, num_classes) dep prior

      # Do not include _ tag because we are doing addressing
      prior_list = []
      if hparams.one_down:
        with tf.variable_scope("1down_addressing"):
          if not hparams.relu_implementation:
            if hparams.elevate_underline:
              dep_prior = get_lprior_up_mtx(parse_gold, parse_label, num_labels, tokens_to_keep=tokens_to_keep,
                                            extreme_value=False, layer_norm_to_heads=hparams.layer_norm_to_heads,
                                            transpose=True, memory_efficient=hparams.memory_efficient,
                                            joint_par_srl_training=hparams.joint_par_srl_training, relu_imp=False,
                                            pos_tag=pos_tag if hparams.use_pos_tag else None)
            else:
              dep_prior = get_lprior_up_mtx(parse_gold, parse_label, num_labels - 1, tokens_to_keep=tokens_to_keep,
                                            extreme_value=False, layer_norm_to_heads=hparams.layer_norm_to_heads,
                                            transpose=True, memory_efficient=hparams.memory_efficient,
                                            joint_par_srl_training=hparams.joint_par_srl_training, relu_imp=False,
                                            pos_tag=pos_tag if hparams.use_pos_tag else None)
              sdp = tf.shape(dep_prior)
              dep_prior_empty_filler = tf.fill([sdp[0], sdp[1], sdp[2], 1], 0.)
              dep_prior = tf.concat([dep_prior_empty_filler, dep_prior], axis=-1)
            dep_prior_gathered = tf.gather_nd(dep_prior, predicate_gather_indices)
          else:
            dep_prior = get_lprior_up_mtx(parse_gold, parse_label, num_labels, tokens_to_keep=tokens_to_keep,
                                          extreme_value=False, layer_norm_to_heads=hparams.layer_norm_to_heads,
                                          transpose=True, memory_efficient=hparams.memory_efficient,
                                          joint_par_srl_training=hparams.joint_par_srl_training, relu_imp=True,
                                          pos_tag=pos_tag if hparams.use_pos_tag else None)
            dep_prior_gathered = tf.gather_nd(dep_prior, predicate_gather_indices)
          if hparams.srl_layernorm:
            with tf.variable_scope('LayerNorm'):
              dep_prior_gathered = tf.contrib.layers.layer_norm(dep_prior_gathered)
          prior_list.append(dep_prior_gathered)

      if hparams.one_up:
        with tf.variable_scope("1up_addressing"):
          if not hparams.relu_implementation:
            if hparams.elevate_underline:
              dep_prior = get_lprior_up_mtx(parse_gold, parse_label, num_labels, tokens_to_keep=tokens_to_keep,
                                            extreme_value=False, layer_norm_to_heads=hparams.layer_norm_to_heads,
                                            transpose=False, memory_efficient=hparams.memory_efficient,
                                            joint_par_srl_training=hparams.joint_par_srl_training, relu_imp=False)
            else:
              dep_prior = get_lprior_up_mtx(parse_gold, parse_label, num_labels - 1, tokens_to_keep=tokens_to_keep,
                                            extreme_value=False, layer_norm_to_heads=hparams.layer_norm_to_heads,
                                            transpose=False, memory_efficient=hparams.memory_efficient,
                                            joint_par_srl_training=hparams.joint_par_srl_training, relu_imp=False)
              sdp = tf.shape(dep_prior)
              dep_prior_empty_filler = tf.fill([sdp[0], sdp[1], sdp[2], 1], 0.)
              dep_prior = tf.concat([dep_prior_empty_filler, dep_prior], axis=-1)
            dep_prior_gathered = tf.gather_nd(dep_prior, predicate_gather_indices)
          else:
            dep_prior = get_lprior_up_mtx(parse_gold, parse_label, num_labels, tokens_to_keep=tokens_to_keep,
                                          extreme_value=False, layer_norm_to_heads=hparams.layer_norm_to_heads,
                                          transpose=False, memory_efficient=hparams.memory_efficient,
                                          joint_par_srl_training=hparams.joint_par_srl_training, relu_imp=True)
            dep_prior_gathered = tf.gather_nd(dep_prior, predicate_gather_indices)
          if hparams.srl_layernorm:
            with tf.variable_scope('LayerNorm'):
              dep_prior_gathered = tf.contrib.layers.layer_norm(dep_prior_gathered)
          prior_list.append(dep_prior_gathered)
      # Now we get a tensor of shape (batch, seq, seq, srl_labels)

      if hparams.kup1down_up_to > 0:
        for k in range(hparams.kup1down_up_to):
          with tf.variable_scope("{}up1down_addressing".format(k + 1)):
            if not hparams.relu_implementation:
              if hparams.elevate_underline:
                dep_prior_ku1d = get_lprior_kup1down_mtx(parse_gold, parse_label, num_labels,
                                                         tokens_to_keep=tokens_to_keep,
                                                         extreme_value=False,
                                                         layer_norm_to_heads=hparams.layer_norm_to_heads, k=k + 1,
                                                         memory_efficient=hparams.memory_efficient,
                                                         joint_par_srl_training=hparams.joint_par_srl_training,
                                                         relu_imp=False)
              else:
                dep_prior_ku1d = get_lprior_kup1down_mtx(parse_gold, parse_label, num_labels - 1,
                                                         tokens_to_keep=tokens_to_keep,
                                                         extreme_value=False,
                                                         layer_norm_to_heads=hparams.layer_norm_to_heads, k=k + 1,
                                                         memory_efficient=hparams.memory_efficient,
                                                         joint_par_srl_training=hparams.joint_par_srl_training,
                                                         relu_imp=False)
                sdp = tf.shape(dep_prior_ku1d)
                dep_prior_ku1d_empty_filler = tf.fill([sdp[0], sdp[1], sdp[2], 1], 0.)
                dep_prior_ku1d = tf.concat([dep_prior_ku1d_empty_filler, dep_prior_ku1d], axis=-1)
              dep_prior_ku1d_gathered = tf.gather_nd(dep_prior_ku1d, predicate_gather_indices)
            else:
              dep_prior_ku1d = get_lprior_kup1down_mtx(parse_gold, parse_label, num_labels,
                                                       tokens_to_keep=tokens_to_keep,
                                                       extreme_value=False,
                                                       layer_norm_to_heads=hparams.layer_norm_to_heads, k=k + 1,
                                                       memory_efficient=hparams.memory_efficient,
                                                       joint_par_srl_training=hparams.joint_par_srl_training,
                                                       relu_imp=True)
              dep_prior_ku1d_gathered = tf.gather_nd(dep_prior_ku1d, predicate_gather_indices)
            if hparams.srl_layernorm:
              with tf.variable_scope('LayerNorm'):
                dep_prior_ku1d_gathered = tf.contrib.layers.layer_norm(dep_prior_ku1d_gathered)
            prior_list.append(dep_prior_ku1d_gathered)

      tf.logging.log(tf.logging.INFO, "dep prior: {}".format(dep_prior))

      tf.logging.log(tf.logging.INFO, "gathered dep prior gather indices: {}".format(predicate_gather_indices))
      tf.logging.log(tf.logging.INFO, "gathered dep prior: {}".format(dep_prior_gathered))
      if not hparams.weight_per_label:
        lbda = tf.get_variable("dep_prior_weight", [len(prior_list), 1, 1, 1], initializer=tf.ones_initializer,
                               trainable=hparams.dep_prior_trainable)
      else:
        lbda = tf.get_variable("dep_prior_weight", [len(prior_list), 1, 1, num_labels],
                               initializer=tf.ones_initializer,
                               trainable=hparams.dep_prior_trainable)
      # NOTE:!! The print op here was lbda * dep_prior_gathered -> mis-use of boardcasting
      lbda = tf.Print(lbda, [lbda, (lbda * tf.stack(prior_list, axis=0))[:, :, :, :5]], "dep_prior_weight",
                      summarize=40)

      srl_logits_transposed += tf.reduce_sum(lbda * tf.stack(prior_list, axis=0), axis=0)
      srl_logits_transposed = tf.Print(srl_logits_transposed, [srl_logits_transposed[:, :, :5]],
                                       "srl_logits_transposed", summarize=40)

    # (3) compute loss

    # need to repeat each of these once for each target in the sentence
    mask_tiled = tf.reshape(tf.tile(tokens_to_keep, [1, batch_seq_len]), [batch_size, batch_seq_len, batch_seq_len])
    mask = tf.gather_nd(mask_tiled, predicate_gather_indices)

    # now we have k sets of targets for the k frames
    # (p1) f1 f2 f3
    # (p2) f1 f2 f3

    # get all the tags for each token (which is the predicate for a frame), structuring
    # targets as follows (assuming p1 and p2 are predicates for f1 and f3, respectively):
    # (p1) f1 f1 f1
    # (p2) f3 f3 f3
    srl_targets_transposed = tf.transpose(targets, [0, 2, 1])

    gold_predicate_counts = tf.reduce_sum(tf.cast(bool_mask_where_predicates(predicate_targets), tf.int32), -1)
    srl_targets_indices = tf.where(tf.sequence_mask(tf.reshape(gold_predicate_counts, [-1])))

    # num_predicates_in_batch x seq_len
    srl_targets_gold_predicates = tf.gather_nd(srl_targets_transposed, srl_targets_indices)

    predicted_predicate_counts = tf.reduce_sum(tf.cast(bool_mask_where_predicates(predicate_preds), tf.int32), -1)
    srl_targets_pred_indices = tf.where(tf.sequence_mask(tf.reshape(predicted_predicate_counts, [-1])))
    srl_targets_predicted_predicates = tf.gather_nd(srl_targets_transposed, srl_targets_pred_indices)

    # num_predicates_in_batch x seq_len
    predictions = tf.cast(tf.argmax(srl_logits_transposed, axis=-1), tf.int32)

    seq_lens = tf.cast(tf.reduce_sum(mask, 1), tf.int32)

    if (mode == ModeKeys.PREDICT or mode == ModeKeys.EVAL):
      srl_targets_onehot = tf.one_hot(indices=srl_targets_predicted_predicates, depth=num_labels, axis=-1)
      loss_emission = tf.losses.softmax_cross_entropy(logits=tf.reshape(srl_logits_transposed, [-1, num_labels]),
                                                      onehot_labels=tf.reshape(srl_targets_onehot, [-1, num_labels]),
                                                      weights=tf.reshape(mask, [-1]),
                                                      label_smoothing=hparams.label_smoothing,
                                                      reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS)

      loss = loss_emission
      predictions = tf.argmax(srl_logits_transposed, axis=-1)

    if mode == ModeKeys.TRAIN:  # and tf_utils.is_trainable(transition_params):
      # flat_seq_lens = tf.reshape(tf.tile(seq_lens, [1, bucket_size]), tf.stack([batch_size * bucket_size]))
      srl_targets_onehot = tf.one_hot(indices=srl_targets_predicted_predicates, depth=num_labels, axis=-1)
      loss_emission = tf.losses.softmax_cross_entropy(logits=tf.reshape(srl_logits_transposed, [-1, num_labels]),
                                                      onehot_labels=tf.reshape(srl_targets_onehot, [-1, num_labels]),
                                                      weights=tf.reshape(mask, [-1]),
                                                      label_smoothing=hparams.label_smoothing,
                                                      reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS)

      loss = loss_emission

    output = {
      'loss': loss,
      'predictions': predictions,
      'scores': srl_logits_transposed,
      'targets': srl_targets_gold_predicates,
      'probabilities': tf.nn.softmax(srl_logits_transposed, -1)
    }

    return output


def get_dep_transition_mtx(parse_gold, tokens_to_keep=None, extreme_value=False, layer_norm_to_heads=False,
                           transpose=False, memory_efficient=False, joint_par_srl_training=False, relu_imp=False,
                           pos_tag=None, parse_labels=None, number_labels=1):
  """
    heads: head-dependent distribution of shape (B, seq_len, seq_len)
    labels: label distribution for each head-dependent choice, shape: (B, seq_len, labels)
  """
  heads = parse_gold
  labels = parse_labels
  token_mask_row = tf.expand_dims(tf.cast(1 - tokens_to_keep, tf.float32) * constants.VERY_SMALL, 1)
  token_mask_col = tf.expand_dims(tf.cast(tokens_to_keep, tf.float32), 2)

  tf.logging.log(tf.logging.INFO, "Generating labeled-prior for SRL")

  if len(heads.get_shape()) < 3:
    if extreme_value:
      on_value = constants.VERY_LARGE
      off_value = constants.VERY_SMALL
    else:
      on_value = 10.
      off_value = -10.

    heads = tf.one_hot(heads, tf.shape(heads)[-1], off_value=off_value, on_value=on_value)
    heads = heads + token_mask_row
    if labels is not None:
      labels = tf.one_hot(labels, 69, off_value=off_value, on_value=on_value)

  else:

    heads = heads + token_mask_row
    if True:
      heads = tf.stop_gradient(heads)
      if labels is not None:
        labels = tf.stop_gradient(labels)
    # heads = tf.concat([heads[:, :1, :] + 100, heads[:, 1:, :]], axis=1)

  masked_heads = tf.expand_dims(tf.nn.softmax(heads, axis=-1), axis=-1)
  if labels is not None:
    with tf.variable_scope("dep_transition_mtx_labels"):
      dense = tf.keras.layers.Dense(
        number_labels, activation=tf.nn.sigmoid, use_bias=True,
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros'
      )
      # convert to (B, seq, 1) tensor -> open_close gate of each dependency arc
      output = dense(labels)
      # gating dependency graph with dependency type
      masked_heads *= tf.expand_dims(output, axis=-2)
  masked_heads *= tf.expand_dims(token_mask_col, -1)
  if transpose:
    masked_heads = tf.transpose(masked_heads, perm=[0, 2, 1, 3])

  # masked_heads = tf.Print(masked_heads, [masked_heads[:, :, :]], "priors inferred from dependency graph", summarize=40)
  return masked_heads


def get_dep_transition_kup1down_mtx(parse_gold, tokens_to_keep=None, extreme_value=False, layer_norm_to_heads=False,
                                    transpose=False, memory_efficient=False, joint_par_srl_training=False, k=-1,
                                    parse_labels=None, pow_norm=False):
  """
    heads: head-dependent distribution of shape (B, seq_len, seq_len)
    labels: label distribution for each head-dependent choice, shape: (B, seq_len, labels)
  """
  print("kup1down:", k)
  assert k > 0
  heads = parse_gold
  labels = parse_labels
  token_mask_row = tf.expand_dims(tf.cast(1 - tokens_to_keep, tf.float32) * constants.VERY_SMALL, 1)
  token_mask_col = tf.expand_dims(tf.cast(tokens_to_keep, tf.float32), 2)
  # token_mask_col = tf.expand_dims(tf.cast(1 - tokens_to_keep, tf.float32) * constants.VERY_SMALL, 2)

  tf.logging.log(tf.logging.INFO, "Using simple {}up1down direction, removing all prob to ROOT".format(k))

  def rm_sr(mtx):
    return mtx * (1 - tf.eye(tf.shape(heads)[1], batch_shape=[tf.shape(heads)[0]]))

  def prod(l, r):
    # l, r are of (B, S, S)
    l_t = l
    r_t = r
    tmp_t = tf.linalg.matmul(l_t, r_t)
    tmp = tmp_t
    return tmp

  if len(heads.get_shape()) < 3:
    # max_seq_len = tf.cast(tf.shape(heads)[-1], dtype=tf.float32)

    if extreme_value:
      on_value = constants.VERY_LARGE
      off_value = constants.VERY_SMALL
    else:
      on_value = 10.
      off_value = -10.

    heads = tf.one_hot(heads, tf.shape(heads)[-1], off_value=off_value, on_value=on_value)
    if labels is not None:
      labels = tf.one_hot(labels, 69, off_value=off_value, on_value=on_value)
    heads = heads + token_mask_row
  else:

    heads = heads + token_mask_row
    if True:
      heads = tf.stop_gradient(heads)
      if labels is not None:
        labels = tf.stop_gradient(labels)

  masked_heads = tf.nn.softmax(heads, axis=-1)
  # including k steps up and 1 step down
  masked_heads_list = []
  for label_per_step in range(k + 1):
    with tf.variable_scope("{}th_label_transformation".format(label_per_step)):
      if labels is not None:
        with tf.variable_scope("dep_transition_mtx_labels"):
          dense = tf.keras.layers.Dense(
            1, activation=tf.nn.sigmoid, use_bias=True,
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros'
          )
          # convert to (B, seq, 1) tensor -> open_close gate of each dependency arc
          output = dense(labels)
          # gating dependency graph with dependency type
          masked_heads_list.append(masked_heads * output)
      # else:
      #   masked_heads_list.append(masked_heads)
  # Applying column-wise masking
  masked_heads_list = [item * token_mask_col for item in masked_heads_list]
  up, down = masked_heads_list[-2:]
  # Prevent back-looping
  # updown = prod(up, tf.transpose(down, perm=[0, 2, 1])) * (1 - tf.eye(tf.shape(heads)[1], batch_shape=[tf.shape(heads)[0]]))
  updown = rm_sr(prod(rm_sr(up), rm_sr(tf.transpose(down, perm=[0, 2, 1]))))
  tmp = updown
  for idx in reversed(range(k - 1)):
    tmp = prod(rm_sr(masked_heads_list[idx]), tmp)
  if pow_norm:
    tmp = tf.math.pow(tmp, 1 / (k + 1))
  return tmp


def get_dep_transition_kup1down_mtx_collect_dep_path(parse_gold, hiddens, tokens_to_keep=None, extreme_value=False,
                                                     layer_norm_to_heads=False, transpose=False, memory_efficient=False,
                                                     joint_par_srl_training=False, k=-1, parse_labels=None,
                                                     pow_norm=False):
  """
    heads: head-dependent distribution of shape (B, seq_len, seq_len)
    labels: label distribution for each head-dependent choice, shape: (B, seq_len, labels)
  """
  print("kup1down:", k)
  assert k > 0
  heads = parse_gold
  labels = parse_labels
  token_mask_row = tf.expand_dims(tf.cast(1 - tokens_to_keep, tf.float32) * constants.VERY_SMALL, 1)
  token_mask_col = tf.expand_dims(tf.cast(tokens_to_keep, tf.float32), 2)
  # token_mask_col = tf.expand_dims(tf.cast(1 - tokens_to_keep, tf.float32) * constants.VERY_SMALL, 2)

  tf.logging.log(tf.logging.INFO, "Using simple {}up1down direction, removing all prob to ROOT".format(k))

  def rm_sr(mtx):
    return mtx * (1 - tf.eye(tf.shape(heads)[1], batch_shape=[tf.shape(heads)[0]]))

  def prod(l, r):
    # l, r are of (B, S, S)
    l_t = l
    r_t = r
    tmp_t = tf.linalg.matmul(l_t, r_t)
    tmp = tmp_t
    return tmp

  if len(heads.get_shape()) < 3:
    # max_seq_len = tf.cast(tf.shape(heads)[-1], dtype=tf.float32)

    if extreme_value:
      on_value = constants.VERY_LARGE
      off_value = constants.VERY_SMALL
    else:
      on_value = 10.
      off_value = -10.

    heads = tf.one_hot(heads, tf.shape(heads)[-1], off_value=off_value, on_value=on_value)
    if labels is not None:
      labels = tf.one_hot(labels, 69, off_value=off_value, on_value=on_value)
    heads = heads + token_mask_row
  else:

    heads = heads + token_mask_row
    if True:
      heads = tf.stop_gradient(heads)
      if labels is not None:
        labels = tf.stop_gradient(labels)

  masked_heads = tf.nn.softmax(heads, axis=-1)
  # including k steps up and 1 step down
  masked_heads_list = []
  for label_per_step in range(k + 1):
    masked_heads_list.append(masked_heads)
  # Applying column-wise masking
  masked_heads_list = [item * token_mask_col for item in masked_heads_list]
  up, down = masked_heads_list[-2:]
  # Prevent back-looping
  # updown = prod(up, tf.transpose(down, perm=[0, 2, 1])) * (1 - tf.eye(tf.shape(heads)[1], batch_shape=[tf.shape(heads)[0]]))
  updown = rm_sr(prod(rm_sr(up), rm_sr(tf.transpose(down, perm=[0, 2, 1]))))
  tmp = updown

  for idx in reversed(range(k - 1)):
    tmp = prod(rm_sr(masked_heads_list[idx]), tmp)
  if pow_norm:
    tmp = tf.math.pow(tmp, 1 / (k + 1))
  return tf.expand_dims(tmp, axis=-1)


def get_dep_transition_kup1down_mtx_collect_ste(parse_gold, hiddens, tokens_to_keep=None, extreme_value=False,
                                                layer_norm_to_heads=False, transpose=False, memory_efficient=False,
                                                joint_par_srl_training=False, k=-1, parse_labels=None, pow_norm=False,
                                                use_lstm=False):
  """
    heads: head-dependent distribution of shape (B, seq_len, seq_len)
    labels: label distribution for each head-dependent choice, shape: (B, seq_len, labels)
  """
  print("kup1down:", k)
  assert k == 1
  heads = parse_gold
  labels = parse_labels
  token_mask_row = tf.expand_dims(tf.cast(1 - tokens_to_keep, tf.float32) * constants.VERY_SMALL, 1)
  token_mask_col = tf.expand_dims(tf.cast(tokens_to_keep, tf.float32), 2)
  # token_mask_col = tf.expand_dims(tf.cast(1 - tokens_to_keep, tf.float32) * constants.VERY_SMALL, 2)

  tf.logging.log(tf.logging.INFO, "Using simple {}up1down direction, removing all prob to ROOT".format(k))

  def rm_sr(mtx):
    return mtx * (1 - tf.eye(tf.shape(heads)[1], batch_shape=[tf.shape(heads)[0]]))

  def prod(l, r):
    # l, r are of (B, S, S)
    l_t = l
    r_t = r
    tmp_t = tf.linalg.matmul(l_t, r_t)
    tmp = tmp_t
    return tmp

  if len(heads.get_shape()) < 3:
    # max_seq_len = tf.cast(tf.shape(heads)[-1], dtype=tf.float32)

    if extreme_value:
      on_value = constants.VERY_LARGE
      off_value = constants.VERY_SMALL
    else:
      on_value = 10.
      off_value = -10.

    heads = tf.one_hot(heads, tf.shape(heads)[-1], off_value=off_value, on_value=on_value)
    if labels is not None:
      labels = tf.one_hot(labels, 69, off_value=off_value, on_value=on_value)
    heads = heads + token_mask_row
  else:

    heads = heads + token_mask_row
    if True:
      heads = tf.stop_gradient(heads)
      if labels is not None:
        labels = tf.stop_gradient(labels)

  masked_heads = tf.nn.softmax(heads, axis=-1)
  # including k steps up and 1 step down
  masked_heads_list = []
  for label_per_step in range(k + 1):
    masked_heads_list.append(masked_heads)
  # Applying column-wise masking
  masked_heads_list = [item * token_mask_col for item in masked_heads_list]

  up, down = masked_heads_list[-2:]

  hidden_list = []
  # hidden_list.append(hiddens)
  # hidden_list.append(tf.linalg.matmul(tf.transpose(up, perm=[0, 2, 1]), hiddens))

  # Prevent back-looping
  # updown = prod(up, tf.transpose(down, perm=[0, 2, 1])) * (1 - tf.eye(tf.shape(heads)[1], batch_shape=[tf.shape(heads)[0]]))
  updown = rm_sr(prod(rm_sr(up), rm_sr(tf.transpose(down, perm=[0, 2, 1]))))
  tmp = updown
  # going through a double tranposition
  hidden_list.append(tf.linalg.matmul(down, hiddens))
  hidden_list.append(hiddens)

  if not use_lstm:
    output_r = tf.reduce_mean(tf.stack(hidden_list, axis=-1), axis=-1)
  else:
    with tf.variable_scope("bilstm"):
      _, _, hidden_size = hiddens.get_shape().as_list()
      # batch_size, outer_seq_len, _ =
      lstm = tf.keras.layers.LSTM(int(hidden_size / 2), dropout=0.1)
      bidirectional = tf.keras.layers.Bidirectional(lstm, merge_mode='concat')
      output_seq = tf.reshape(tf.stack(hidden_list, axis=-2), shape=[-1, k + 1, hidden_size])
      output_state = bidirectional(output_seq)
      output_r = tf.reshape(output_state, shape=[tf.shape(hiddens)[0], tf.shape(hiddens)[1], hidden_size])

  return tf.expand_dims(tmp, axis=-1), output_r


def get_dep_transition_xupydown_mtx_collect_ste(parse_gold, hiddens, tokens_to_keep=None, extreme_value=False,
                                                layer_norm_to_heads=False, transpose=False, memory_efficient=False,
                                                joint_par_srl_training=False, x=-1, y=-1, parse_labels=None,
                                                pow_norm=False, use_lstm=False, return_last=False,
                                                use_new_version=False):
  """
    heads: head-dependent distribution of shape (B, seq_len, seq_len)
    labels: label distribution for each head-dependent choice, shape: (B, seq_len, labels)
  """
  # print(
  assert x >= 0
  assert y >= 0
  if x == 0 and y == 0:
    sr_position_mask = tf.eye(tf.shape(parse_gold)[1], batch_shape=[tf.shape(parse_gold)[0]])
    sr_mask = tf.expand_dims(sr_position_mask, axis=-1)
    return sr_mask, hiddens
  elif x > 0 and y == 0:
    return get_dep_transition_kup_mtx_collect_ste(parse_gold, hiddens, transpose=False,
                                                  tokens_to_keep=tokens_to_keep,
                                                  parse_labels=parse_labels,
                                                  k=x, use_lstm=use_lstm, return_last=return_last)
  elif x == 0 and y > 0:
    return get_dep_transition_kup_mtx_collect_ste(parse_gold, hiddens, transpose=True,
                                                  tokens_to_keep=tokens_to_keep,
                                                  parse_labels=parse_labels,
                                                  k=y, use_lstm=use_lstm, return_last=return_last)

  heads = parse_gold
  labels = parse_labels
  token_mask_row = tf.expand_dims(tf.cast(1 - tokens_to_keep, tf.float32) * constants.VERY_SMALL, 1)
  token_mask_col = tf.expand_dims(tf.cast(tokens_to_keep, tf.float32), 2)
  # token_mask_col = tf.expand_dims(tf.cast(1 - tokens_to_keep, tf.float32) * constants.VERY_SMALL, 2)

  tf.logging.log(tf.logging.INFO, "{}up{}down:".format(x, y))

  def rm_sr(mtx):
    return mtx * (1 - tf.eye(tf.shape(heads)[1], batch_shape=[tf.shape(heads)[0]]))

  def prod(l, r):
    # l, r are of (B, S, S)
    l_t = l
    r_t = r
    tmp_t = tf.linalg.matmul(l_t, r_t)
    tmp = tmp_t
    return tmp

  if len(heads.get_shape()) < 3:
    # max_seq_len = tf.cast(tf.shape(heads)[-1], dtype=tf.float32)

    if extreme_value:
      on_value = constants.VERY_LARGE
      off_value = constants.VERY_SMALL
    else:
      on_value = 10.
      off_value = -10.

    heads = tf.one_hot(heads, tf.shape(heads)[-1], off_value=off_value, on_value=on_value)
    if labels is not None:
      labels = tf.one_hot(labels, 69, off_value=off_value, on_value=on_value)
    heads = heads + token_mask_row
  else:

    heads = heads + token_mask_row
    if True:
      heads = tf.stop_gradient(heads)
      if labels is not None:
        labels = tf.stop_gradient(labels)

  masked_heads = tf.nn.softmax(heads, axis=-1)
  masked_heads_list = []
  for label_per_step in range(x + y):
    masked_heads_list.append(masked_heads)
  # Applying column-wise masking
  masked_heads_list = [item * token_mask_col for item in masked_heads_list]

  up_l, down_f = masked_heads_list[x - 1:x + 1]

  hidden_list = []
  updown = rm_sr(prod(rm_sr(up_l), rm_sr(tf.transpose(down_f, perm=[0, 2, 1]))))
  tmp = updown

  for idx in reversed(range(0, x - 1)):
    l_prod = masked_heads_list[idx]  # if not transpose else tf.transpose(masked_heads_list[idx], perm=[0, 2, 1])
    tmp = prod(l_prod, tmp)
  # hidden_list.append(tf.linalg.matmul(down_f, hiddens))
  for idx in range(y + 1, x + y):
    r_prod = masked_heads_list[idx]  # if not transpose else tf.transpose(masked_heads_list[idx], perm=[0, 2, 1])
    tmp = tf.matmul(tmp, r_prod, transpose_b=use_new_version)

  # hidden_list.append(hiddens)
  tf.logging.log(tf.logging.INFO, "assembling hidden_list for [0, {})".format(x))
  for idx in range(0, x):
    l_prod = masked_heads_list[idx]
    hidden_list = [tf.linalg.matmul(tf.transpose(l_prod, perm=[0, 2, 1]), item) for item in hidden_list]
    hidden_list.append(hiddens)
  tf.logging.log(tf.logging.INFO, "assembling hidden_list for [{}, {})".format(x, x + y))
  for idx in range(x, x + y):
    r_prod = masked_heads_list[idx]  # if not transpose else tf.transpose(masked_heads_list[idx], perm=[0, 2, 1])
    hidden_list = [tf.linalg.matmul(r_prod, item) for item in hidden_list]
    hidden_list.append(hiddens)
  tf.logging.log(tf.logging.INFO, "hidden list {}".format(hidden_list))

  if not use_lstm:
    if return_last:
      output_r = hidden_list[-1]
    else:
      output_r = tf.reduce_mean(tf.stack(hidden_list, axis=-1), axis=-1)
  else:
    with tf.variable_scope("bilstm"):
      _, _, hidden_size = hiddens.get_shape().as_list()
      cell_fw = tf.keras.layers.LSTMCell(int(hidden_size / 2), dropout=0.1, recurrent_dropout=0.1)
      cell_bw = tf.keras.layers.LSTMCell(int(hidden_size / 2), dropout=0.1, recurrent_dropout=0.1)
      output_seq = tf.reshape(tf.stack(hidden_list, axis=-2), shape=[-1, x + y, hidden_size])
      output, output_state = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, output_seq, swap_memory=True,
                                                             dtype=tf.float32)
      output_state = [item[1] for item in output_state]
      tf.logging.log(tf.logging.INFO, "bidirectional_output {}".format(output_state))
      output_r = tf.reshape(tf.concat(list(output_state), axis=-1),
                            shape=[tf.shape(hiddens)[0], tf.shape(hiddens)[1], hidden_size])
  # if return_last:
  #   output_r = output_r[-1:]

  return tf.expand_dims(tmp, axis=-1), output_r


def get_dep_transition_wfs(parse_gold, mode, tokens_to_keep=None, extreme_value=False, parse_labels=None,
                           use_lstm=False, return_last=False, max_up_depth=5, max_down_depth=3, num_clusters=3,
                           inference_mode='full', gumbel_temperature=1e-5, cluster_prob=False, propagation_depth=0,
                           pred_mlp=None, role_mlp=None, use_pred_role_feature=False, predicate_gather_indices=None,
                           legacy_mode=False, before_after_indicator=False, layer_norm=False, use_gumbel_max=False,
                           deal_with_masked_elements=False, use_dep_label=False, ignore_masks=False, relu6=False,
                           parse_label_count=69):
  """
    heads: head-dependent distribution of shape (B, seq_len, seq_len)
    labels: label distribution for each head-dependent choice, shape: (B, seq_len, labels)
  """

  heads = parse_gold
  labels = parse_labels

  if use_dep_label:
    assert labels is not None

  token_mask_row = tf.expand_dims(tf.cast(1 - tokens_to_keep, tf.float32) * constants.VERY_SMALL, 1)
  token_mask_row_normal_scale = tf.expand_dims(tf.cast(tokens_to_keep, tf.float32), 1)
  token_mask_col = tf.expand_dims(tf.cast(tokens_to_keep, tf.float32), 2)

  # token_mask_col = tf.expand_dims(tf.cast(1 - tokens_to_keep, tf.float32) * constants.VERY_SMALL, 2)

  tf.logging.log(tf.logging.INFO, "mixture model with {} clusters and {} maximum exploring steps".format(num_clusters, (
    max_up_depth, max_down_depth)))

  def rm_sr(mtx):
    return mtx * (1 - tf.eye(tf.shape(heads)[1], batch_shape=[tf.shape(heads)[0]]))

  if len(heads.get_shape()) < 3:
    # max_seq_len = tf.cast(tf.shape(heads)[-1], dtype=tf.float32)

    if extreme_value:
      on_value = constants.VERY_LARGE
      off_value = constants.VERY_SMALL
    else:
      on_value = 10.
      off_value = -10.

    heads = tf.one_hot(heads, tf.shape(heads)[-1], off_value=off_value, on_value=on_value)
    labels = tf.one_hot(labels, parse_label_count, off_value=off_value, on_value=on_value)
    heads = heads + token_mask_row
  else:
    heads = heads + token_mask_row
    if labels is not None:
      labels = tf.stop_gradient(labels)
    heads = tf.stop_gradient(heads)
    if use_gumbel_max:
      labels_dist = tfp.distributions.OneHotCategorical(logits=labels)
      heads_dist = tfp.distributions.OneHotCategorical(logits=heads)
      labels = tf.cast(tfp.distributions.Sample(labels_dist, sample_shape=()).sample(), tf.float32)
      heads = tf.cast(tfp.distributions.Sample(heads_dist, sample_shape=()).sample(), tf.float32)

  if not use_gumbel_max:
    masked_heads = tf.nn.softmax(heads, axis=-1)
    if labels is not None:
      labels = tf.nn.softmax(labels, axis=-1)
  else:
    masked_heads = heads
    labels = labels

  batch_size, seq_len = tf.shape(masked_heads)[0], tf.shape(masked_heads)[1]
  gathered_tokens_to_keep = tf.gather_nd(tf.tile(tf.reshape(tokens_to_keep, [batch_size, 1, seq_len]), [1, seq_len, 1]),
                                         predicate_gather_indices)
  trigger_size = tf.shape(gathered_tokens_to_keep)[0]

  synt_mask_mtx = tf.ones_like(heads) * token_mask_row_normal_scale * token_mask_col

  masked_heads = masked_heads * token_mask_col

  up_l, down_f = masked_heads, masked_heads

  updown = rm_sr(tf.matmul(rm_sr(up_l), rm_sr(down_f), transpose_b=True))
  no_op = tf.eye(tf.shape(masked_heads)[-1], batch_shape=[tf.shape(masked_heads)[0]])
  no_op *= token_mask_row_normal_scale * token_mask_col

  LABEL_NOOP = parse_label_count
  LABEL_MASK = parse_label_count + 1

  with tf.variable_scope('wfs_embedding'):
    step_embedding_mtx = tf.get_variable('step_embedding', [len(step_to_feature_idx) + 2, 64])
    if use_dep_label:
      dep_label_embedding_mtx = tf.get_variable('label_embedding', [parse_label_count + 2, 64],
                                                initializer=tf.keras.initializers.Orthogonal())

  if use_dep_label:
    label_hidden = tf.matmul(labels, tf.reshape(dep_label_embedding_mtx[:-2], [1, parse_label_count, 64]))
    tiled_down_label_hidden = tf.tile(tf.reshape(label_hidden, [batch_size, 1, seq_len, 64]), [1, seq_len, 1, 1])
    tiled_up_label_hidden = tf.transpose(tiled_down_label_hidden, [0, 2, 1, 3])
    tiled_noop_label_hidden = tf.tile(
      tf.reshape(tf.nn.embedding_lookup(step_embedding_mtx, tf.reshape(LABEL_NOOP, [1])), [1, 1, 1, 64]),
      [batch_size, seq_len, seq_len, 1])
    gathered_tiled_down_label_hidden = tf.gather_nd(tiled_down_label_hidden, predicate_gather_indices)
    gathered_tiled_up_label_hidden = tf.gather_nd(tiled_up_label_hidden, predicate_gather_indices)
    gathered_tiled_noop_label_hidden = tf.gather_nd(tiled_noop_label_hidden, predicate_gather_indices)
  else:
    gathered_tiled_up_label_hidden = None
    gathered_tiled_down_label_hidden = None
    gathered_tiled_noop_label_hidden = None

  # queue = [masked_heads]
  queue = [(1, (1, 0), masked_heads, 'up', gathered_tiled_up_label_hidden),
           (2, (1, 1), updown, 'down', gathered_tiled_down_label_hidden),
           (0, (0, 0), no_op, 'down', gathered_tiled_noop_label_hidden)]
  heapq.heapify(queue)
  # queue_step = [(1, 1, 0)]
  # (steps, B, seq_len, seq_len)
  complete_list = []
  # (steps, patterns)
  complete_step = []

  MASK_STEP = len(step_to_feature_idx) + 1
  OTHERS_STEP = len(step_to_feature_idx)

  # step_embedding_layer = tf.keras.layers.Embedding(len(step_to_feature_idx), 64, input_length=1, trainable=True, embeddings_initializer='GlorotNormal')

  if use_pred_role_feature:
    pred_hiddens = tf.expand_dims(pred_mlp, axis=2)
    pred_hiddens_tiled = tf.tile(pred_hiddens, [1, 1, seq_len, 1])
    role_hiddens = tf.expand_dims(pred_mlp, axis=1)
    role_hiddens_tiled = tf.tile(role_hiddens, [1, seq_len, 1, 1])
    gathered_role_mlp = tf.gather_nd(role_hiddens_tiled, predicate_gather_indices)

  with tf.variable_scope("syn_pattern_to_cluster"):
    if legacy_mode:
      dense1 = tf.keras.layers.Dense(int((101 + num_clusters) / 2), activation='relu')
    else:
      if use_pred_role_feature:
        dense1 = tf.keras.layers.Dense(256, activation='relu')
      else:
        with tf.variable_scope('batchnorm_srl_score'):
          batch_norm_1 = tf.keras.layers.BatchNormalization(
            axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
            beta_initializer='zeros', gamma_initializer='ones',
            moving_mean_initializer='zeros',
            moving_variance_initializer='ones', beta_regularizer=None,
            gamma_regularizer=None, beta_constraint=None, gamma_constraint=None
          )
          batch_norm_2 = tf.keras.layers.BatchNormalization(
            axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
            beta_initializer='zeros', gamma_initializer='ones',
            moving_mean_initializer='zeros',
            moving_variance_initializer='ones', beta_regularizer=None,
            gamma_regularizer=None, beta_constraint=None, gamma_constraint=None
          )
        dense1 = tf.keras.layers.Dense(512, activation='relu')
    dense2 = tf.keras.layers.Dense(num_clusters)

  # def explore():
  while len(queue) > 0:
    # retrieve current mtx
    total_step, step, mtx, state, label_hidden = heapq.heappop(queue)

    if step[0] > max_up_depth or step[1] > max_down_depth:
      continue

    if state == 'up':
      plausible_action = ['up', 'updown', 'complete' if legacy_mode else 'complete_new']
    elif state == 'down':
      plausible_action = ['down', 'complete' if legacy_mode else 'complete_new']
    else:
      raise NotImplementedError

    if 'up' in plausible_action:
      next_step = tf.matmul(mtx, masked_heads)
      heapq.heappush(queue, (total_step + 1, (step[0] + 1, step[1]), next_step, 'up', gathered_tiled_up_label_hidden))

    if 'down' in plausible_action:
      next_step = tf.matmul(mtx, masked_heads, transpose_b=True)
      heapq.heappush(queue,
                     (total_step + 1, (step[0], step[1] + 1), next_step, 'down', gathered_tiled_down_label_hidden))

    if 'updown' in plausible_action:
      next_step = tf.matmul(mtx, updown)
      heapq.heappush(queue,
                     (total_step + 2, (step[0] + 1, step[1] + 1), next_step, 'down', gathered_tiled_down_label_hidden))

    if 'complete' in plausible_action:
      to_be_appended = tf.stop_gradient(tf.minimum(mtx, synt_mask_mtx))
      complete_list.append(to_be_appended)
      synt_mask_mtx -= to_be_appended
      complete_step.append(tf.one_hot(step_to_feature_idx[step], depth=101))
      # tf.logging.log(tf.logging.INFO, "complete searching mtx of {}".format(step))

    if 'complete_new' in plausible_action:
      addr_mtx = tf.stop_gradient(tf.minimum(mtx, synt_mask_mtx))
      synt_mask_mtx -= addr_mtx
      # addr_mtx: (n_trigger, seq, 1)
      gathered_addr_mtx = tf.expand_dims(tf.gather_nd(addr_mtx, predicate_gather_indices), axis=-1)

      step_hiddens = tf.nn.embedding_lookup(step_embedding_mtx, tf.reshape(step_to_feature_idx[step], [1]))
      step_hiddens_reshaped = tf.reshape(step_hiddens, [1, 1, 64])
      step_hiddens_gathered = tf.tile(step_hiddens_reshaped, [trigger_size, seq_len, 1])
      synt_features = [step_hiddens_gathered]

      if use_pred_role_feature:
        synt_features += [gathered_role_mlp]

      if use_dep_label:
        synt_features += [label_hidden]

        # feature_set = tf.concat([feature_set, gathered_tileed_label_hidden], axis=-1)

      synt_features = tf.concat(synt_features, axis=-1)
      # score: (n_trigger, seq, hidden) -> (n_trigger, seq, n_cluster)
      # score = dense2(dense1(synt_features))
      score = synt_features
      complete_list.append(gathered_addr_mtx * score)

      # complete_step.append(tf.one_hot(step_to_feature_idx[step], depth=101))
      # tf.logging.log(tf.logging.INFO, "complete searching mtx of {}".format(step))

  if legacy_mode:
    complete_list.append(tf.stop_gradient(synt_mask_mtx))
    complete_step.append(tf.one_hot(100, depth=101))

    complete_step = tf.stack(complete_step, axis=0)
    z_prob = dense2(dense1(complete_step))
    if cluster_prob:
      patterns = tf.eye(101)
      # tf.logging.log("")
      z_prob_per_pattern = dense2(dense1(patterns))
      z_selection_per_pattern = tf.expand_dims(tf.math.argmax(z_prob_per_pattern, axis=-1), axis=-1)
      z_prob_per_pattern = tf.nn.softmax(z_prob_per_pattern, axis=-1)
      z_prob = tf.Print(z_prob, [z_prob_per_pattern, tf.math.less(tf.reduce_max(z_prob_per_pattern, axis=-1), 0.8)],
                        "z_probability_per_item", summarize=303)
      z_prob = tf.Print(z_prob, [z_selection_per_pattern], "z_selection_per_pattern", summarize=303)

    # it has a shape of (B, Seq, Seq, St)
    return_masks = tf.stack(complete_list, axis=-1)
    # return_masks = tf.Print(return_masks, [return_masks], summarize=30)
    # it has a shape of (B, Seq, Seq, num_clusters)
    return_masks = tf.matmul(return_masks, tf.expand_dims(tf.expand_dims(z_prob, axis=0), axis=0))
    z_logit = return_masks
  else:
    if not ignore_masks:
      non_keep_tokens = tf.expand_dims(1. - tf.cast(gathered_tokens_to_keep, tf.float32), axis=-1)
      step_hiddens = tf.nn.embedding_lookup(step_embedding_mtx, tf.reshape(MASK_STEP, [1]))
      step_hiddens_reshaped = tf.reshape(step_hiddens, [1, 1, 64])
      step_hiddens_gathered = tf.tile(step_hiddens_reshaped, [trigger_size, seq_len, 1])
      synt_features = [step_hiddens_gathered]
      if use_dep_label:
        tiled_mask_label_hidden = tf.tile(
          tf.reshape(tf.nn.embedding_lookup(step_embedding_mtx, tf.reshape(LABEL_MASK, [1])), [1, 1, 1, 64]),
          [batch_size, seq_len, seq_len, 1])
        gathered_tiled_mask_label_hidden = tf.gather_nd(tiled_mask_label_hidden, predicate_gather_indices)
        synt_features += [gathered_tiled_mask_label_hidden]
      if use_pred_role_feature:
        synt_features += [gathered_role_mlp]
      synt_features = tf.concat(synt_features, axis=-1)
      # score = dense2(dense1(synt_features))
      score = synt_features
      complete_list.append(non_keep_tokens * score)

    '--------------------------------'
    gathered_synt_mask_mtx = tf.expand_dims(tf.gather_nd(tf.stop_gradient(synt_mask_mtx), predicate_gather_indices),
                                            axis=-1)
    step_hiddens = tf.nn.embedding_lookup(step_embedding_mtx, tf.reshape(OTHERS_STEP, [1]))
    step_hiddens_reshaped = tf.reshape(step_hiddens, [1, 1, 64])
    step_hiddens_gathered = tf.tile(step_hiddens_reshaped, [trigger_size, seq_len, 1])
    synt_features = [step_hiddens_gathered]
    if use_dep_label:
      synt_features += [gathered_tiled_noop_label_hidden]
    if use_pred_role_feature:
      synt_features += [gathered_role_mlp]
    synt_features = tf.concat(synt_features, axis=-1)
    # score = dense2(dense1(synt_features))
    score = synt_features
    complete_list.append(gathered_synt_mask_mtx * score)
    feature_set = tf.reduce_sum(complete_list, axis=0)

    # stacked_complete_list = tf.stack(complete_list, axis=-1)
    # if relu6:
    #   z_logit = batch_norm_2(dense2(batch_norm_1(dense1(tf.reduce_sum(complete_list, axis=0)))))
    # else:
    #   z_logit = dense2(dense1(tf.reduce_sum(complete_list, axis=0)))

    z_logit = batch_norm_2(dense2(batch_norm_1(dense1(feature_set))))
  # output_r = None

  # if mode != ModeKeys.TRAIN:
    # applied_patterns = [step_to_feature_idx[(i, j)] for i in range(max_up_depth) for j in range(max_down_depth)] + [OTHERS_STEP, MASK_STEP]
    # ap_emb = tf.nn.embedding_lookup(step_embedding_mtx, applied_patterns)
    # ap_emb = tf.expand_dims(ap_emb, 0)
    # patterns_assigned_cluster = batch_norm_2(dense2(batch_norm_1(dense1(ap_emb))))
    # z_logit = tf.Print(z_logit, [tf.nn.softmax(z_logit), tf.argmax(z_logit, axis=-1)], "z_logit", summarize=30)
    # z_logit = tf.Print(z_logit, [tf.nn.softmax(patterns_assigned_cluster), tf.argmax(patterns_assigned_cluster, axis=-1)], "cluster_assignment", summarize=max_up_depth*max_down_depth+10)

  if inference_mode == 'full':
    z_prob = tf.nn.softmax(z_logit, -1)
  elif inference_mode == 'nll' or inference_mode == 'nll_normal':
    z_prob = z_logit
  elif inference_mode == 'gumbel':
    if mode != ModeKeys.TRAIN:
      temperature = 1e-5
      dist = tfp.distributions.RelaxedOneHotCategorical(temperature, logits=z_logit)
      sample = tfp.distributions.Sample(dist)
      z_prob = sample.sample()
    else:
      current_step = tf.train.get_global_step()
      temperature = tf.maximum(0.5, tf.exp(-1e-5 * tf.cast(current_step, tf.float32)))
      dist = tfp.distributions.RelaxedOneHotCategorical(temperature, logits=z_logit)
      sample = tfp.distributions.Sample(dist)
      z_prob = sample.sample()
    tf.logging.log(tf.logging.INFO, "z_sample:{}".format(z_prob))
  else:
    raise NotImplementedError
  additional_loss = 0.
  return z_prob, additional_loss


def get_dep_transition_wfs_label(parse_gold, mode, tokens_to_keep=None, extreme_value=False, parse_labels=None,
                                 use_lstm=False, return_last=False, max_up_depth=5, max_down_depth=3, num_clusters=3,
                                 inference_mode='full', gumbel_temperature=1e-5, cluster_prob=False,
                                 propagation_depth=0,
                                 pred_mlp=None, role_mlp=None, use_pred_role_feature=False,
                                 predicate_gather_indices=None,
                                 legacy_mode=False, before_after_indicator=False, layer_norm=False,
                                 use_gumbel_max=False,
                                 deal_with_masked_elements=False, use_dep_label=False, ignore_masks=False, relu6=False):
  """
    heads: head-dependent distribution of shape (B, seq_len, seq_len)
    labels: label distribution for each head-dependent choice, shape: (B, seq_len, labels)
  """

  heads = parse_gold
  labels = parse_labels
  assert parse_labels is not None

  token_mask_row = tf.expand_dims(tf.cast(1 - tokens_to_keep, tf.float32) * constants.VERY_SMALL, 1)
  token_mask_row_normal_scale = tf.expand_dims(tf.cast(tokens_to_keep, tf.float32), 1)
  token_mask_col = tf.expand_dims(tf.cast(tokens_to_keep, tf.float32), 2)

  # token_mask_col = tf.expand_dims(tf.cast(1 - tokens_to_keep, tf.float32) * constants.VERY_SMALL, 2)

  tf.logging.log(tf.logging.INFO, "mixture model with {} clusters and {} maximum exploring steps".format(num_clusters, (
    max_up_depth, max_down_depth)))

  def rm_sr(mtx):
    return mtx * (1 - tf.eye(tf.shape(heads)[1], batch_shape=[tf.shape(heads)[0]]))

  if len(heads.get_shape()) < 3:
    # max_seq_len = tf.cast(tf.shape(heads)[-1], dtype=tf.float32)

    if extreme_value:
      on_value = constants.VERY_LARGE
      off_value = constants.VERY_SMALL
    else:
      on_value = 10.
      off_value = -10.

    heads = tf.one_hot(heads, tf.shape(heads)[-1], off_value=off_value, on_value=on_value)
    labels = tf.one_hot(labels, 69, off_value=off_value, on_value=on_value)
    heads = heads + token_mask_row
  else:
    heads = heads + token_mask_row
    labels = tf.stop_gradient(labels)
    heads = tf.stop_gradient(heads)
    if use_gumbel_max:
      labels_dist = tfp.distributions.OneHotCategorical(logits=labels)
      heads_dist = tfp.distributions.OneHotCategorical(logits=heads)
      labels = tf.cast(tfp.distributions.Sample(labels_dist, sample_shape=()).sample(), tf.float32)
      heads = tf.cast(tfp.distributions.Sample(heads_dist, sample_shape=()).sample(), tf.float32)

  if not use_gumbel_max:
    masked_heads = tf.nn.softmax(heads, axis=-1)
    labels = tf.nn.softmax(labels, axis=-1)
  else:
    masked_heads = heads
    labels = labels

  batch_size, seq_len = tf.shape(masked_heads)[0], tf.shape(masked_heads)[1]
  gathered_tokens_to_keep = tf.gather_nd(tf.tile(tf.reshape(tokens_to_keep, [batch_size, 1, seq_len]), [1, seq_len, 1]),
                                         predicate_gather_indices)
  trigger_size = tf.shape(gathered_tokens_to_keep)[0]

  synt_mask_mtx = tf.ones_like(heads) * token_mask_row_normal_scale * token_mask_col

  masked_heads = masked_heads * token_mask_col

  up_l, down_f = masked_heads, masked_heads

  updown = rm_sr(tf.matmul(rm_sr(up_l), rm_sr(down_f), transpose_b=True))
  no_op = tf.eye(tf.shape(masked_heads)[-1], batch_shape=[tf.shape(masked_heads)[0]])
  no_op *= token_mask_row_normal_scale * token_mask_col

  # queue = [masked_heads]
  queue = [(1, (1, 0), masked_heads, 'up'), (2, (1, 1), updown, 'down'), (0, (0, 0), no_op, 'down')]
  heapq.heapify(queue)
  # queue_step = [(1, 1, 0)]
  # (steps, B, seq_len, seq_len)
  complete_list = []
  # (steps, patterns)
  complete_step = []

  MASK_STEP = len(step_to_feature_idx) + 1
  OTHERS_STEP = len(step_to_feature_idx)
  with tf.variable_scope('wfs_embedding'):
    step_embedding_mtx = tf.get_variable('step_embedding', [len(step_to_feature_idx) + 2, 64])
    # step_embedding_layer = tf.keras.layers.Embedding(len(step_to_feature_idx), 64, input_length=1, trainable=True, embeddings_initializer='GlorotNormal')

  if use_pred_role_feature:
    pred_hiddens = tf.expand_dims(pred_mlp, axis=2)
    pred_hiddens_tiled = tf.tile(pred_hiddens, [1, 1, seq_len, 1])
    role_hiddens = tf.expand_dims(pred_mlp, axis=1)
    role_hiddens_tiled = tf.tile(role_hiddens, [1, seq_len, 1, 1])
    gathered_role_mlp = tf.gather_nd(role_hiddens_tiled, predicate_gather_indices)

  with tf.variable_scope("syn_pattern_to_cluster"):
    if legacy_mode:
      dense1 = tf.keras.layers.Dense(int((101 + num_clusters) / 2), activation='relu')
    else:
      if use_pred_role_feature:
        dense1 = tf.keras.layers.Dense(256, activation='relu')
      else:
        with tf.variable_scope('batchnorm_srl_score'):
          batch_norm_1 = tf.keras.layers.BatchNormalization(
            axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
            beta_initializer='zeros', gamma_initializer='ones',
            moving_mean_initializer='zeros',
            moving_variance_initializer='ones', beta_regularizer=None,
            gamma_regularizer=None, beta_constraint=None, gamma_constraint=None
          )
          batch_norm_2 = tf.keras.layers.BatchNormalization(
            axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
            beta_initializer='zeros', gamma_initializer='ones',
            moving_mean_initializer='zeros',
            moving_variance_initializer='ones', beta_regularizer=None,
            gamma_regularizer=None, beta_constraint=None, gamma_constraint=None
          )
        dense1 = tf.keras.layers.Dense(512, activation='relu')
    dense2 = tf.keras.layers.Dense(num_clusters)

  # def explore():
  while len(queue) > 0:
    # retrieve current mtx
    total_step, step, mtx, state = heapq.heappop(queue)

    if step[0] > max_up_depth or step[1] > max_down_depth:
      continue

    if state == 'up':
      plausible_action = ['up', 'updown', 'complete' if legacy_mode else 'complete_new']
    elif state == 'down':
      plausible_action = ['down', 'complete' if legacy_mode else 'complete_new']
    else:
      raise NotImplementedError

    if 'up' in plausible_action:
      next_step = tf.matmul(mtx, masked_heads)
      heapq.heappush(queue, (total_step + 1, (step[0] + 1, step[1]), next_step, 'up'))

    if 'down' in plausible_action:
      next_step = tf.matmul(mtx, masked_heads, transpose_b=True)
      heapq.heappush(queue, (total_step + 1, (step[0], step[1] + 1), next_step, 'down'))

    if 'updown' in plausible_action:
      next_step = tf.matmul(mtx, updown)
      heapq.heappush(queue, (total_step + 2, (step[0] + 1, step[1] + 1), next_step, 'down'))

    if 'complete' in plausible_action:
      to_be_appended = tf.stop_gradient(tf.minimum(mtx, synt_mask_mtx))
      complete_list.append(to_be_appended)
      synt_mask_mtx -= to_be_appended
      complete_step.append(tf.one_hot(step_to_feature_idx[step], depth=101))
      # tf.logging.log(tf.logging.INFO, "complete searching mtx of {}".format(step))

    if 'complete_new' in plausible_action:
      addr_mtx = tf.stop_gradient(tf.minimum(mtx, synt_mask_mtx))
      synt_mask_mtx -= addr_mtx
      # addr_mtx: (n_trigger, seq, 1)
      gathered_addr_mtx = tf.expand_dims(tf.gather_nd(addr_mtx, predicate_gather_indices), axis=-1)

      step_hiddens = tf.nn.embedding_lookup(step_embedding_mtx, tf.reshape(step_to_feature_idx[step], [1]))
      step_hiddens_reshaped = tf.reshape(step_hiddens, [1, 1, 64])
      step_hiddens_gathered = tf.tile(step_hiddens_reshaped, [trigger_size, seq_len, 1])
      synt_features = [step_hiddens_gathered]

      if use_pred_role_feature:
        synt_features += [gathered_role_mlp]

      synt_features = tf.concat(synt_features, axis=-1)
      # score: (n_trigger, seq, hidden) -> (n_trigger, seq, n_cluster)
      # score = dense2(dense1(synt_features))
      score = synt_features
      complete_list.append(gathered_addr_mtx * score)

      # complete_step.append(tf.one_hot(step_to_feature_idx[step], depth=101))
      # tf.logging.log(tf.logging.INFO, "complete searching mtx of {}".format(step))

  if legacy_mode:
    complete_list.append(tf.stop_gradient(synt_mask_mtx))
    complete_step.append(tf.one_hot(100, depth=101))

    complete_step = tf.stack(complete_step, axis=0)
    z_prob = dense2(dense1(complete_step))
    if cluster_prob:
      patterns = tf.eye(101)
      # tf.logging.log("")
      z_prob_per_pattern = dense2(dense1(patterns))
      z_selection_per_pattern = tf.expand_dims(tf.math.argmax(z_prob_per_pattern, axis=-1), axis=-1)
      z_prob_per_pattern = tf.nn.softmax(z_prob_per_pattern, axis=-1)
      z_prob = tf.Print(z_prob, [z_prob_per_pattern, tf.math.less(tf.reduce_max(z_prob_per_pattern, axis=-1), 0.8)],
                        "z_probability_per_item", summarize=303)
      z_prob = tf.Print(z_prob, [z_selection_per_pattern], "z_selection_per_pattern", summarize=303)

    # it has a shape of (B, Seq, Seq, St)
    return_masks = tf.stack(complete_list, axis=-1)
    # return_masks = tf.Print(return_masks, [return_masks], summarize=30)
    # it has a shape of (B, Seq, Seq, num_clusters)
    return_masks = tf.matmul(return_masks, tf.expand_dims(tf.expand_dims(z_prob, axis=0), axis=0))
    z_logit = return_masks
  else:
    if not ignore_masks:
      non_keep_tokens = tf.expand_dims(1. - tf.cast(gathered_tokens_to_keep, tf.float32), axis=-1)
      step_hiddens = tf.nn.embedding_lookup(step_embedding_mtx, tf.reshape(MASK_STEP, [1]))
      step_hiddens_reshaped = tf.reshape(step_hiddens, [1, 1, 64])
      step_hiddens_gathered = tf.tile(step_hiddens_reshaped, [trigger_size, seq_len, 1])
      synt_features = [step_hiddens_gathered]
      if use_pred_role_feature:
        synt_features += [gathered_role_mlp]
      synt_features = tf.concat(synt_features, axis=-1)
      # score = dense2(dense1(synt_features))
      score = synt_features
      complete_list.append(non_keep_tokens * score)

    '--------------------------------'
    gathered_synt_mask_mtx = tf.expand_dims(tf.gather_nd(tf.stop_gradient(synt_mask_mtx), predicate_gather_indices),
                                            axis=-1)
    step_hiddens = tf.nn.embedding_lookup(step_embedding_mtx, tf.reshape(OTHERS_STEP, [1]))
    step_hiddens_reshaped = tf.reshape(step_hiddens, [1, 1, 64])
    step_hiddens_gathered = tf.tile(step_hiddens_reshaped, [trigger_size, seq_len, 1])
    synt_features = [step_hiddens_gathered]
    if use_pred_role_feature:
      synt_features += [gathered_role_mlp]
    synt_features = tf.concat(synt_features, axis=-1)
    # score = dense2(dense1(synt_features))
    score = synt_features
    complete_list.append(gathered_synt_mask_mtx * score)

    # stacked_complete_list = tf.stack(complete_list, axis=-1)
    # if relu6:
    #   z_logit = batch_norm_2(dense2(batch_norm_1(dense1(tf.reduce_sum(complete_list, axis=0)))))
    # else:
    #   z_logit = dense2(dense1(tf.reduce_sum(complete_list, axis=0)))

    z_logit = batch_norm_2(dense2(batch_norm_1(dense1(tf.reduce_sum(complete_list, axis=0)))))
  # output_r = None

  if mode != ModeKeys.TRAIN:
    applied_patterns = [step_to_feature_idx[(i, j)] for i in range(max_up_depth) for j in range(max_down_depth)] + [
      OTHERS_STEP, MASK_STEP]
    ap_emb = tf.nn.embedding_lookup(step_embedding_mtx, applied_patterns)
    ap_emb = tf.expand_dims(ap_emb, 0)
    patterns_assigned_cluster = batch_norm_2(dense2(batch_norm_1(dense1(ap_emb))))
    z_logit = tf.Print(z_logit, [tf.nn.softmax(z_logit), tf.argmax(z_logit, axis=-1)], "z_logit", summarize=30)
    z_logit = tf.Print(z_logit,
                       [tf.nn.softmax(patterns_assigned_cluster), tf.argmax(patterns_assigned_cluster, axis=-1)],
                       "cluster_assignment", summarize=max_up_depth * max_down_depth + 10)

  if inference_mode == 'full':
    z_prob = tf.nn.softmax(z_logit, -1)
  elif inference_mode == 'nll' or inference_mode == 'nll_normal':
    z_prob = z_logit
  elif inference_mode == 'gumbel':
    if mode != ModeKeys.TRAIN:
      temperature = 1e-5
      dist = tfp.distributions.RelaxedOneHotCategorical(temperature, logits=z_logit)
      sample = tfp.distributions.Sample(dist)
      z_prob = sample.sample()
    else:
      current_step = tf.train.get_global_step()
      temperature = tf.maximum(0.5, tf.exp(-1e-5 * tf.cast(current_step, tf.float32)))
      dist = tfp.distributions.RelaxedOneHotCategorical(temperature, logits=z_logit)
      sample = tfp.distributions.Sample(dist)
      z_prob = sample.sample()
    tf.logging.log(tf.logging.INFO, "z_sample:{}".format(z_prob))
  else:
    raise NotImplementedError
  additional_loss = 0.
  return z_prob, additional_loss


def get_dep_transition_wfs_binary(parse_gold, mode, tokens_to_keep=None, extreme_value=False, parse_labels=None,
                                  use_lstm=False, return_last=False, max_up_depth=5, max_down_depth=3, num_clusters=3,
                                  inference_mode='full', gumbel_temperature=1e-5, cluster_prob=False,
                                  propagation_depth=0,
                                  pred_mlp=None, role_mlp=None, use_pred_role_feature=False,
                                  predicate_gather_indices=None,
                                  legacy_mode=False, before_after_indicator=False, layer_norm=False,
                                  use_gumbel_max=False,
                                  deal_with_masked_elements=False, use_dep_label=False, ignore_masks=False):
  """
    heads: head-dependent distribution of shape (B, seq_len, seq_len)
    labels: label distribution for each head-dependent choice, shape: (B, seq_len, labels)
  """

  heads = parse_gold
  labels = parse_labels

  token_mask_row = tf.expand_dims(tf.cast(1 - tokens_to_keep, tf.float32) * constants.VERY_SMALL, 1)
  token_mask_row_normal_scale = tf.expand_dims(tf.cast(tokens_to_keep, tf.float32), 1)
  token_mask_col = tf.expand_dims(tf.cast(tokens_to_keep, tf.float32), 2)

  # token_mask_col = tf.expand_dims(tf.cast(1 - tokens_to_keep, tf.float32) * constants.VERY_SMALL, 2)

  tf.logging.log(tf.logging.INFO, "mixture model with {} clusters and {} maximum exploring steps".format(num_clusters, (
    max_up_depth, max_down_depth)))

  def rm_sr(mtx):
    return mtx * (1 - tf.eye(tf.shape(heads)[1], batch_shape=[tf.shape(heads)[0]]))

  if len(heads.get_shape()) < 3:
    # max_seq_len = tf.cast(tf.shape(heads)[-1], dtype=tf.float32)

    if extreme_value:
      on_value = constants.VERY_LARGE
      off_value = constants.VERY_SMALL
    else:
      on_value = 10.
      off_value = -10.

    heads = tf.one_hot(heads, tf.shape(heads)[-1], off_value=off_value, on_value=on_value)
    labels = tf.one_hot(labels, 69, off_value=off_value, on_value=on_value)
    heads = heads + token_mask_row
  else:
    heads = heads + token_mask_row
    labels = tf.stop_gradient(labels)
    heads = tf.stop_gradient(heads)
    if use_gumbel_max:
      labels_dist = tfp.distributions.OneHotCategorical(logits=labels)
      heads_dist = tfp.distributions.OneHotCategorical(logits=heads)
      labels = tf.cast(tfp.distributions.Sample(labels_dist, sample_shape=()).sample(), tf.float32)
      heads = tf.cast(tfp.distributions.Sample(heads_dist, sample_shape=()).sample(), tf.float32)

  if not use_gumbel_max:
    masked_heads = tf.nn.softmax(heads, axis=-1)
    labels = tf.nn.softmax(labels, axis=-1)
  else:
    masked_heads = heads
    labels = labels

  batch_size, seq_len = tf.shape(masked_heads)[0], tf.shape(masked_heads)[1]
  gathered_tokens_to_keep = tf.gather_nd(tf.tile(tf.reshape(tokens_to_keep, [batch_size, 1, seq_len]), [1, seq_len, 1]),
                                         predicate_gather_indices)
  trigger_size = tf.shape(gathered_tokens_to_keep)[0]

  synt_mask_mtx = tf.ones_like(heads) * token_mask_row_normal_scale * token_mask_col

  masked_heads = masked_heads * token_mask_col

  up_l, down_f = masked_heads, masked_heads

  updown = rm_sr(tf.matmul(rm_sr(up_l), rm_sr(down_f), transpose_b=True))
  no_op = tf.eye(tf.shape(masked_heads)[-1], batch_shape=[tf.shape(masked_heads)[0]])
  no_op *= token_mask_row_normal_scale * token_mask_col

  # queue = [masked_heads]
  queue = [(1, (1, 0), masked_heads, 'up'), (2, (1, 1), updown, 'down'), (0, (0, 0), no_op, 'down')]
  heapq.heapify(queue)
  # queue_step = [(1, 1, 0)]
  # (steps, B, seq_len, seq_len)
  complete_list = []
  # (steps, patterns)
  complete_step = []

  MASK_STEP = len(step_to_feature_idx) + 1
  OTHERS_STEP = len(step_to_feature_idx)
  # with tf.variable_scope('wfs_embedding'):
  #   step_embedding_mtx = tf.get_variable('step_embedding', [len(step_to_feature_idx)+2, 64])
  # step_embedding_layer = tf.keras.layers.Embedding(len(step_to_feature_idx), 64, input_length=1, trainable=True, embeddings_initializer='GlorotNormal')

  if use_pred_role_feature:
    pred_hiddens = tf.expand_dims(pred_mlp, axis=2)
    pred_hiddens_tiled = tf.tile(pred_hiddens, [1, 1, seq_len, 1])
    role_hiddens = tf.expand_dims(pred_mlp, axis=1)
    role_hiddens_tiled = tf.tile(role_hiddens, [1, seq_len, 1, 1])
    gathered_role_mlp = tf.gather_nd(role_hiddens_tiled, predicate_gather_indices)

  with tf.variable_scope("syn_pattern_to_cluster"):
    if legacy_mode:
      dense1 = tf.keras.layers.Dense(int((101 + num_clusters) / 2), activation='relu')
    else:
      if use_pred_role_feature:
        dense1 = tf.keras.layers.Dense(256, activation='relu')
      else:
        dense1 = tf.keras.layers.Dense(512, activation='relu')
    dense2 = tf.keras.layers.Dense(num_clusters)

  # def explore():
  while len(queue) > 0:
    # retrieve current mtx
    total_step, step, mtx, state = heapq.heappop(queue)

    if step[0] > max_up_depth or step[1] > max_down_depth:
      continue

    if state == 'up':
      plausible_action = ['up', 'updown', 'complete' if legacy_mode else 'complete_new']
    elif state == 'down':
      plausible_action = ['down', 'complete' if legacy_mode else 'complete_new']
    else:
      raise NotImplementedError

    if 'up' in plausible_action:
      next_step = tf.matmul(mtx, masked_heads)
      heapq.heappush(queue, (total_step + 1, (step[0] + 1, step[1]), next_step, 'up'))

    if 'down' in plausible_action:
      next_step = tf.matmul(mtx, masked_heads, transpose_b=True)
      heapq.heappush(queue, (total_step + 1, (step[0], step[1] + 1), next_step, 'down'))

    if 'updown' in plausible_action:
      next_step = tf.matmul(mtx, updown)
      heapq.heappush(queue, (total_step + 2, (step[0] + 1, step[1] + 1), next_step, 'down'))

    if 'complete' in plausible_action:
      to_be_appended = tf.stop_gradient(tf.minimum(mtx, synt_mask_mtx))
      complete_list.append(to_be_appended)
      synt_mask_mtx -= to_be_appended
      complete_step.append(tf.one_hot(step_to_feature_idx[step], depth=101))
      # tf.logging.log(tf.logging.INFO, "complete searching mtx of {}".format(step))

    if 'complete_new' in plausible_action:
      addr_mtx = tf.stop_gradient(tf.minimum(mtx, synt_mask_mtx))
      synt_mask_mtx -= addr_mtx
      # addr_mtx: (n_trigger, seq, 1)
      gathered_addr_mtx = tf.expand_dims(tf.gather_nd(addr_mtx, predicate_gather_indices), axis=-1)

      # step_hiddens = tf.nn.embedding_lookup(step_embedding_mtx, tf.reshape(MASK_STEP, [1]))
      step_hiddens_reshaped = tf.reshape(tf.one_hot(1, depth=2), [1, 1, 2])
      step_hiddens_gathered = tf.tile(step_hiddens_reshaped, [trigger_size, seq_len, 1])
      synt_features = [step_hiddens_gathered]

      if use_pred_role_feature:
        synt_features += [gathered_role_mlp]

      synt_features = tf.concat(synt_features, axis=-1)
      # score: (n_trigger, seq, hidden) -> (n_trigger, seq, n_cluster)
      # score = dense2(dense1(synt_features))
      score = synt_features
      complete_list.append(gathered_addr_mtx * score)

      # complete_step.append(tf.one_hot(step_to_feature_idx[step], depth=101))
      # tf.logging.log(tf.logging.INFO, "complete searching mtx of {}".format(step))

  if legacy_mode:
    complete_list.append(tf.stop_gradient(synt_mask_mtx))
    complete_step.append(tf.one_hot(100, depth=101))

    complete_step = tf.stack(complete_step, axis=0)
    z_prob = dense2(dense1(complete_step))
    if cluster_prob:
      patterns = tf.eye(101)
      # tf.logging.log("")
      z_prob_per_pattern = dense2(dense1(patterns))
      z_selection_per_pattern = tf.expand_dims(tf.math.argmax(z_prob_per_pattern, axis=-1), axis=-1)
      z_prob_per_pattern = tf.nn.softmax(z_prob_per_pattern, axis=-1)
      z_prob = tf.Print(z_prob, [z_prob_per_pattern, tf.math.less(tf.reduce_max(z_prob_per_pattern, axis=-1), 0.8)],
                        "z_probability_per_item", summarize=303)
      z_prob = tf.Print(z_prob, [z_selection_per_pattern], "z_selection_per_pattern", summarize=303)

    # it has a shape of (B, Seq, Seq, St)
    return_masks = tf.stack(complete_list, axis=-1)
    # return_masks = tf.Print(return_masks, [return_masks], summarize=30)
    # it has a shape of (B, Seq, Seq, num_clusters)
    return_masks = tf.matmul(return_masks, tf.expand_dims(tf.expand_dims(z_prob, axis=0), axis=0))
    z_logit = return_masks
  else:
    if not ignore_masks:
      non_keep_tokens = tf.expand_dims(1. - tf.cast(gathered_tokens_to_keep, tf.float32), axis=-1)
      # step_hiddens = tf.nn.embedding_lookup(step_embedding_mtx, tf.reshape(MASK_STEP, [1]))
      step_hiddens_reshaped = tf.reshape(tf.one_hot(1, depth=2), [1, 1, 2])
      step_hiddens_gathered = tf.tile(step_hiddens_reshaped, [trigger_size, seq_len, 1])
      synt_features = [step_hiddens_gathered]
      if use_pred_role_feature:
        synt_features += [gathered_role_mlp]
      synt_features = tf.concat(synt_features, axis=-1)
      # score = dense2(dense1(synt_features))
      score = synt_features
      complete_list.append(non_keep_tokens * score)

    '--------------------------------'
    gathered_synt_mask_mtx = tf.expand_dims(tf.gather_nd(tf.stop_gradient(synt_mask_mtx), predicate_gather_indices),
                                            axis=-1)
    # step_hiddens = tf.nn.embedding_lookup(step_embedding_mtx, tf.reshape(OTHER_STEP, [1]))
    step_hiddens_reshaped = tf.reshape(tf.one_hot(0, depth=2), [1, 1, 2])
    step_hiddens_gathered = tf.tile(step_hiddens_reshaped, [trigger_size, seq_len, 1])
    synt_features = [step_hiddens_gathered]
    if use_pred_role_feature:
      synt_features += [gathered_role_mlp]
    synt_features = tf.concat(synt_features, axis=-1)
    # score = dense2(dense1(synt_features))
    score = synt_features
    complete_list.append(gathered_synt_mask_mtx * score)

    # stacked_complete_list = tf.stack(complete_list, axis=-1)
    # z_logit = dense2(dense1(tf.reduce_sum(complete_list, axis=0)))
    z_logit = tf.reduce_sum(complete_list, axis=0)
  # output_r = None

  if mode != ModeKeys.TRAIN:
    z_logit = tf.Print(z_logit, [tf.nn.softmax(z_logit), tf.argmax(z_logit, axis=-1)], "z_logit", summarize=30)

  if inference_mode == 'full':
    z_prob = tf.nn.softmax(z_logit, -1)
  elif inference_mode == 'nll' or inference_mode == 'nll_normal':
    z_prob = z_logit
  elif inference_mode == 'gumbel':
    if mode != ModeKeys.TRAIN:
      temperature = 1e-5
      dist = tfp.distributions.RelaxedOneHotCategorical(temperature, logits=z_logit)
      sample = tfp.distributions.Sample(dist)
      z_prob = sample.sample()
    else:
      current_step = tf.train.get_global_step()
      temperature = tf.maximum(0.5, tf.exp(-1e-5 * tf.cast(current_step, tf.float32)))
      dist = tfp.distributions.RelaxedOneHotCategorical(temperature, logits=z_logit)
      sample = tfp.distributions.Sample(dist)
      z_prob = sample.sample()
    tf.logging.log(tf.logging.INFO, "z_sample:{}".format(z_prob))
  else:
    raise NotImplementedError
  additional_loss = 0.
  return z_prob, additional_loss


# def get_dep_transition_wfs_normal(parse_gold, mode, tokens_to_keep=None, extreme_value=False, parse_labels=None,
#                            use_lstm=False, return_last=False, max_up_depth=5, max_down_depth=3, num_clusters=3,
#                            inference_mode='full', gumbel_temperature=1e-5, cluster_prob=False, propagation_depth=0,
#                            pred_mlp = None, role_mlp = None, use_pred_role_feature=False, predicate_gather_indices=None,
#                            legacy_mode = False, before_after_indicator = False, layer_norm=False, num_samples=1):
#   """
#     heads: head-dependent distribution of shape (B, seq_len, seq_len)
#     labels: label distribution for each head-dependent choice, shape: (B, seq_len, labels)
#   """
#
#   heads = parse_gold
#   labels = parse_labels
#   token_mask_row = tf.expand_dims(tf.cast(1 - tokens_to_keep, tf.float32) * constants.VERY_SMALL, 1)
#   token_mask_row_normal_scale = tf.expand_dims(tf.cast(tokens_to_keep, tf.float32), 1)
#   token_mask_col = tf.expand_dims(tf.cast(tokens_to_keep, tf.float32), 2)
#
#   # token_mask_col = tf.expand_dims(tf.cast(1 - tokens_to_keep, tf.float32) * constants.VERY_SMALL, 2)
#
#   tf.logging.log(tf.logging.INFO, "mixture model with {} clusters and {} maximum exploring steps".format(num_clusters, (
#   max_up_depth, max_down_depth)))
#
#   def rm_sr(mtx):
#     return mtx * (1 - tf.eye(tf.shape(heads)[1], batch_shape=[tf.shape(heads)[0]]))
#
#   if len(heads.get_shape()) < 3:
#     # max_seq_len = tf.cast(tf.shape(heads)[-1], dtype=tf.float32)
#
#     if extreme_value:
#       on_value = constants.VERY_LARGE
#       off_value = constants.VERY_SMALL
#     else:
#       on_value = 10.
#       off_value = -10.
#
#     heads = tf.one_hot(heads, tf.shape(heads)[-1], off_value=off_value, on_value=on_value)
#     heads = heads + token_mask_row
#   else:
#     heads = heads + token_mask_row
#     heads = tf.stop_gradient(heads)
#
#   masked_heads = tf.nn.softmax(heads, axis=-1)
#
#   # syn_feature_mtx = tf.zeros_like(heads)
#   synt_mask_mtx = tf.ones_like(heads) * token_mask_row_normal_scale * token_mask_col
#
#   masked_heads = masked_heads * token_mask_col
#
#   up_l, down_f = masked_heads, masked_heads
#
#   updown = rm_sr(tf.matmul(rm_sr(up_l), rm_sr(down_f), transpose_b=True))
#   no_op = tf.eye(tf.shape(masked_heads)[-1], batch_shape=[tf.shape(masked_heads)[0]])
#   no_op *= token_mask_row_normal_scale * token_mask_col
#
#   # queue = [masked_heads]
#   queue = [(1, (1, 0), masked_heads, 'up'), (2, (1, 1), updown, 'down'), (0, (0, 0), no_op, 'down')]
#   heapq.heapify(queue)
#   # queue_step = [(1, 1, 0)]
#   # (steps, B, seq_len, seq_len)
#   complete_list_bias = []
#   complete_list_var = []
#   # (steps, patterns)
#   complete_step = []
#
#   with tf.variable_scope('wfs_embedding'):
#     step_embedding_mtx = tf.get_variable('step_embedding', [len(step_to_feature_idx), 64])
#     # step_embedding_layer = tf.keras.layers.Embedding(len(step_to_feature_idx), 64, input_length=1, trainable=True, embeddings_initializer='GlorotNormal')
#   batch_size, seq_len = tf.shape(masked_heads)[0], tf.shape(masked_heads)[1]
#   pred_hiddens = tf.expand_dims(pred_mlp, axis=2)
#   pred_hiddens_tiled = tf.tile(pred_hiddens, [1, 1, seq_len, 1])
#   role_hiddens = tf.expand_dims(pred_mlp, axis=1)
#   role_hiddens_tiled = tf.tile(role_hiddens, [1, seq_len, 1, 1])
#   gathered_pred_mlp = tf.gather_nd(pred_hiddens_tiled ,predicate_gather_indices)
#   gathered_role_mlp = tf.gather_nd(role_hiddens_tiled ,predicate_gather_indices)
#   trigger_size = tf.shape(gathered_pred_mlp)[0]
#
#   with tf.variable_scope("syn_pattern_to_cluster_normal"):
#     with tf.variable_scope("bias"):
#       bias_dense1 = tf.keras.layers.Dense(256, activation='relu')
#       bias_dense2 = tf.keras.layers.Dense(64)
#     with tf.variable_scope("variance"):
#       var_dense1 = tf.keras.layers.Dense(256, activation='relu')
#       var_dense2 = tf.keras.layers.Dense(64)
#
#
#
#   # def explore():
#   while len(queue) > 0:
#     # retrieve current mtx
#     total_step, step, mtx, state = heapq.heappop(queue)
#
#     if step[0] > max_up_depth or step[1] > max_down_depth:
#       continue
#
#     if state == 'up':
#       plausible_action = ['up', 'updown', 'complete' if legacy_mode else 'complete_new']
#     elif state == 'down':
#       plausible_action = ['down', 'complete' if legacy_mode else 'complete_new']
#     else:
#       raise NotImplementedError
#
#     if 'up' in plausible_action:
#       next_step = tf.matmul(mtx, masked_heads)
#       heapq.heappush(queue, (total_step + 1, (step[0] + 1, step[1]), next_step, 'up'))
#
#     if 'down' in plausible_action:
#       next_step = tf.matmul(mtx, masked_heads, transpose_b=True)
#       heapq.heappush(queue, (total_step + 1, (step[0], step[1] + 1), next_step, 'down'))
#
#     if 'updown' in plausible_action:
#       next_step = tf.matmul(mtx, updown)
#       heapq.heappush(queue, (total_step + 2, (step[0] + 1, step[1] + 1), next_step, 'down'))
#     if 'complete' in plausible_action:
#       raise NotImplementedError
#
#     if 'complete_new' in plausible_action:
#       addr_mtx = tf.stop_gradient(tf.minimum(mtx, synt_mask_mtx))
#       synt_mask_mtx -= addr_mtx
#       # addr_mtx: (n_trigger, seq)
#       gathered_addr_mtx = tf.expand_dims(tf.gather_nd(addr_mtx ,predicate_gather_indices), axis=-1)
#
#       step_hiddens = tf.nn.embedding_lookup(step_embedding_mtx, tf.reshape(step_to_feature_idx[step], [1]))
#       step_hiddens_reshaped = tf.reshape(step_hiddens, [1, 1, 64])
#       step_hiddens_gathered = tf.tile(step_hiddens_reshaped, [trigger_size, seq_len, 1])
#       synt_features = [step_hiddens_gathered]
#
#       synt_features = tf.concat(synt_features, axis=-1)
#       joint_feature = synt_features
#       # score: (n_trigger, seq, hidden) -> (n_trigger, seq, n_cluster)
#       score_bias = bias_dense2(bias_dense1(joint_feature))
#       score_var = var_dense2(var_dense1(joint_feature))
#       complete_list_bias.append(gathered_addr_mtx*score_bias)
#       complete_list_var.append(gathered_addr_mtx*score_var)
#
#       # complete_step.append(tf.one_hot(step_to_feature_idx[step], depth=101))
#       # tf.logging.log(tf.logging.INFO, "complete searching mtx of {}".format(step))
#
#   gathered_synt_mask_mtx = tf.expand_dims(tf.gather_nd(tf.stop_gradient(synt_mask_mtx), predicate_gather_indices), axis=-1)
#   step_hiddens = tf.nn.embedding_lookup(step_embedding_mtx, tf.reshape(100, [1]))
#   step_hiddens_reshaped = tf.reshape(step_hiddens, [1, 1, 64])
#   step_hiddens_gathered = tf.tile(step_hiddens_reshaped, [trigger_size, seq_len, 1])
#   synt_features = [step_hiddens_gathered]
#
#   synt_features = tf.concat(synt_features, axis=-1)
#   joint_feature = synt_features
#
#   score_bias = bias_dense2(bias_dense1(joint_feature))
#   score_var = var_dense2(var_dense1(joint_feature))
#   complete_list_bias.append(gathered_synt_mask_mtx * score_bias)
#   complete_list_var.append(gathered_synt_mask_mtx * score_var)
#
#
#   stacked_complete_list_bias = tf.stack(complete_list_bias, axis=-1)
#   stacked_complete_list_var = tf.stack(complete_list_var, axis=-1)
#   # 1e-3 + tf.math.softplus(var_dense2(var_dense1(joint_feature)))
#   z_bias = tf.reduce_sum(stacked_complete_list_bias, axis=-1)
#   z_var = 1e-3 + tf.math.softplus(tf.reduce_sum(stacked_complete_list_var, axis=-1))
#
#   if mode != ModeKeys.TRAIN:
#     z_bias = tf.Print(z_bias, [z_bias], "z_bias", summarize=30)
#     z_var = tf.Print(z_var, [z_var], "z_var", summarize=30)
#
#   dist = tfp.distributions.Normal(z_bias, z_var, validate_args=False, allow_nan_stats=True, name='Normal')
#   sample = tfp.distributions.Sample(dist, sample_shape=(num_samples))
#   ref_dist = tfp.distributions.Normal(tf.zeros_like(z_bias), tf.ones_like(z_var), name='Normal_reference')
#   z_var = sample.sample()
#
#   kl_div = dist.kl_divergence(ref_dist)
#
#   return z_var, kl_div


# def get_dep_transition_wfs_with_hiddens(
#         parse_gold, hiddens, mode, predicate_gather_indices, tokens_to_keep=None, extreme_value=False,
#         parse_labels=None,
#         use_lstm=False, return_last=False, max_up_depth=6, max_down_depth=3, num_clusters=3,
#         inference_mode='full', gumbel_temperature=1e-5, cluster_prob=False, propagation_depth=0,
#         return_gathered_hiddens=False, half_return_size=False):
#   """
#     heads: head-dependent distribution of shape (B, seq_len, seq_len)
#     labels: label distribution for each head-dependent choice, shape: (B, seq_len, labels)
#   """
#
#   heads = parse_gold
#   labels = parse_labels
#   token_mask_row = tf.expand_dims(tf.cast(1 - tokens_to_keep, tf.float32) * constants.VERY_SMALL, 1)
#   token_mask_row_normal_scale = tf.expand_dims(tf.cast(tokens_to_keep, tf.float32), 1)
#   token_mask_col = tf.expand_dims(tf.cast(tokens_to_keep, tf.float32), 2)
#
#   # token_mask_col = tf.expand_dims(tf.cast(1 - tokens_to_keep, tf.float32) * constants.VERY_SMALL, 2)
#
#   tf.logging.log(tf.logging.INFO, "mixture model with {} clusters and {} maximum exploring steps".format(num_clusters, (
#   max_up_depth, max_down_depth)))
#
#   def rm_sr(mtx):
#     return mtx * (1 - tf.eye(tf.shape(heads)[1], batch_shape=[tf.shape(heads)[0]]))
#
#   if len(heads.get_shape()) < 3:
#     # max_seq_len = tf.cast(tf.shape(heads)[-1], dtype=tf.float32)
#
#     if extreme_value:
#       on_value = constants.VERY_LARGE
#       off_value = constants.VERY_SMALL
#     else:
#       on_value = 10.
#       off_value = -10.
#
#     heads = tf.one_hot(heads, tf.shape(heads)[-1], off_value=off_value, on_value=on_value)
#     heads = heads + token_mask_row
#   else:
#
#     heads = heads + token_mask_row
#     heads = tf.stop_gradient(heads)
#
#   masked_heads = tf.nn.softmax(heads, axis=-1)
#
#   # syn_feature_mtx = tf.zeros_like(heads)
#   synt_mask_mtx = tf.ones_like(heads) * token_mask_row_normal_scale * token_mask_col
#
#   masked_heads = masked_heads * token_mask_col
#
#   up_l, down_f = masked_heads, masked_heads
#
#   updown = rm_sr(tf.matmul(rm_sr(up_l), rm_sr(down_f), transpose_b=True))
#   no_op = tf.eye(tf.shape(masked_heads)[-1], batch_shape=[tf.shape(masked_heads)[0]])
#   no_op *= token_mask_row_normal_scale * token_mask_col
#
#   batch_size, seq_len = tf.shape(hiddens)[0], tf.shape(hiddens)[1]
#   hidden_size = hiddens.get_shape().as_list()[2]
#   is_training = mode == ModeKeys.TRAIN
#   lstm_cell = tf.keras.layers.GRUCell(
#     int(hidden_size / 2), activation='tanh', recurrent_activation='sigmoid',
#     use_bias=True, kernel_initializer='glorot_uniform',
#     recurrent_initializer='orthogonal',
#     bias_initializer='zeros', kernel_regularizer=None,
#     recurrent_regularizer=None, bias_regularizer=None, kernel_constraint=None,
#     recurrent_constraint=None, bias_constraint=None, dropout=0.1,
#     recurrent_dropout=0.1, reset_after=True
#   )
#   # lstm_init_state = lstm_cell.get_initial_state(batch_size=batch_size*seq_len)
#
#   gathered_adjmtx_up = tf.stop_gradient(
#     tf.gather_nd(tf.reshape(tf.tile(masked_heads, [1, seq_len, 1]), [batch_size, seq_len, seq_len, seq_len]),
#                  predicate_gather_indices))
#   gathered_adjmtx_down = tf.stop_gradient(tf.gather_nd(
#     tf.reshape(tf.tile(tf.transpose(masked_heads, [0, 2, 1]), [1, seq_len, 1]),
#                [batch_size, seq_len, seq_len, seq_len]),
#     predicate_gather_indices))
#
#   def lstm_forward(lstm_state, input):
#     # hidden to be of (Gat_Pred, Seq, H)
#     tf.logging.log(tf.logging.INFO, "three_tensors {} {}".format(lstm_state, input))
#     lstm_state_reshaped = tf.reshape(lstm_state, [-1, int(hidden_size / 2)])
#     input_reshaped = tf.reshape(input, [-1, hidden_size])
#     new_lstm_state, _ = lstm_cell(input_reshaped, [lstm_state_reshaped], training=is_training)
#     return tf.reshape(new_lstm_state, tf.shape(lstm_state))
#
#   def lstm_forward_adv(adjmtx, current_activation, gathered_hidden, slot, down=False, zero_init=False):
#     gathered_activation = tf.gather_nd(current_activation, predicate_gather_indices)
#     # gathered_adjmtx = tf.gather_nd(tf.reshape(tf.tile(adjmtx, [1, seq_len, 1]), [batch_size, seq_len, seq_len, seq_len]), predicate_gather_indices)
#     # gathered_adjmtx = tf.stop_gradient(gathered_adjmtx)
#     if down:
#       gathered_adjmtx = gathered_adjmtx_down
#     else:
#       gathered_adjmtx = gathered_adjmtx_up
#
#     lstm_state = tf.matmul(gathered_adjmtx, slot)
#     if zero_init:
#       lstm_state = slot
#     new_lstm_state = lstm_forward(lstm_state, gathered_hidden)
#     slot += new_lstm_state * tf.stop_gradient(tf.expand_dims(gathered_activation, axis=-1))
#     return slot
#
#     pass
#
#   # gathered_predicates = tf.expand_dims(tf.gather_nd(predicate_mlp, predicate_gather_indices), 1)
#
#   tiled_hidden = tf.reshape(tf.tile(hiddens, [1, seq_len, 1]), shape=[batch_size, seq_len, seq_len, hidden_size])
#   gathered_hidden = tf.gather_nd(tiled_hidden, predicate_gather_indices)
#   slot, final_slot = tf.split(tf.zeros_like(gathered_hidden), num_or_size_splits=2, axis=-1)
#   # final_slot = tf.zeros_like(gathered_hidden)
#
#   zero_init_activation = tf.eye(seq_len, batch_shape=[batch_size]) * token_mask_row_normal_scale * token_mask_col
#   zero_init_adjmtx = tf.eye(seq_len, batch_shape=[batch_size]) * token_mask_row_normal_scale * token_mask_col
#   slot_zero_init = lstm_forward_adv(zero_init_adjmtx, zero_init_activation, gathered_hidden, slot, down=False,
#                                     zero_init=True)
#
#   slot_up_init = lstm_forward_adv(masked_heads, masked_heads, gathered_hidden, slot_zero_init, down=False)
#   slot_updown_init = lstm_forward_adv(masked_heads, updown, gathered_hidden, slot_zero_init, down=True)
#
#   # updown_init_output, updown_init_lstm = lstm_forward(up_init_lstm, tf.linalg.matmul(updown, hiddens, adjoint_a=True))
#
#   # queue = [masked_heads]
#   queue = [(1, (1, 0), masked_heads, 'up', slot_zero_init, slot_zero_init),
#            (2, (1, 1), updown, 'down', slot_up_init, slot_up_init),
#            (0, (0, 0), no_op, 'down', slot_updown_init, slot_updown_init)]
#   heapq.heapify(queue)
#   # queue_step = [(1, 1, 0)]
#   # (steps, B, seq_len, seq_len)
#   complete_list = []
#   # (steps, patterns)
#   complete_step = []
#   complete_hidden_by_lstm = []
#
#   # def explore():
#   while len(queue) > 0:
#     # retrieve current mtx
#     total_step, step, mtx, state, _, slot = heapq.heappop(queue)
#
#     if step[0] > max_up_depth or step[1] > max_down_depth:
#       continue
#
#     if state == 'up':
#       plausible_action = ['up', 'updown', 'complete']
#     elif state == 'down':
#       plausible_action = ['down', 'complete']
#     else:
#       raise NotImplementedError
#
#     if 'up' in plausible_action:
#       next_step = tf.matmul(mtx, masked_heads)
#       # next_step_hidden = tf.matmul(next_step, hiddens, adjoint_a=True)
#       # output, next_step_lstm = lstm_forward(lstm_state, next_step_hidden)
#       new_slot = lstm_forward_adv(masked_heads, next_step, gathered_hidden, slot, down=False)
#       heapq.heappush(queue, (total_step + 1, (step[0] + 1, step[1]), next_step, 'up', new_slot, new_slot))
#
#     if 'down' in plausible_action:
#       next_step = tf.matmul(mtx, masked_heads, transpose_b=True)
#       # next_step_hidden = tf.matmul(next_step, hiddens, adjoint_a=True)
#       new_slot = lstm_forward_adv(masked_heads, next_step, gathered_hidden, slot, down=True)
#       heapq.heappush(queue, (total_step + 1, (step[0], step[1] + 1), next_step, 'down', new_slot, new_slot))
#
#     if 'updown' in plausible_action:
#       next_step_1 = tf.matmul(mtx, masked_heads)
#       next_step_2 = tf.matmul(mtx, updown)
#       new_slot_1 = lstm_forward_adv(masked_heads, next_step_1, gathered_hidden, slot, down=False)
#       new_slot_2 = lstm_forward_adv(masked_heads, next_step_2, gathered_hidden, new_slot_1, down=True)
#       heapq.heappush(queue, (total_step + 2, (step[0] + 1, step[1] + 1), next_step_2, 'down', new_slot_2, new_slot_2))
#
#     if 'complete' in plausible_action:
#       to_be_appended = tf.stop_gradient(tf.minimum(mtx, synt_mask_mtx))
#       to_be_appended_reshaped = tf.expand_dims(tf.gather_nd(to_be_appended, predicate_gather_indices), axis=-1)
#       # complete_list.append(to_be_appended)
#       synt_mask_mtx -= to_be_appended
#       # complete_step.append(tf.one_hot(step_to_feature_idx[step], depth=101))
#       tf.logging.log(tf.logging.INFO, "search exit, with lstm state of {}".format(slot))
#       # lstm_state_reshaped = tf.reshape(slot, shape=[batch_size, -1, hidden_size])
#       # complete_hidden_by_lstm.append(lstm_state_reshaped)
#       final_slot += to_be_appended_reshaped * slot
#       # tf.logging.log(tf.logging.INFO, "complete searching mtx of {}".format(step))
#
#   # complete_list.append(tf.stop_gradient(synt_mask_mtx))
#   remainders = tf.expand_dims(tf.gather_nd(synt_mask_mtx, predicate_gather_indices), axis=-1)
#   # complete_hidden_by_lstm.append(hiddens)
#   with tf.variable_scope("input_to_hidden"):
#     dense = tf.keras.layers.Dense(int(hidden_size) / 2)
#     final_slot += tf.stop_gradient(remainders) * dense(gathered_hidden)
#   # complete_step.append(tf.one_hot(100, depth=101))
#
#   with tf.variable_scope("syn_pattern_to_cluster"):
#     # complete_step = tf.stack(complete_hidden_by_lstm, axis=-2)
#     dense_1 = tf.keras.layers.Dense(hidden_size*2, activation="relu")
#     dense_2 = tf.keras.layers.Dense(num_clusters)
#     # it has a shape of (st, number_clusters)
#     z_logit = dense_2(dense_1(final_slot))
#     if mode != ModeKeys.TRAIN:
#       z_logit = tf.Print(z_logit, [tf.nn.softmax(z_logit)], "z_logit", summarize=30)
#
#   #############
#   # building returning hidden states
#   #############
#   # it has a shape of (B, Seq, Seq, St) Seq->Seq: pred->arg accessibility
#   # addressing_mtx = tf.stack(complete_list, axis=-1)
#   # # it has a shape of (B, Seq, St, H) Seq->Seq: pred->arg accessibility
#   # hidden_mtx = tf.stack(complete_hidden_by_lstm, axis=-2)
#   # hidden_mtx = tf.matmul(addressing_mtx, hidden_mtx)
#   # it has a shape of (B, Seq, Seq, H) Seq->Seq: pred->arg by route
#   if return_gathered_hiddens:
#     if half_return_size:
#       with tf.variable_scope("input_to_hidden"):
#         dense = tf.keras.layers.Dense(int(hidden_size) / 2)
#         output_r = dense(gathered_hidden)
#     else:
#       output_r = gathered_hidden
#   else:
#     output_r = final_slot
#
#   #############
#   # building assigning scorer
#   #############
#   # it has a shape of (B, Seq, Seq, St)
#
#   # return_masks = tf.gather_nd(tf.stack(complete_list, axis=-1), predicate_gather_indices)
#   # # return_masks = tf.Print(return_masks, [return_masks], summarize=30)
#   # # it has a shape of (B, Seq, Seq, num_clusters)
#   # return_masks = tf.matmul(return_masks, z_logit)
#   # # return_masks = tf.Print(return_masks, [return_masks], summarize=30)
#   # z_logit = return_masks
#
#   if inference_mode == 'full':
#     z_prob = tf.nn.softmax(z_logit, -1)
#   elif inference_mode == 'nll' or inference_mode == 'nll_normal':
#     z_prob = z_logit
#   else:
#     raise NotImplementedError
#
#   if mode != ModeKeys.TRAIN:
#     z_prob = tf.Print(z_prob, [z_prob], summarize=30)
#
#   return z_prob, output_r
def get_dep_transition_wfs_dp(
    parse_gold, parse_labels, mode, predicate_gather_indices, tokens_to_keep=None, extreme_value=False,
    batched_predicate_gather_indices=None,
    max_up_depth=5, max_down_depth=4, num_clusters=3,
    inference_mode='full',
    return_gathered_hiddens=False, half_return_size=False, l1_regularizer=False,
    layer_norm=False, use_bai=False, num_samples=1, hiddens=None, latent_hidden_size=64, use_direction=True,
    use_lr_dir=False,
    use_dep_label=False, use_gumbel_max=False, force_to_learn_count=False, returns_lstm_state=False,
    use_trigger_batch=False, use_fixed_pattern=False):
  """
    heads: head-dependent distribution of shape (B, seq_len, seq_len)
    labels: label distribution for each head-dependent choice, shape: (B, seq_len, labels)
  """
  assert use_trigger_batch
  # FIX PATTERN part needs to adapt to non-trigger batch mode

  # use_gumbel_max=True
  heads = parse_gold
  labels = parse_labels
  token_mask_row = tf.expand_dims(tf.cast(1 - tokens_to_keep, tf.float32) * constants.VERY_SMALL, 1)
  token_mask_row_normal_scale = tf.expand_dims(tf.cast(tokens_to_keep, tf.float32), 1)
  token_mask_col = tf.expand_dims(tf.cast(tokens_to_keep, tf.float32), 2)

  tf.logging.log(tf.logging.INFO, "mixture model with {} clusters and {} maximum exploring steps".format(num_clusters, (
    max_up_depth, max_down_depth)))

  def rm_sr(mtx):
    return mtx * (1 - tf.eye(tf.shape(heads)[1], batch_shape=[tf.shape(heads)[0]]))

  if len(heads.get_shape()) < 3:
    # max_seq_len = tf.cast(tf.shape(heads)[-1], dtype=tf.float32)

    if extreme_value:
      on_value = constants.VERY_LARGE
      off_value = constants.VERY_SMALL
    else:
      on_value = constants.VERY_LARGE
      off_value = constants.VERY_SMALL

    heads = tf.one_hot(heads, tf.shape(heads)[-1], off_value=off_value, on_value=on_value)
    labels = tf.one_hot(labels, 69, off_value=off_value, on_value=on_value)
    heads = heads + token_mask_row
  else:

    heads = heads + token_mask_row
    labels = tf.stop_gradient(labels)
    heads = tf.stop_gradient(heads)
    if use_gumbel_max:
      labels_dist = tfp.distributions.OneHotCategorical(logits=labels)
      heads_dist = tfp.distributions.OneHotCategorical(logits=heads)
      labels = tf.cast(tfp.distributions.Sample(labels_dist, sample_shape=()).sample(), tf.float32)
      heads = tf.cast(tfp.distributions.Sample(heads_dist, sample_shape=()).sample(), tf.float32)

  if not use_gumbel_max:
    masked_heads = tf.nn.softmax(heads, axis=-1)
    labels = tf.nn.softmax(labels, axis=-1)
  else:
    masked_heads = heads
    labels = labels
  batched_predicate_gather_indices, unbatch_bpgi = batched_predicate_gather_indices
  synt_mask_mtx = tf.ones_like(heads) * token_mask_row_normal_scale * token_mask_col
  batch_size, seq_len = tf.shape(masked_heads)[0], tf.shape(masked_heads)[1]

  # Here, gathered_x has shape of (b, pred, 1, seq)
  masked_heads = masked_heads * token_mask_col
  gathered_masked_head = tf.gather_nd(masked_heads, batched_predicate_gather_indices, batch_dims=1)
  synt_mask_mtx = tf.gather_nd(synt_mask_mtx, batched_predicate_gather_indices, batch_dims=1)
  up_l, down_f = masked_heads, masked_heads

  updown = rm_sr(tf.matmul(rm_sr(up_l), rm_sr(down_f), transpose_b=True))
  gathered_updown = tf.gather_nd(updown, batched_predicate_gather_indices, batch_dims=1)

  no_op = tf.eye(tf.shape(masked_heads)[-1], batch_shape=[tf.shape(masked_heads)[0]])
  no_op *= token_mask_row_normal_scale * token_mask_col
  gathered_no_op = tf.gather_nd(no_op, batched_predicate_gather_indices, batch_dims=1)
  tf.logging.log(tf.logging.INFO, "gathered_tensors {} {} {}".format(gathered_masked_head, gathered_updown,
                                                                     batched_predicate_gather_indices))
  # exit()
  max_pred_count = tf.shape(gathered_masked_head)[1]
  gathered_tokens_to_keep = tf.tile(tf.reshape(tokens_to_keep, [batch_size, 1, seq_len]), [1, max_pred_count, 1])

  gathered_transition_updown = tf.tile(tf.reshape(updown, [batch_size, 1, seq_len, seq_len]), [1, max_pred_count, 1, 1])

  hidden_size = latent_hidden_size
  is_training = mode == ModeKeys.TRAIN
  lstm_cell = tf.keras.layers.LSTMCell(
    hidden_size, activation='tanh', recurrent_activation='sigmoid',
    use_bias=True, kernel_initializer='glorot_uniform',
    recurrent_initializer='orthogonal',
    bias_initializer='zeros', unit_forget_bias=True,
    kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None,
    kernel_constraint=None, recurrent_constraint=None, bias_constraint=None,
    dropout=0.1, recurrent_dropout=0.1)
  if use_fixed_pattern:
    with tf.variable_scope("syn_pattern_to_cluster_fixed_pattern"):
      dense1 = tf.keras.layers.Dense(256, activation='relu')
      dense2 = tf.keras.layers.Dense(num_clusters)

  # of shape (b, pred, seq, seq)
  gathered_adjmtx_up = tf.stop_gradient(
    tf.tile(tf.reshape(masked_heads, [batch_size, 1, seq_len, seq_len]), [1, max_pred_count, 1, 1]))
  gathered_adjmtx_down = tf.stop_gradient(
    tf.tile(tf.reshape(tf.transpose(masked_heads, [0, 2, 1]), [batch_size, 1, seq_len, seq_len]),
            [1, max_pred_count, 1, 1]))

  with tf.variable_scope('wfs_embedding'):
    direction_embedding_mtx = tf.get_variable('dir_embedding', [5, hidden_size],
                                              initializer=tf.keras.initializers.Orthogonal(), trainable=False)
    dep_label_embedding_mtx = tf.get_variable('label_embedding', [69, hidden_size],
                                              initializer=tf.keras.initializers.Orthogonal())
    step_embedding_mtx = tf.get_variable('step_embedding', [len(step_to_feature_idx) + 2, 64])

  up_embedding = []
  down_embedding = []
  zero_input_features = []

  input_size = 0
  MASK_STEP = len(step_to_feature_idx) + 1
  OTHERS_STEP = len(step_to_feature_idx)

  if use_dep_label:
    tf.logging.log(tf.logging.INFO, "using dependency label information")
    dep_label_hiddens = tf.matmul(labels, dep_label_embedding_mtx)
    gathered_dep_label_hiddens = tf.tile(tf.reshape(dep_label_hiddens, [batch_size, 1, seq_len, hidden_size]),
                                         [1, max_pred_count, 1, 1])
    input_size += hidden_size

  if use_direction:
    up_embedding += [
      tf.tile(tf.nn.embedding_lookup(direction_embedding_mtx, tf.reshape(0, [1, 1, 1])),
              [batch_size, max_pred_count, seq_len, 1])]
    down_embedding += [
      tf.tile(tf.nn.embedding_lookup(direction_embedding_mtx, tf.reshape(1, [1, 1, 1])),
              [batch_size, max_pred_count, seq_len, 1])]
    zero_input_features += [
      tf.tile(tf.nn.embedding_lookup(direction_embedding_mtx, tf.reshape(2, [1, 1, 1])),
              [batch_size, max_pred_count, seq_len, 1])]
    input_size += hidden_size
  if hiddens is not None:
    tiled_hidden = tf.reshape(tf.tile(hiddens, [1, max_pred_count, 1]),
                              shape=[batch_size, max_pred_count, seq_len, hidden_size])
    gathered_hidden = tiled_hidden  # tf.gather_nd(tiled_hidden, predicate_gather_indices)
    input_size += hidden_size

  up_embedding = tf.reduce_mean(tf.stack(up_embedding, -1), axis=-1)
  down_embedding = tf.reduce_mean(tf.stack(down_embedding, -1), axis=-1)
  zero_input_features = tf.reduce_mean(tf.stack(zero_input_features, -1), axis=-1)

  def lstm_forward(lstm_state, lstm_state_c, input):
    # hidden to be of (Gat_Pred, Seq, H)
    tf.logging.log(tf.logging.INFO, "three_tensors {} {}".format(lstm_state, input))
    lstm_state_reshaped = tf.reshape(lstm_state, [-1, hidden_size])
    lstm_state_c_reshaped = tf.reshape(lstm_state_c, [-1, hidden_size])
    input_reshaped = tf.reshape(input, [-1, input_size])
    # input_reshaped = tf.Print(input_reshaped, [tf.shape(input_reshaped), tf.shape(lstm_state_reshaped)])
    new_lstm_state, state = lstm_cell(input_reshaped, [lstm_state_reshaped, lstm_state_c_reshaped],
                                      training=is_training)
    return tf.reshape(state[0], tf.shape(lstm_state)), tf.reshape(state[1], tf.shape(lstm_state_c))

  def lstm_forward_adv(prev_activation, current_activation, input_direction, slot, slot_c, down=False, zero_init=False):
    # shape (b, pred, seq, 1)

    prev_activation_mask = tf.expand_dims(prev_activation, axis=-1)
    current_activation_mask = tf.expand_dims(current_activation, axis=-1)
    tf.logging.log(tf.logging.INFO, "transition matrix & mask {} {}".format(gathered_adjmtx_down, prev_activation_mask))
    if down:
      gathered_adjmtx = gathered_adjmtx_down * prev_activation_mask
    else:
      gathered_adjmtx = gathered_adjmtx_up * prev_activation_mask
    input_features = [input_direction]
    if use_dep_label:
      if not down:
        input_features += [tf.transpose(tf.matmul(gathered_dep_label_hiddens, gathered_adjmtx, adjoint_a=True),
                                        input_feature_transpose_list)]
      else:
        input_features += [gathered_dep_label_hiddens]
    if hiddens is not None:
      input_features += [gathered_hidden]
    input_features = tf.concat(input_features, axis=-1)

    if zero_init:
      lstm_state = slot
      lstm_state_c = slot_c
    else:
      lstm_state = tf.matmul(gathered_adjmtx, slot)
      lstm_state_c = tf.matmul(gathered_adjmtx, slot_c)
    new_lstm_state, new_lstm_state_c = lstm_forward(lstm_state, lstm_state_c, input_features)
    tf.logging.log(tf.logging.INFO, "returning state tensor {}, {}".format(new_lstm_state, current_activation_mask))
    slot += new_lstm_state * current_activation_mask  # tf.stop_gradient(tf.expand_dims(current_activation_mask, axis=-1))
    slot_c += new_lstm_state_c * current_activation_mask  # tf.stop_gradient(tf.expand_dims(current_activation_mask, axis=-1))
    return slot, slot_c

  if use_trigger_batch:
    if hiddens is not None:
      gathered_hidden = tf.gather_nd(gathered_hidden, unbatch_bpgi)
    gathered_tokens_to_keep = tf.gather_nd(gathered_tokens_to_keep, unbatch_bpgi)
    gathered_adjmtx_up = tf.gather_nd(gathered_adjmtx_up, unbatch_bpgi)
    gathered_adjmtx_down = tf.gather_nd(gathered_adjmtx_down, unbatch_bpgi)
    gathered_masked_head = tf.gather_nd(gathered_masked_head, unbatch_bpgi)
    gathered_no_op = tf.gather_nd(gathered_no_op, unbatch_bpgi)
    gathered_updown = tf.gather_nd(gathered_updown, unbatch_bpgi)
    gathered_transition_updown = tf.gather_nd(gathered_transition_updown, unbatch_bpgi)
    if use_dep_label:
      gathered_dep_label_hiddens = tf.gather_nd(gathered_dep_label_hiddens, unbatch_bpgi)
    synt_mask_mtx = tf.gather_nd(synt_mask_mtx, unbatch_bpgi)
    up_embedding = tf.gather_nd(up_embedding, unbatch_bpgi)
    down_embedding = tf.gather_nd(down_embedding, unbatch_bpgi)
    zero_input_features = tf.gather_nd(zero_input_features, unbatch_bpgi)
    n_triggers = tf.shape(gathered_tokens_to_keep)[0]
    slot = tf.zeros([n_triggers, seq_len, hidden_size])
    slot_c = tf.zeros([n_triggers, seq_len, hidden_size])
    final_slot_dim = hidden_size
    final_slot = tf.zeros([n_triggers, seq_len, final_slot_dim])
    if use_fixed_pattern:
      ud_slot = tf.zeros([n_triggers, seq_len, hidden_size])
    input_feature_transpose_list = [0, 2, 1]
  else:
    slot = tf.zeros([batch_size, max_pred_count, seq_len, hidden_size])
    slot_c = tf.zeros([batch_size, max_pred_count, seq_len, hidden_size])
    final_slot_dim = hidden_size
    final_slot = tf.zeros([batch_size, max_pred_count, seq_len, final_slot_dim])
    if use_fixed_pattern:
      ud_slot = tf.zeros([batch_size, max_pred_count, seq_len, hidden_size])
    input_feature_transpose_list = [0, 1, 3, 2]

  init_slot = slot
  init_slot_c = slot_c
  slot_zero_init, slot_c_zero_init = lstm_forward_adv(gathered_no_op, gathered_no_op,
                                                      zero_input_features,
                                                      slot, slot_c, down=False,
                                                      zero_init=True)

  slot_up_init, slot_c_up_init = lstm_forward_adv(gathered_no_op, gathered_masked_head, up_embedding, slot_zero_init,
                                                  slot_c_zero_init, down=False)
  slot_updown_init, slot_c_updown_init = lstm_forward_adv(gathered_masked_head, gathered_updown, down_embedding,
                                                          slot_up_init, slot_c_up_init, down=True)

  queue = [(1, (1, 0), gathered_masked_head, 'up', slot_up_init, slot_c_up_init),
           (2, (1, 1), gathered_updown, 'down', slot_updown_init, slot_c_updown_init),
           (0, (0, 0), gathered_no_op, 'down', slot_zero_init, slot_c_zero_init)]
  heapq.heapify(queue)

  # def explore():
  while len(queue) > 0:
    # retrieve current mtx
    total_step, step, mtx, state, slot, slot_c = heapq.heappop(queue)
    tf.logging.log(tf.logging.INFO, "extract following info from queue: {} {} {}".format(step, mtx, slot))

    if step[0] > max_up_depth or step[1] > max_down_depth:
      continue

    if state == 'up':
      plausible_action = ['up', 'updown', 'complete']
    elif state == 'down':
      plausible_action = ['down', 'complete']
    else:
      raise NotImplementedError

    if 'up' in plausible_action:
      # mtx is of shape (b, pred, seq)
      mtx_reshaped = tf.expand_dims(mtx, axis=-2)
      next_step = tf.matmul(mtx_reshaped, gathered_adjmtx_up)
      next_step_squeezed = tf.squeeze(next_step, axis=-2)
      new_slot, new_slot_c = lstm_forward_adv(mtx, next_step_squeezed, up_embedding, slot, slot_c, down=False)
      heapq.heappush(queue, (total_step + 1, (step[0] + 1, step[1]), next_step_squeezed, 'up', new_slot, new_slot_c))

    if 'down' in plausible_action:
      mtx_reshaped = tf.expand_dims(mtx, axis=-2)
      next_step = tf.matmul(mtx_reshaped, gathered_adjmtx_up, transpose_b=True)
      next_step_squeezed = tf.squeeze(next_step, axis=-2)
      new_slot, new_slot_c = lstm_forward_adv(mtx, next_step_squeezed, down_embedding, slot, slot_c, down=True)
      tf.logging.log(tf.logging.INFO, "change slot at down step @ step {}: {}".format(step, new_slot))
      heapq.heappush(queue, (total_step + 1, (step[0], step[1] + 1), next_step_squeezed, 'down', new_slot, new_slot_c))

    if 'updown' in plausible_action:
      mtx_reshaped = tf.expand_dims(mtx, axis=-2)
      next_step_1 = tf.matmul(mtx_reshaped, gathered_adjmtx_up)
      next_step_1_squeezed = tf.squeeze(next_step_1, axis=-2)
      next_step_2 = tf.matmul(mtx_reshaped, gathered_transition_updown)
      next_step_2_squeezed = tf.squeeze(next_step_2, axis=-2)
      tf.logging.log(tf.logging.INFO,
                     "compute activation mtx at updown step @ step {}: {} {} {}".format(step, next_step_1_squeezed,
                                                                                        next_step_2_squeezed,
                                                                                        gathered_transition_updown))
      new_slot_1, new_slot_c_1 = lstm_forward_adv(mtx, next_step_1_squeezed, up_embedding, slot, slot_c, down=False)
      new_slot_2, new_slot_c_2 = lstm_forward_adv(next_step_1_squeezed, next_step_2_squeezed, down_embedding,
                                                  new_slot_1, new_slot_c_1, down=True)
      tf.logging.log(tf.logging.INFO, "change slot at updown step @ step {}: {}".format(step, new_slot_2))
      heapq.heappush(queue, (
      total_step + 2, (step[0] + 1, step[1] + 1), next_step_2_squeezed, 'down', new_slot_2, new_slot_c_2))

    if 'complete' in plausible_action:
      to_be_appended = tf.minimum(mtx, synt_mask_mtx)
      to_be_appended_reshaped = tf.stop_gradient(tf.expand_dims(to_be_appended, axis=-1))
      synt_mask_mtx -= to_be_appended

      if use_fixed_pattern:
        trigger_size = tf.shape(to_be_appended)[0]
        step_hiddens = tf.nn.embedding_lookup(step_embedding_mtx, tf.reshape(step_to_feature_idx[step], [1]))
        step_hiddens_reshaped = tf.reshape(step_hiddens, [1, 1, hidden_size])
        step_hiddens_gathered = tf.tile(step_hiddens_reshaped, [trigger_size, seq_len, 1])
        ud_slot += to_be_appended_reshaped * step_hiddens_gathered

      tf.logging.log(tf.logging.INFO, "search exit, with lstm state of {}".format(slot))
      output_hiddens = [to_be_appended_reshaped * slot]
      final_slot += tf.concat(output_hiddens, axis=-1)

  tf.logging.log(tf.logging.INFO, "non_keep_tokens tensor shape {} {}".format(1, gathered_tokens_to_keep))
  non_keep_tokens = 1 - gathered_tokens_to_keep

  output_hiddens = []
  if use_direction:
    if not use_trigger_batch:
      output_hiddens += [tf.tile(tf.nn.embedding_lookup(direction_embedding_mtx, tf.reshape(4, [1, 1, 1])),
                                 [batch_size, max_pred_count, seq_len, 1])]
    else:
      output_hiddens += [tf.tile(tf.nn.embedding_lookup(direction_embedding_mtx, tf.reshape(4, [1, 1])),
                                 [n_triggers, seq_len, 1])]
  non_keep_tokens_input_features = tf.concat(output_hiddens, axis=-1)
  tf.logging.log(tf.logging.INFO, "non_keep_tokens tensor shape {} {}".format(non_keep_tokens, gathered_tokens_to_keep))
  non_keep_tokens_slot, non_keep_tokens_slot_c = lstm_forward_adv(non_keep_tokens, non_keep_tokens,
                                                                  non_keep_tokens_input_features, init_slot,
                                                                  init_slot_c, down=False, zero_init=True)
  final_slot += non_keep_tokens_slot * tf.expand_dims(non_keep_tokens, axis=-1)

  if use_fixed_pattern:
    trigger_size = tf.shape(non_keep_tokens)[0]
    step_hiddens = tf.nn.embedding_lookup(step_embedding_mtx, tf.reshape(MASK_STEP, [1]))
    step_hiddens_reshaped = tf.reshape(step_hiddens, [1, 1, hidden_size])
    step_hiddens_gathered = tf.tile(step_hiddens_reshaped, [trigger_size, seq_len, 1])
    ud_slot += tf.expand_dims(non_keep_tokens, axis=-1) * step_hiddens_gathered

  '----------------------------------'
  remainders = tf.stop_gradient(synt_mask_mtx)
  output_hiddens = []
  if use_direction:
    if not use_trigger_batch:
      output_hiddens += [tf.tile(tf.nn.embedding_lookup(direction_embedding_mtx, tf.reshape(3, [1, 1, 1])),
                                 [batch_size, max_pred_count, seq_len, 1])]
    else:
      output_hiddens += [tf.tile(tf.nn.embedding_lookup(direction_embedding_mtx, tf.reshape(3, [1, 1])),
                                 [n_triggers, seq_len, 1])]
  others = tf.concat(output_hiddens, axis=-1)
  others_slot, others_slot_c = lstm_forward_adv(remainders, remainders, others, init_slot, init_slot_c, down=False,
                                                zero_init=True)
  final_slot += others_slot * tf.expand_dims(remainders, axis=-1)
  if use_fixed_pattern:
    trigger_size = tf.shape(remainders)[0]
    step_hiddens = tf.nn.embedding_lookup(step_embedding_mtx, tf.reshape(OTHERS_STEP, [1]))
    step_hiddens_reshaped = tf.reshape(step_hiddens, [1, 1, hidden_size])
    step_hiddens_gathered = tf.tile(step_hiddens_reshaped, [trigger_size, seq_len, 1])
    ud_slot += tf.expand_dims(remainders, axis=-1) * step_hiddens_gathered

  if not use_trigger_batch:
    final_slot = tf.gather_nd(final_slot, unbatch_bpgi)
  tf.logging.log(tf.logging.INFO, "final slot shape {}".format(final_slot))

  if use_fixed_pattern:
    z_logit = dense2(dense1(ud_slot))
  else:
    with tf.variable_scope("syn_pattern_to_cluster"):
      with tf.variable_scope('batchnorm_srl_score'):
        batch_norm_1 = tf.keras.layers.BatchNormalization(
          axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
          beta_initializer='zeros', gamma_initializer='ones',
          moving_mean_initializer='zeros',
          moving_variance_initializer='ones', beta_regularizer=None,
          gamma_regularizer=None, beta_constraint=None, gamma_constraint=None
        )
        batch_norm_2 = tf.keras.layers.BatchNormalization(
          axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
          beta_initializer='zeros', gamma_initializer='ones',
          moving_mean_initializer='zeros',
          moving_variance_initializer='ones', beta_regularizer=None,
          gamma_regularizer=None, beta_constraint=None, gamma_constraint=None
        )
        dropout_layer = tf.keras.layers.Dropout(0.1)
      dense1 = tf.keras.layers.Dense(256, activation='relu')
      dense2 = tf.keras.layers.Dense(num_clusters)
      final_slot_cluster = tf.stop_gradient(final_slot)
      z_logit = batch_norm_2(dense2(dropout_layer(batch_norm_1(dense1(dropout_layer(final_slot_cluster))))))
    # with tf.variable_scope("syn_pattern_to_cluster"):
    #   # complete_step = tf.stack(complete_hidden_by_lstm, axis=-2)
    #   dense_2 = tf.keras.layers.Dense(num_clusters, kernel_regularizer='l1' if l1_regularizer else None)
    #
    #   # it has a shape of (st, number_clusters)
    #   final_slot_cluster = tf.stop_gradient(final_slot)
    #   z_logit = dense_2(dropout_layer(final_slot_cluster))
  if mode != ModeKeys.TRAIN:
    z_logit = tf.Print(z_logit, [tf.nn.softmax(z_logit), tf.argmax(z_logit, axis=-1)], "z_logit", summarize=30)

  if inference_mode == 'nll':
    z_prob = z_logit
  elif inference_mode == 'gumbel':
    if mode != ModeKeys.TRAIN:
      temperature = 1e-5
      dist = tfp.distributions.RelaxedOneHotCategorical(temperature, logits=z_logit)
      sample = tfp.distributions.Sample(dist)
      z_prob = sample.sample()
    else:
      current_step = tf.train.get_global_step()
      temperature = tf.maximum(0.5, tf.exp(-1e-5 * tf.cast(current_step, tf.float32)))
      dist = tfp.distributions.RelaxedOneHotCategorical(temperature, logits=z_logit)
      sample = tfp.distributions.Sample(dist)
      z_prob = sample.sample()
    tf.logging.log(tf.logging.INFO, "z_sample:{}".format(z_prob))
  else:
    raise NotImplementedError

  additional_loss = 0.
  # if returns_lstm_state:
  #   z_prob = final_slot

  return z_prob, additional_loss, final_slot


# def get_dep_transition_wfs_dp_normal(
#         parse_gold, parse_labels, mode, predicate_gather_indices, tokens_to_keep=None, extreme_value=False,
#         max_up_depth=4, max_down_depth=3, num_clusters=3,
#         inference_mode='full',
#         return_gathered_hiddens=False, half_return_size=False, l1_regularizer = False,
#         layer_norm = False, use_bai=False, num_samples=1, hiddens = None, latent_hidden_size = 64, use_direction=True):
#   """
#     heads: head-dependent distribution of shape (B, seq_len, seq_len)
#     labels: label distribution for each head-dependent choice, shape: (B, seq_len, labels)
#   """
#   assert hiddens is not None or not use_direction
#   heads = parse_gold
#   labels = parse_labels
#   token_mask_row = tf.expand_dims(tf.cast(1 - tokens_to_keep, tf.float32) * constants.VERY_SMALL, 1)
#   token_mask_row_normal_scale = tf.expand_dims(tf.cast(tokens_to_keep, tf.float32), 1)
#   token_mask_col = tf.expand_dims(tf.cast(tokens_to_keep, tf.float32), 2)
#
#   # token_mask_col = tf.expand_dims(tf.cast(1 - tokens_to_keep, tf.float32) * constants.VERY_SMALL, 2)
#
#   tf.logging.log(tf.logging.INFO, "mixture model with {} clusters and {} maximum exploring steps".format(num_clusters, (
#   max_up_depth, max_down_depth)))
#
#   def rm_sr(mtx):
#     return mtx * (1 - tf.eye(tf.shape(heads)[1], batch_shape=[tf.shape(heads)[0]]))
#
#   if len(heads.get_shape()) < 3:
#     # max_seq_len = tf.cast(tf.shape(heads)[-1], dtype=tf.float32)
#
#     if extreme_value:
#       on_value = constants.VERY_LARGE
#       off_value = constants.VERY_SMALL
#     else:
#       on_value = 10.
#       off_value = -10.
#
#     heads = tf.one_hot(heads, tf.shape(heads)[-1], off_value=off_value, on_value=on_value)
#     heads = heads + token_mask_row
#   else:
#
#     heads = heads + token_mask_row
#     heads = tf.stop_gradient(heads)
#
#   masked_heads = tf.nn.softmax(heads, axis=-1)
#
#   # syn_feature_mtx = tf.zeros_like(heads)
#   synt_mask_mtx = tf.ones_like(heads) * token_mask_row_normal_scale * token_mask_col
#
#   masked_heads = masked_heads * token_mask_col
#
#   up_l, down_f = masked_heads, masked_heads
#
#   updown = rm_sr(tf.matmul(rm_sr(up_l), rm_sr(down_f), transpose_b=True))
#   no_op = tf.eye(tf.shape(masked_heads)[-1], batch_shape=[tf.shape(masked_heads)[0]])
#   no_op *= token_mask_row_normal_scale * token_mask_col
#
#
#
#
#   batch_size, seq_len = tf.shape(masked_heads)[0], tf.shape(masked_heads)[1]
#   hidden_size = latent_hidden_size
#   is_training = mode == ModeKeys.TRAIN
#   lstm_cell = tf.keras.layers.GRUCell(
#     hidden_size, activation='tanh', recurrent_activation='sigmoid',
#     use_bias=True, kernel_initializer='glorot_uniform',
#     recurrent_initializer='orthogonal',
#     bias_initializer='zeros', kernel_regularizer=None,
#     recurrent_regularizer=None, bias_regularizer=None, kernel_constraint=None,
#     recurrent_constraint=None, bias_constraint=None, dropout=0.1,
#     recurrent_dropout=0.1, reset_after=True
#   )
#   # lstm_init_state = lstm_cell.get_initial_state(batch_size=batch_size*seq_len)
#
#   gathered_adjmtx_up = tf.stop_gradient(
#     tf.gather_nd(tf.reshape(tf.tile(masked_heads, [1, seq_len, 1]), [batch_size, seq_len, seq_len, seq_len]),
#                  predicate_gather_indices))
#   gathered_adjmtx_down = tf.stop_gradient(tf.gather_nd(
#     tf.reshape(tf.tile(tf.transpose(masked_heads, [0, 2, 1]), [1, seq_len, 1]),
#                [batch_size, seq_len, seq_len, seq_len]),
#     predicate_gather_indices))
#
#
#   number_triggers = tf.shape(gathered_adjmtx_up)[0]
#
#   with tf.variable_scope('wfs_embedding'):
#     direction_embedding_mtx = tf.get_variable('dir_embedding', [3, hidden_size])
#     dep_label_embedding_mtx = tf.get_variable('label_embedding', [70, hidden_size])
#     others_embedding_mtx = tf.get_variable('others_embedding', [1, hidden_size])
#
#
#
#
#
#   def lstm_forward(lstm_state, input):
#     # hidden to be of (Gat_Pred, Seq, H)
#     reshaped_hidden_size = hidden_size
#     # if use_direction:
#     #   reshaped_hidden_size+=16
#     tf.logging.log(tf.logging.INFO, "three_tensors {} {}".format(lstm_state, input))
#     lstm_state_reshaped = tf.reshape(lstm_state, [-1, reshaped_hidden_size])
#     input_reshaped = tf.reshape(input, [-1, reshaped_hidden_size])
#     new_lstm_state, _ = lstm_cell(input_reshaped, [lstm_state_reshaped], training=is_training)
#     return tf.reshape(new_lstm_state, tf.shape(lstm_state))
#
#   def lstm_forward_adv(adjmtx, current_activation, input_direction, slot, down=False, zero_init=False):
#     gathered_activation = tf.gather_nd(current_activation, predicate_gather_indices)
#     # if down:
#     #   gathered_adjmtx = gathered_adjmtx_down
#     # else:
#     #   gathered_adjmtx = gathered_adjmtx_up
#
#     lstm_state = tf.matmul(gathered_adjmtx_up, slot, adjoint_a=down)
#     if zero_init:
#       lstm_state = slot
#     input_features = input_direction
#     # input_features = tf.concat([input_direction], axis=-1)
#     new_lstm_state = lstm_forward(lstm_state, input_features)
#     slot += new_lstm_state * tf.stop_gradient(tf.expand_dims(gathered_activation, axis=-1))
#     return slot
#
#     pass
#
#   # gathered_predicates = tf.expand_dims(tf.gather_nd(predicate_mlp, predicate_gather_indices), 1)
#   up_embedding = []
#   down_embedding = []
#   zero_input_features = []
#
#   if use_direction:
#     up_embedding += [tf.nn.embedding_lookup(direction_embedding_mtx, tf.tile(tf.reshape(0, [1, 1]), [number_triggers, seq_len]))]
#     down_embedding += [tf.nn.embedding_lookup(direction_embedding_mtx, tf.tile(tf.reshape(1, [1, 1]), [number_triggers, seq_len]))]
#     zero_input_features += [tf.nn.embedding_lookup(direction_embedding_mtx,
#                                                 tf.tile(tf.reshape(2, [1, 1]), [number_triggers, seq_len]))]
#   if hiddens is not None:
#     tiled_hidden = tf.reshape(tf.tile(hiddens, [1, seq_len, 1]), shape=[batch_size, seq_len, seq_len, hidden_size])
#     gathered_hidden = tf.gather_nd(tiled_hidden, predicate_gather_indices)
#
#     up_embedding += [gathered_hidden]
#     down_embedding += [gathered_hidden]
#     zero_input_features += [gathered_hidden]
#
#   up_embedding = tf.reduce_mean(tf.stack(up_embedding, -1), axis=-1)
#   down_embedding = tf.reduce_mean(tf.stack(down_embedding, -1), axis=-1)
#   zero_input_features = tf.reduce_mean(tf.stack(zero_input_features, -1), axis=-1)
#
#
#
#   slot = tf.zeros([number_triggers, seq_len, hidden_size])
#   final_slot_dim = hidden_size
#   final_slot = tf.zeros([number_triggers, seq_len, final_slot_dim])
#   # final_slot = tf.zeros_like(gathered_hidden)
#
#   zero_init_activation = tf.eye(seq_len, batch_shape=[batch_size]) * token_mask_row_normal_scale * token_mask_col
#   zero_init_adjmtx = tf.eye(seq_len, batch_shape=[batch_size]) * token_mask_row_normal_scale * token_mask_col
#
#   # zero_input_features = tf.concat(zero_input_features, axis=-1)
#   slot_zero_init = lstm_forward_adv(zero_init_adjmtx, zero_init_activation,
#                                     zero_input_features,
#                                     slot, down=False,
#                                     zero_init=True)
#
#   slot_up_init = lstm_forward_adv(masked_heads, masked_heads, up_embedding, slot_zero_init, down=False)
#   slot_updown_init = lstm_forward_adv(masked_heads, updown, down_embedding, slot_up_init, down=True)
#
#   # updown_init_output, updown_init_lstm = lstm_forward(up_init_lstm, tf.linalg.matmul(updown, hiddens, adjoint_a=True))
#
#   # queue = [masked_heads]
#   queue = [(1, (1, 0), masked_heads, 'up', slot_zero_init, slot_zero_init),
#            (2, (1, 1), updown, 'down', slot_up_init, slot_up_init),
#            (0, (0, 0), no_op, 'down', slot_updown_init, slot_updown_init)]
#   heapq.heapify(queue)
#
#   # def explore():
#   while len(queue) > 0:
#     # retrieve current mtx
#     total_step, step, mtx, state, _, slot = heapq.heappop(queue)
#
#     if step[0] > max_up_depth or step[1] > max_down_depth:
#       continue
#
#     if state == 'up':
#       plausible_action = ['up', 'updown', 'complete']
#     elif state == 'down':
#       plausible_action = ['down', 'complete']
#     else:
#       raise NotImplementedError
#
#     if 'up' in plausible_action:
#       next_step = tf.matmul(mtx, masked_heads)
#       # next_step_hidden = tf.matmul(next_step, hiddens, adjoint_a=True)
#       # output, next_step_lstm = lstm_forward(lstm_state, next_step_hidden)
#       new_slot = lstm_forward_adv(masked_heads, next_step, up_embedding, slot, down=False)
#       heapq.heappush(queue, (total_step + 1, (step[0] + 1, step[1]), next_step, 'up', new_slot, new_slot))
#
#     if 'down' in plausible_action:
#       next_step = tf.matmul(mtx, masked_heads, transpose_b=True)
#       # next_step_hidden = tf.matmul(next_step, hiddens, adjoint_a=True)
#       new_slot = lstm_forward_adv(masked_heads, next_step, down_embedding, slot, down=True)
#       heapq.heappush(queue, (total_step + 1, (step[0], step[1] + 1), next_step, 'down', new_slot, new_slot))
#
#     if 'updown' in plausible_action:
#       next_step_1 = tf.matmul(mtx, masked_heads)
#       next_step_2 = tf.matmul(mtx, updown)
#       new_slot_1 = lstm_forward_adv(masked_heads, next_step_1, up_embedding, slot, down=False)
#       new_slot_2 = lstm_forward_adv(masked_heads, next_step_2, down_embedding, new_slot_1, down=True)
#       heapq.heappush(queue, (total_step + 2, (step[0] + 1, step[1] + 1), next_step_2, 'down', new_slot_2, new_slot_2))
#
#     if 'complete' in plausible_action:
#       to_be_appended = tf.minimum(mtx, synt_mask_mtx)
#       to_be_appended_reshaped = tf.stop_gradient(tf.expand_dims(tf.gather_nd(to_be_appended, predicate_gather_indices), axis=-1))
#       # complete_list.append(to_be_appended)
#       synt_mask_mtx -= to_be_appended
#       # complete_step.append(tf.one_hot(step_to_feature_idx[step], depth=101))
#       tf.logging.log(tf.logging.INFO, "search exit, with lstm state of {}".format(slot))
#       output_hiddens = [to_be_appended_reshaped * slot]
#       final_slot += tf.concat(output_hiddens, axis=-1)
#       # tf.logging.log(tf.logging.INFO, "complete searching mtx of {}".format(step))
#
#
#
#   remainders = tf.stop_gradient(tf.expand_dims(tf.gather_nd(synt_mask_mtx, predicate_gather_indices), axis=-1))
#   output_hiddens = []
#   if hiddens is not None:
#     output_hiddens += [remainders * gathered_hidden]
#   else:
#     output_hiddens += [remainders * tf.nn.embedding_lookup(others_embedding_mtx, tf.tile(tf.reshape(0, [1, 1]), [number_triggers, seq_len]))]
#   final_slot += tf.reduce_mean(tf.stack(output_hiddens, axis=-1), axis=-1)#tf.concat(output_hiddens, axis=-1)
#
#
#
#
#   # final_slot += tf.stop_gradient(remainders) * tf.nn.embedding_lookup(others_embedding_mtx, tf.tile(tf.reshape(0, [1, 1]), [number_triggers, seq_len]))
#
#   with tf.variable_scope("syn_pattern_to_cluster_normal"):
#     with tf.variable_scope("bias_variance"):
#       # bias_dense1 = tf.keras.layers.Dense(256, activation='relu')
#       composed_bias_var_size = tfp.layers.MultivariateNormalTriL.params_size(hidden_size)
#       bias_var = tf.keras.layers.Dense(composed_bias_var_size)
#     # with tf.variable_scope("variance"):
#     #   var_dense1 = tf.keras.layers.Dense(256, activation='relu')
#     #   var_dense2 = tf.keras.layers.Dense(128)
#     # z_bias = bias_dense2(bias_dense1(final_slot))
#     # z_var = 1e-3 + tf.math.softplus(var_dense2(var_dense1(final_slot)))
#       bias_var_param = bias_var(final_slot)
#
#       z_bias = bias_var_param[:, :, :hidden_size]
#       z_var = tfp.math.fill_triangular(bias_var_param[:, :, hidden_size:])
#     if mode != ModeKeys.TRAIN:
#       bias_var_param = tf.Print(bias_var_param, [z_bias], "z_bias", summarize=30)
#       bias_var_param = tf.Print(bias_var_param, [tf.linalg.diag_part(z_var)], "z_var", summarize=30)
#
#     prior = tfp.distributions.Independent(tfp.distributions.Normal(loc=tf.zeros(hidden_size), scale=1),
#                             reinterpreted_batch_ndims=1)
#     latent_layer = tfp.layers.MultivariateNormalTriL(hidden_size,
#       activity_regularizer=tfp.layers.KLDivergenceRegularizer(prior))
#
#     n_triggers, seq_len = tf.shape(bias_var_param)[0], tf.shape(bias_var_param)[1]
#     # sampled = latent_layer(tf.reshape(z_bias, [-1, hidden_size]), tf.reshape(z_var, [-1, hidden_size, hidden_size]))
#     tf.logging.log(tf.logging.INFO, "distribution size {}".format(composed_bias_var_size))
#     dist = latent_layer(tf.reshape(bias_var_param, [-1, composed_bias_var_size]))
#     sample = tfp.distributions.Sample(dist, sample_shape=(num_samples)).sample()
#     sample_reshaped = tf.reshape(sample, [n_triggers, seq_len, num_samples, hidden_size])
#     sample_switched = tf.stack(tf.unstack(sample_reshaped, axis=2), axis=0)
#     tf.logging.log(tf.logging.INFO, "{} samples {}".format(dist, sample))
#     # exit()
#
#     # dist = tfp.distributions.Normal(z_bias, z_var, validate_args=False, allow_nan_stats=True, name='Normal')
#     # sample = tfp.distributions.Sample(dist, sample_shape=(num_samples))
#     # ref_dist = tfp.distributions.Normal(tf.zeros_like(z_bias), tf.ones_like(z_var), name='Normal_reference')
#     # kl_div = dist.kl_divergence(ref_dist)
#     # z_var = sample.sample()
#
#   return sample_switched, 0.


def get_dep_transition_kup_mtx(parse_gold, tokens_to_keep=None, extreme_value=False, layer_norm_to_heads=False,
                               transpose=False, memory_efficient=False, joint_par_srl_training=False, k=1,
                               parse_labels=None):
  """
    heads: head-dependent distribution of shape (B, seq_len, seq_len)
    labels: label distribution for each head-dependent choice, shape: (B, seq_len, labels)
  """
  print("kup1down:", k)
  assert k > 0
  heads = parse_gold
  labels = parse_labels
  token_mask_row = tf.expand_dims(tf.cast(1 - tokens_to_keep, tf.float32) * constants.VERY_SMALL, 1)
  token_mask_col = tf.expand_dims(tf.cast(tokens_to_keep, tf.float32), 2)
  tf.logging.log(tf.logging.INFO, "Using simple {}up1down direction, removing all prob to ROOT".format(k))

  def prod(l, r):
    # l, r are of (B, S, S)
    l_t = l
    r_t = r
    tmp_t = tf.linalg.matmul(l_t, r_t)
    tmp = tmp_t
    return tmp

  if len(heads.get_shape()) < 3:
    # max_seq_len = tf.cast(tf.shape(heads)[-1], dtype=tf.float32)

    if extreme_value:
      on_value = constants.VERY_LARGE
      off_value = constants.VERY_SMALL
    else:
      on_value = 10.
      off_value = -10.

    heads = tf.one_hot(heads, tf.shape(heads)[-1], off_value=off_value, on_value=on_value)
    if labels is not None:
      labels = tf.one_hot(labels, 69, off_value=off_value, on_value=on_value)
    heads = heads + token_mask_row
  else:

    heads = heads + token_mask_row
    if True:
      heads = tf.stop_gradient(heads)
      if labels is not None:
        labels = tf.stop_gradient(labels)

  masked_heads = tf.nn.softmax(heads, axis=-1)
  # including k steps up and 1 step down
  masked_heads_list = []
  for label_per_step in range(k):
    with tf.variable_scope("{}th_label_transformation".format(label_per_step)):
      if labels is not None:
        with tf.variable_scope("dep_transition_mtx_labels"):
          dense = tf.keras.layers.Dense(
            1, activation=tf.nn.sigmoid, use_bias=True,
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros'
          )
          # convert to (B, seq, 1) tensor -> open_close gate of each dependency arc
          output = dense(labels)
          # gating dependency graph with dependency type
          masked_heads_list.append(masked_heads * output)
      else:
        masked_heads_list.append(masked_heads)
  # Applying column-wise masking
  masked_heads_list = [item * token_mask_col for item in masked_heads_list]
  # Prevent back-looping
  # updown = prod(up, tf.transpose(down, perm=[0, 2, 1])) * (1 - tf.eye(tf.shape(heads)[1], batch_shape=[tf.shape(heads)[0]]))
  tmp = masked_heads_list[-1]
  for idx in reversed(range(k - 1)):
    tmp = prod(masked_heads_list[idx], tmp)
  if transpose:
    tmp = tf.transpose(tmp, perm=[0, 2, 1])

  return tmp


def get_dep_transition_kup_mtx_collect_dep_path(parse_gold, hiddens_l, hiddens_r, tokens_to_keep=None,
                                                extreme_value=False, layer_norm_to_heads=False, transpose=False,
                                                memory_efficient=False, joint_par_srl_training=False, k=1,
                                                parse_labels=None):
  """
    heads: head-dependent distribution of shape (B, seq_len, seq_len)
    labels: label distribution for each head-dependent choice, shape: (B, seq_len, labels)
  """
  print("kup1down:", k)
  assert k > 0
  heads = parse_gold
  labels = parse_labels
  token_mask_row = tf.expand_dims(tf.cast(1 - tokens_to_keep, tf.float32) * constants.VERY_SMALL, 1)
  token_mask_col = tf.expand_dims(tf.cast(tokens_to_keep, tf.float32), 2)
  # token_mask_col = tf.expand_dims(tf.cast(1 - tokens_to_keep, tf.float32) * constants.VERY_SMALL, 2)

  tf.logging.log(tf.logging.INFO,
                 "Using simple {} direction, removing all prob to ROOT".format("up" if not transpose else "down", k))

  def prod(l, r):
    # l, r are of (B, S, S)
    l_t = l
    r_t = r
    tmp_t = tf.linalg.matmul(l_t, r_t)
    tmp = tmp_t
    return tmp

  if len(heads.get_shape()) < 3:
    # max_seq_len = tf.cast(tf.shape(heads)[-1], dtype=tf.float32)

    if extreme_value:
      on_value = constants.VERY_LARGE
      off_value = constants.VERY_SMALL
    else:
      on_value = 10.
      off_value = -10.

    heads = tf.one_hot(heads, tf.shape(heads)[-1], off_value=off_value, on_value=on_value)
    if labels is not None:
      labels = tf.one_hot(labels, 69, off_value=off_value, on_value=on_value)
    heads = heads + token_mask_row
  else:

    heads = heads + token_mask_row
    if True:
      heads = tf.stop_gradient(heads)
      if labels is not None:
        labels = tf.stop_gradient(labels)

  masked_heads = tf.nn.softmax(heads, axis=-1)
  # including k steps up and 1 step down
  masked_heads_list = []
  for label_per_step in range(k):
    masked_heads_list.append(masked_heads)
  # Applying column-wise masking
  masked_heads_list = [item * token_mask_col for item in masked_heads_list]
  # Prevent back-looping
  # updown = prod(up, tf.transpose(down, perm=[0, 2, 1])) * (1 - tf.eye(tf.shape(heads)[1], batch_shape=[tf.shape(heads)[0]]))
  tmp = masked_heads_list[0] if not transpose else tf.transpose(masked_heads_list[0], perm=[0, 2, 1])
  hidden_r = []
  hidden_l = []
  hidden_r.append(tf.linalg.matmul(tmp, hiddens_r))
  hidden_l.append(hiddens_l)
  for idx in range(1, k):
    hidden_l.append(tf.linalg.matmul(tmp, hiddens_l))
    tmp = prod(masked_heads_list[idx], tmp if not transpose else tf.transpose(tmp, perm=[0, 2, 1]))
    hidden_r.append(tf.linalg.matmul(tmp, hiddens_r))

  return tf.expand_dims(tmp, axis=-1), tf.concat(hidden_l, axis=-1), tf.concat(hidden_r, axis=-1)


def get_dep_transition_kup_mtx_collect_ste(parse_gold, hiddens, tokens_to_keep=None, extreme_value=False,
                                           layer_norm_to_heads=False, transpose=False, memory_efficient=False,
                                           joint_par_srl_training=False, k=1, parse_labels=None, use_lstm=False,
                                           return_last=False):
  """
    heads: head-dependent distribution of shape (B, seq_len, seq_len)
    labels: label distribution for each head-dependent choice, shape: (B, seq_len, labels)
  """
  print("kup1down:", k)
  assert k > 0
  heads = parse_gold
  labels = parse_labels
  token_mask_row = tf.expand_dims(tf.cast(1 - tokens_to_keep, tf.float32) * constants.VERY_SMALL, 1)
  token_mask_col = tf.expand_dims(tf.cast(tokens_to_keep, tf.float32), 2)
  # token_mask_col = tf.expand_dims(tf.cast(1 - tokens_to_keep, tf.float32) * constants.VERY_SMALL, 2)

  tf.logging.log(tf.logging.INFO,
                 "Using simple {} direction, removing all prob to ROOT".format("up" if not transpose else "down", k))

  def prod(l, r):
    # l, r are of (B, S, S)
    l_t = l
    r_t = r
    tmp_t = tf.linalg.matmul(l_t, r_t)
    tmp = tmp_t
    return tmp

  if len(heads.get_shape()) < 3:
    # max_seq_len = tf.cast(tf.shape(heads)[-1], dtype=tf.float32)

    if extreme_value:
      on_value = constants.VERY_LARGE
      off_value = constants.VERY_SMALL
    else:
      on_value = 10.
      off_value = -10.

    heads = tf.one_hot(heads, tf.shape(heads)[-1], off_value=off_value, on_value=on_value)
    if labels is not None:
      labels = tf.one_hot(labels, 69, off_value=off_value, on_value=on_value)
    heads = heads + token_mask_row
  else:

    heads = heads + token_mask_row
    if True:
      heads = tf.stop_gradient(heads)
      if labels is not None:
        labels = tf.stop_gradient(labels)

  masked_heads = tf.nn.softmax(heads, axis=-1)
  # including k steps up and 1 step down
  masked_heads_list = []
  for label_per_step in range(k):
    masked_heads_list.append(masked_heads)
  # Applying column-wise masking
  masked_heads_list = [item * token_mask_col for item in masked_heads_list]
  # Prevent back-looping
  # updown = prod(up, tf.transpose(down, perm=[0, 2, 1])) * (1 - tf.eye(tf.shape(heads)[1], batch_shape=[tf.shape(heads)[0]]))
  tmp = masked_heads_list[0] if not transpose else tf.transpose(masked_heads_list[0], perm=[0, 2, 1])
  hidden_list = []
  # hidden_list.append(hiddens)
  hidden_list.append(hiddens)

  for idx in range(1, k):
    # hidden_l.append(tf.linalg.matmul(tmp, hiddens_l))

    r_prod = masked_heads_list[idx] if not transpose else tf.transpose(masked_heads_list[idx], perm=[0, 2, 1])
    hidden_list = [tf.linalg.matmul(tf.transpose(r_prod, perm=[0, 2, 1]), item) for item in hidden_list]
    hidden_list.append(hiddens)
    tmp = prod(tmp, r_prod)
    # hidden_list.append()
  if not use_lstm:
    if return_last:
      output_r = hidden_list[-1]
    else:
      output_r = tf.reduce_mean(tf.stack(hidden_list, axis=-1), axis=-1)
  else:
    with tf.variable_scope("bilstm"):
      _, _, hidden_size = hiddens.get_shape().as_list()
      # batch_size, outer_seq_len, _ =
      lstm = tf.keras.layers.LSTM(int(hidden_size / 2), dropout=0.1)
      bidirectional = tf.keras.layers.Bidirectional(lstm, merge_mode='concat')
      output_seq = tf.reshape(tf.stack(hidden_list, axis=-2), shape=[-1, k, hidden_size])
      output_state = bidirectional(output_seq)
      output_r = tf.reshape(output_state, shape=[tf.shape(hiddens)[0], tf.shape(hiddens)[1], hidden_size])

  return tf.expand_dims(tmp, axis=-1), output_r


def get_dep_transition_kup_mtx_preid_patterns(parse_gold, tokens_to_keep=None, extreme_value=False,
                                              layer_norm_to_heads=False, transpose=False, memory_efficient=False,
                                              joint_par_srl_training=False, k=1, parse_labels=None, conf_fn=None):
  """
    heads: head-dependent distribution of shape (B, seq_len, seq_len)
    labels: label distribution for each head-dependent choice, shape: (B, seq_len, labels)
  """
  print("kup1down:", k)
  assert k > 0
  assert conf_fn is not None
  heads = parse_gold
  labels = parse_labels
  token_mask_row = tf.expand_dims(tf.cast(1 - tokens_to_keep, tf.float32) * constants.VERY_SMALL, 1)
  token_mask_col = tf.expand_dims(tf.cast(tokens_to_keep, tf.float32), 2)
  # token_mask_col = tf.expand_dims(tf.cast(1 - tokens_to_keep, tf.float32) * constants.VERY_SMALL, 2)

  tf.logging.log(tf.logging.INFO, "Using simple {}up1down direction, removing all prob to ROOT".format(k))

  def rm_sr(mtx):
    return mtx * (1 - tf.eye(tf.shape(heads)[1], batch_shape=[tf.shape(heads)[0]]))

  def prod(l, r):
    # l, r are of (B, S, S)
    l_t = l
    r_t = r
    tmp_t = tf.linalg.matmul(l_t, r_t)
    tmp = tmp_t
    return tmp

  if len(heads.get_shape()) < 3:
    # max_seq_len = tf.cast(tf.shape(heads)[-1], dtype=tf.float32)

    if extreme_value:
      on_value = constants.VERY_LARGE
      off_value = constants.VERY_SMALL
    else:
      on_value = 10.
      off_value = -10.

    heads = tf.one_hot(heads, tf.shape(heads)[-1], off_value=off_value, on_value=on_value)
    if labels is not None:
      labels = tf.one_hot(labels, 69, off_value=off_value, on_value=on_value)
    heads = heads + token_mask_row
  else:

    heads = heads + token_mask_row
    if True:
      heads = tf.stop_gradient(heads)
      if labels is not None:
        labels = tf.stop_gradient(labels)

  masked_heads = tf.nn.softmax(heads, axis=-1)
  # including k steps up and 1 step down
  weight = train_utils.load_dep_pattern(conf_fn)
  masked_heads_list = []
  tf.logging.log(tf.logging.INFO, "loaded pre_identified weight {}".format(weight))
  for label_per_step in range(k):
    with tf.variable_scope("{}th_label_transformation".format(label_per_step)):
      if labels is not None:
        with tf.variable_scope("dep_transition_mtx_labels"):
          dense = tf.keras.layers.Dense(
            1, activation=tf.nn.sigmoid, use_bias=False, input_shape=(69,),
            kernel_initializer=tf.constant_initializer(np.expand_dims(weight[label_per_step], axis=-1))  # ,
            # bias_initializer='zeros'
          )
          # convert to (B, seq, 1) tensor -> open_close gate of each dependency arc
          output = dense(tf.nn.softmax(labels, axis=-1))
          # gating dependency graph with dependency type
          masked_heads_list.append(masked_heads * output)
      # else:
      #   masked_heads_list.append(masked_heads)
  # Applying column-wise masking
  masked_heads_list = [item * token_mask_col for item in masked_heads_list]
  # Prevent back-looping
  # updown = prod(up, tf.transpose(down, perm=[0, 2, 1])) * (1 - tf.eye(tf.shape(heads)[1], batch_shape=[tf.shape(heads)[0]]))
  tmp = masked_heads_list[-1]
  for idx in reversed(range(k - 1)):
    tmp = prod(masked_heads_list[idx], tmp)
  if transpose:
    tmp = tf.transpose(tmp, perm=[0, 2, 1])

  return tmp


def srl_bilinear_dep_prior_bilinear(mode, hparams, model_config, inputs, targets, num_labels, tokens_to_keep,
                                    predicate_preds_train,
                                    predicate_preds_eval, predicate_targets, parse_label_predictions,
                                    parse_label_targets, parse_head_predictions, parse_head_targets,
                                    pos_targets, pos_predictions, transition_params, pos_mlp=None):
  '''

  :param input: Tensor with dims: [batch_size, batch_seq_len, hidden_size]
  :param predicate_preds: Tensor of predictions from predicates layer with dims: [batch_size, batch_seq_len]
  :param targets: Tensor of SRL labels with dims: [batch_size, batch_seq_len, batch_num_predicates]
  :param tokens_to_keep:
  :param predictions:
  :param transition_params: [num_labels x num_labels] transition parameters, if doing Viterbi decoding
  :return:
  '''
  # This function includes priors inferred from dependency graph

  if mode == ModeKeys.TRAIN:
    if not hparams.train_with_gold_first:
      parse_gold = parse_head_targets
      if hparams.stop_parse_gradient:
        parse_gold = tf.stop_gradient(parse_gold)
      parse_label = parse_label_targets
    else:
      on_value = 10.
      off_value = -10.

      parse_head_targets = tf.one_hot(parse_head_targets, tf.shape(parse_head_targets)[-1], off_value=off_value,
                                      on_value=on_value)
      parse_label_targets = tf.one_hot(parse_label_targets, 69, off_value=off_value, on_value=on_value)
      parse_gold = tf.cond(tf.train.get_global_step() < 16000, lambda: parse_head_targets,
                           lambda: parse_head_predictions)
      parse_label = tf.cond(tf.train.get_global_step() < 16000, lambda: parse_label_targets,
                            lambda: parse_label_predictions)
    # pos_tag = pos_targets
  else:
    parse_gold = parse_head_predictions
    parse_label = parse_label_predictions
    pos = pos_predictions
    # pos_tag = pos_predictions

  with tf.name_scope('srl_bilinear'):

    def bool_mask_where_predicates(predicates_tensor):
      return tf.logical_and(tf.not_equal(predicates_tensor, predicate_outside_idx), tf.cast(tokens_to_keep, tf.bool))

    input_shape = tf.shape(inputs)
    batch_size = input_shape[0]
    batch_seq_len = input_shape[1]
    num_samples = hparams.num_samples

    predicate_mlp_size = model_config['predicate_mlp_size'] if not hparams.xl_scorer else 2 * model_config[
      'predicate_mlp_size']
    role_mlp_size = model_config['role_mlp_size'] if not hparams.xl_scorer else 2 * model_config['role_mlp_size']
    if hparams.concat_pos_mlp:
      pos_mlp_size = model_config['predicate_pred_mlp_size']
      # predicate_mlp_size+=model_config['predicate_pred_mlp_size']
      # role_mlp_size += model_config['predicate_pred_mlp_size']

    # if hparams.share_pred_role_mlp:
    #   with tf.variable_scope('SHARE_MLP'):
    #     predicate_role_mlp = nn_utils.MLP(inputs, predicate_mlp_size + role_mlp_size, keep_prob=hparams.mlp_dropout)
    #     shared_predicate_mlp, shared_role_mlp = predicate_role_mlp[:, :, :predicate_mlp_size], \
    #                               predicate_role_mlp[:, :, predicate_mlp_size:]
    def bilinear_scoring(allow_concat_pos=False, external_predicate=None, external_role=None, external_role_size=-1,
                         external_predicate_size=-1, t4=False, cluster_emb=None, sample_multiplier=1, concat_pred=False,
                         cluster_emb_only=False, use_distance=False):
      role_mlp = external_role
      predicate_mlp = external_predicate
      role_mlp_size = external_role_size
      # predicate_mlp = tf.Print(predicate_mlp, [predicate_mlp, role_mlp], "input tensors to blinear classifier")
      # print(external_predicate, predicate_mlp)

      # (2) feed through bilinear to obtain scores
      with tf.variable_scope('Bilinear'):
        # gather just the predicates
        # gathered_predicates: num_predicates_in_batch x 1 x predicate_mlp_size
        # role mlp: batch x seq_len x role_mlp_size
        # gathered roles: need a (batch_seq_len x role_mlp_size) role representation for each predicate,
        # i.e. a (num_predicates_in_batch x batch_seq_len x role_mlp_size) tensor
        gathered_predicates = tf.expand_dims(tf.gather_nd(predicate_mlp, predicate_gather_indices), 1)
        if not t4:
          tiled_roles = tf.reshape(tf.tile(role_mlp, [1, batch_seq_len, 1]),
                                   [batch_size, batch_seq_len, batch_seq_len, role_mlp_size])
          gathered_roles = tf.gather_nd(tiled_roles, predicate_gather_indices)
        else:
          gathered_roles = role_mlp

        if sample_multiplier > 1:
          gathered_predicates = tf.tile(gathered_predicates, [sample_multiplier, 1, 1])
          gathered_roles = tf.tile(gathered_roles, [sample_multiplier, 1, 1])

        if cluster_emb is not None and not use_distance:
          if concat_pred:
            gathered_predicates = tf.concat([gathered_predicates, cluster_emb[0]], axis=-1)
          gathered_roles = tf.concat([gathered_roles, cluster_emb[1]], axis=-1)
          if cluster_emb_only:
            gathered_roles = cluster_emb[1]

        # now multiply them together to get (num_predicates_in_batch x batch_seq_len x num_srl_classes) tensor of scores
        # gathered_roles = tf.Print(gathered_roles, [gathered_roles, gathered_predicates], "gathered input tensors to blinear classifier")

        if use_distance:
          gathered_predicates = gathered_predicates - gathered_roles
          gathered_roles = gathered_predicates - gathered_roles
        srl_logits = nn_utils.bilinear_classifier_nary(gathered_predicates, gathered_roles, num_labels,
                                                       hparams.bilinear_dropout)
        # srl_logits = tf.Print(srl_logits, [srl_logits], "srl_logits")
        srl_logits_transposed = tf.transpose(srl_logits, [0, 2, 1])
        # if hparams.srl_layernorm:
        #   with tf.variable_scope('LayerNorm'):
        #     srl_logits_transposed = tf.contrib.layers.layer_norm(srl_logits_transposed)
        return srl_logits_transposed

    # TODO this should really be passed in, not assumed...
    predicate_outside_idx = 0

    predicate_preds = predicate_preds_train if mode == tf.estimator.ModeKeys.TRAIN else predicate_preds_eval
    predicate_gather_indices = tf.where(bool_mask_where_predicates(predicate_preds))
    # predicate_gather_indices = tf.Print(predicate_gather_indices, [tf.shape(predicate_gather_indices)], "predicate_gather_indices shape")
    max_pred_count = tf.reduce_max(tf.reduce_sum(predicate_preds, axis=-1))

    def bool_mask_where_predicates_tmp(predicates_tensor, tokens_to_keep):
      return tf.logical_and(tf.not_equal(predicates_tensor, predicate_outside_idx), tf.cast(tokens_to_keep, tf.bool))

    batched_predicate_gather_indices, lengths = tf.map_fn(lambda x: (
    tf.pad(tf.where(bool_mask_where_predicates_tmp(x[0], x[1])), [[0, max_pred_count - tf.reduce_sum(x[0])], [0, 0]]),
    tf.shape(tf.where(bool_mask_where_predicates_tmp(x[0], x[1])))[0])
                                                          , (predicate_preds, tokens_to_keep),
                                                          dtype=(tf.int64, tf.int32))
    # lengths = tf.Print(lengths, [batched_predicate_gather_indices], "batched_predicate_gather_indices", summarize= 60)
    # lengths = tf.Print(lengths, [predicate_gather_indices, lengths], "predicate_gather_indices", summarize=60)
    unbatch_bpgi = tf.where(
      tf.greater(tf.expand_dims(lengths, -1), tf.expand_dims(tf.range(tf.shape(tokens_to_keep)[1]), 0)))
    # unbatch_bpgi = tf.Print(unbatch_bpgi, [unbatch_bpgi, tf.shape(unbatch_bpgi)], "unbatch_bpgi", summarize=60)
    # unbatch_bpgi = tf.where(tf.not_equal(batched_predicate_gather_indices, -1))
    tf.logging.log(tf.logging.INFO,
                   "batched_pred_gather stuff {} {}".format(batched_predicate_gather_indices, unbatch_bpgi))

    # batched_predicate_gather_indices = batched_predicate_gather_indices.to_tensor(shape=[None, max_pred_count])

    # (1) project into predicate, role representations

    # Here gather (num_predicates_in_batch, seq_len, num_classes) dep prior
    pri_list = []

    additional_loss = 0.

    # need to repeat each of these once for each target in the sentence
    mask_tiled = tf.reshape(tf.tile(tokens_to_keep, [1, batch_seq_len]), [batch_size, batch_seq_len, batch_seq_len])
    mask = tf.gather_nd(mask_tiled, predicate_gather_indices)
    # if hparams.apply_custom_loss_weight:
    #   mask *= loss_weight_mtx

    if not hparams.mixture_model > 0:
      with tf.variable_scope("normal_bilinear"):
        with tf.variable_scope('MLP'):
          predicate_role_mlp = nn_utils.MLP(inputs, predicate_mlp_size + role_mlp_size,
                                            keep_prob=hparams.mlp_dropout)
          predicate_mlp, role_mlp = predicate_role_mlp[:, :, :predicate_mlp_size], \
                                    predicate_role_mlp[:, :, predicate_mlp_size:]
        srl_logits_transposed_normal = bilinear_scoring(external_role=role_mlp,
                                                        external_predicate=predicate_mlp,
                                                        external_role_size=role_mlp_size,
                                                        external_predicate_size=predicate_mlp_size)
        taken_mask = tf.ones_like(srl_logits_transposed_normal)
      # loss_weight_mtx = tf.zeros(tf.shape(srl_logits_transposed_normal)[:2], dtype=tf.float32)

    if hparams.threeupkdown_up_to > 0:
      for y in range(hparams.threeupkdown_up_to):
        tf.logging.log(tf.logging.INFO,
                       "making for 6up{}down, with maximum down depth of {}".format(y, hparams.sixupkdown_up_to))
        with tf.variable_scope("Intermediate_MLP_for_down_{}".format(y)):
          for x in range(4):
            predicate_role_mlp = nn_utils.MLP(inputs, predicate_mlp_size + role_mlp_size,
                                              keep_prob=hparams.mlp_dropout)
            predicate_mlp, role_mlp = predicate_role_mlp[:, :, :predicate_mlp_size], \
                                      predicate_role_mlp[:, :, predicate_mlp_size:]
            mh_mask, hiddens = get_dep_transition_xupydown_mtx_collect_ste(parse_gold,
                                                                           role_mlp if hparams.share_pred_role_mlp else role_mlp,
                                                                           transpose=False,
                                                                           tokens_to_keep=tokens_to_keep, x=x, y=y,
                                                                           parse_labels=None, use_lstm=hparams.use_lstm,
                                                                           return_last=hparams.return_last,
                                                                           use_new_version=hparams.new_updown_search)
            if not hparams.share_scorer:
              if hparams.share_pred_role_mlp:
                raise NotImplementedError
                srl_logits_transposed_mh = bilinear_scoring(external_role=hiddens,
                                                            external_predicate=shared_predicate_mlp,
                                                            external_role_size=role_mlp_size,
                                                            external_predicate_size=predicate_mlp_size)
              else:
                srl_logits_transposed_mh = bilinear_scoring(external_role=hiddens, external_predicate=predicate_mlp,
                                                            external_role_size=role_mlp_size,
                                                            external_predicate_size=predicate_mlp_size)
            else:
              raise NotImplementedError

              # telling how many up-steps the model should undergo

            predicate_mask = tf.gather_nd(mh_mask, predicate_gather_indices)
            srl_logits_transposed_mh *= predicate_mask * tf.nn.relu(taken_mask)
            # loss_weight_mtx += (predicate_mask * 2 * tf.nn.relu(taken_mask))[:, :, 0]
            taken_mask -= predicate_mask * tf.nn.relu(taken_mask)

            pri_list.append(srl_logits_transposed_mh)
    if hparams.one_down:
      for y in [1]:
        tf.logging.log(tf.logging.INFO,
                       "making for 6up{}down, with maximum down depth of {}".format(y, hparams.sixupkdown_up_to))
        with tf.variable_scope("Intermediate_MLP_for_down_{}".format(y)):
          for x in [0]:
            predicate_role_mlp = nn_utils.MLP(inputs, predicate_mlp_size + role_mlp_size,
                                              keep_prob=hparams.mlp_dropout)
            predicate_mlp, role_mlp = predicate_role_mlp[:, :, :predicate_mlp_size], \
                                      predicate_role_mlp[:, :, predicate_mlp_size:]
            mh_mask, hiddens = get_dep_transition_xupydown_mtx_collect_ste(parse_gold,
                                                                           role_mlp if hparams.share_pred_role_mlp else role_mlp,
                                                                           transpose=False,
                                                                           tokens_to_keep=tokens_to_keep, x=x, y=y,
                                                                           parse_labels=None, use_lstm=hparams.use_lstm,
                                                                           return_last=hparams.return_last,
                                                                           use_new_version=hparams.new_updown_search)
            if not hparams.share_scorer:
              if hparams.share_pred_role_mlp:
                raise NotImplementedError
              else:
                srl_logits_transposed_mh = bilinear_scoring(external_role=hiddens, external_predicate=predicate_mlp,
                                                            external_role_size=role_mlp_size,
                                                            external_predicate_size=predicate_mlp_size)
            else:
              raise NotImplementedError

              # telling how many up-steps the model should undergo

            predicate_mask = tf.gather_nd(mh_mask, predicate_gather_indices)
            srl_logits_transposed_mh *= predicate_mask * tf.nn.relu(taken_mask)
            # loss_weight_mtx += (predicate_mask * 2 * tf.nn.relu(taken_mask))[:, :, 0]
            taken_mask -= predicate_mask * tf.nn.relu(taken_mask)

            pri_list.append(srl_logits_transposed_mh)
    # group by up
    if hparams.kuptwodown_up_to > 0:
      for x in range(hparams.kuptwodown_up_to):
        tf.logging.log(tf.logging.INFO,
                       "making for {}up2down, with maximum down depth of {}".format(x, hparams.kuptwodown_up_to))
        with tf.variable_scope("Intermediate_MLP_for_down_{}".format(x)):
          for y in range(3):
            predicate_role_mlp = nn_utils.MLP(inputs, predicate_mlp_size + role_mlp_size,
                                              keep_prob=hparams.mlp_dropout)
            predicate_mlp, role_mlp = predicate_role_mlp[:, :, :predicate_mlp_size], \
                                      predicate_role_mlp[:, :, predicate_mlp_size:]
            mh_mask, hiddens = get_dep_transition_xupydown_mtx_collect_ste(parse_gold,
                                                                           shared_role_mlp if hparams.share_pred_role_mlp else role_mlp,
                                                                           transpose=False,
                                                                           tokens_to_keep=tokens_to_keep, x=x, y=y,
                                                                           parse_labels=None, use_lstm=hparams.use_lstm,
                                                                           return_last=hparams.return_last,
                                                                           use_new_version=hparams.new_updown_search)
            if not hparams.share_scorer:
              if hparams.share_pred_role_mlp:
                srl_logits_transposed_mh = bilinear_scoring(external_role=hiddens,
                                                            external_predicate=shared_predicate_mlp,
                                                            external_role_size=role_mlp_size,
                                                            external_predicate_size=predicate_mlp_size)
              else:
                srl_logits_transposed_mh = bilinear_scoring(external_role=hiddens, external_predicate=predicate_mlp,
                                                            external_role_size=role_mlp_size,
                                                            external_predicate_size=predicate_mlp_size)
            else:
              raise NotImplementedError

              # telling how many up-steps the model should undergo

            predicate_mask = tf.gather_nd(mh_mask, predicate_gather_indices)
            srl_logits_transposed_mh *= predicate_mask * tf.nn.relu(taken_mask)
            # loss_weight_mtx += (predicate_mask * 2 * tf.nn.relu(taken_mask))[:, :, 0]
            taken_mask -= predicate_mask * tf.nn.relu(taken_mask)

            pri_list.append(srl_logits_transposed_mh)

    if hparams.mixture_model > 0:
      num_clusters = hparams.mixture_model
      # if not hparams.use_lstm:
      with tf.variable_scope("shared_pred_role_mlp_latent"):
        injected_hidden = nn_utils.MLP(inputs, hparams.latent_hidden_size,
                                       keep_prob=hparams.mlp_dropout)
      with tf.variable_scope("shared_pred_role_mlp"):
        predicate_role_mlp = nn_utils.MLP(inputs, predicate_mlp_size + role_mlp_size,
                                          keep_prob=hparams.mlp_dropout)
        predicate_mlp, role_mlp = predicate_role_mlp[:, :, :predicate_mlp_size], \
                                  predicate_role_mlp[:, :, predicate_mlp_size:]
      if not hparams.use_lstm:
        mm_mask, additional_loss = get_dep_transition_wfs(parse_gold,
                                                          tokens_to_keep=tokens_to_keep,
                                                          parse_labels=parse_label if hparams.latent_use_dep_label else None,
                                                          use_lstm=hparams.use_lstm,
                                                          return_last=hparams.return_last,
                                                          mode=mode, num_clusters=num_clusters,
                                                          inference_mode=hparams.mixture_model_inference_mode,
                                                          gumbel_temperature=hparams.gumbel_temperature,
                                                          cluster_prob=hparams.cluster_prob,
                                                          propagation_depth=hparams.downward_propagation_depth,
                                                          pred_mlp=None,
                                                          role_mlp=None,
                                                          use_pred_role_feature=hparams.use_pred_role_feature,
                                                          predicate_gather_indices=predicate_gather_indices,
                                                          legacy_mode=hparams.legacy_mode,
                                                          layer_norm=hparams.wfs_layer_norm,
                                                          before_after_indicator=hparams.before_after_indicator,
                                                          max_up_depth=hparams.lstm_search_up_depth,
                                                          max_down_depth=hparams.lstm_search_down_depth,
                                                          use_gumbel_max=hparams.use_gumbel_max_on_input,
                                                          use_dep_label=hparams.latent_use_dep_label,
                                                          parse_label_count=hparams.parse_label_count,
                                                          ignore_masks=hparams.ignore_mask,
                                                          relu6=hparams.correct_vi_objective)
      else:
        mm_mask, additional_loss, final_slot = get_dep_transition_wfs_dp(parse_gold,
                                                                         tokens_to_keep=tokens_to_keep,
                                                                         parse_labels=parse_label,
                                                                         mode=mode, num_clusters=num_clusters,
                                                                         inference_mode=hparams.mixture_model_inference_mode,
                                                                         predicate_gather_indices=predicate_gather_indices,
                                                                         batched_predicate_gather_indices=(
                                                                         batched_predicate_gather_indices,
                                                                         unbatch_bpgi),
                                                                         return_gathered_hiddens=not hparams.use_lstm_hiddens,
                                                                         half_return_size=hparams.half_return_size,
                                                                         l1_regularizer=hparams.l1_dense_regularizer,
                                                                         layer_norm=hparams.wfs_layer_norm,
                                                                         use_bai=hparams.before_after_indicator,
                                                                         num_samples=num_samples,
                                                                         hiddens=injected_hidden if hparams.use_lexicons else None,
                                                                         latent_hidden_size=hparams.latent_hidden_size if not hparams.share_pred_role_mlp else role_mlp_size,
                                                                         use_direction=hparams.latent_use_direction,
                                                                         use_lr_dir=hparams.latent_use_lr_direction,
                                                                         use_dep_label=hparams.latent_use_dep_label,
                                                                         max_up_depth=hparams.lstm_search_up_depth,
                                                                         max_down_depth=hparams.lstm_search_down_depth,
                                                                         use_gumbel_max=hparams.use_gumbel_max_on_input,
                                                                         use_fixed_pattern=hparams.use_fixed_pattern,
                                                                         returns_lstm_state=True,
                                                                         use_trigger_batch=True)

      # entropy = others[0]
      srl_logits_transposed_mm_list = []
      if hparams.add_exclusion_dist:
        num_clusters -= 1
      for scorer_id in range(num_clusters):
        with tf.variable_scope("biaffine_scorer_{}".format(scorer_id)):
          # srl_logits_transposed_mm = bilinear_scoring(external_role=role_mlp, external_predicate=predicate_mlp,
          #                                             external_role_size=role_mlp_size,
          #                                             external_predicate_size=predicate_mlp_size)
          srl_logits_transposed_mm = bilinear_scoring(external_role=role_mlp, external_predicate=predicate_mlp,
                                                      external_role_size=role_mlp_size,
                                                      external_predicate_size=predicate_mlp_size,
                                                      cluster_emb=(None,
                                                                   final_slot) if hparams.use_lstm and hparams.disc_final_slot else None,
                                                      concat_pred=False,
                                                      cluster_emb_only=False,
                                                      use_distance=hparams.bilinear_as_distance)
          srl_logits_transposed_mm_list.append(srl_logits_transposed_mm)
      if hparams.add_exclusion_dist:
        num_clusters += 1
        zeros_dist = tf.zeros_like(srl_logits_transposed_mm)
        zeros_dist = tf.concat([zeros_dist[:, :, :1] + constants.VERY_LARGE, zeros_dist[:, :, 1:]], axis=-1)
        srl_logits_transposed_mm_list.append(zeros_dist)
      # else:
      #   raise NotImplementedError
      if hparams.mixture_model_inference_mode == 'nll' or hparams.mixture_model_inference_mode == 'gumbel':
        #   raise NotImplementedError
        # elif hparams.mixture_model_inference_mode == 'gumbel':
        ##########################
        # z_prob @ normal scale
        # srl_logits_transposed_mm @ log-sacle
        ##########################
        # srl_logits_transposed_mm: (P, seq,num_cluster, srl_tags)
        srl_logits_transposed_mm = tf.stack(srl_logits_transposed_mm_list, axis=-2)
        # predicate_mask: (P, seq, num_clusters, 1)
        if hparams.legacy_mode:
          gathered_mm_mask = tf.gather_nd(mm_mask, predicate_gather_indices)
        else:
          gathered_mm_mask = mm_mask
        # predicate_mask: (P, seq, 1, num_clusters)
        predicate_mask = tf.expand_dims(gathered_mm_mask, axis=-2)
        z_prob = predicate_mask
        if mode != ModeKeys.TRAIN and hparams.exact_est_of_prob:
          def log_prod(l, r):
            l = tf.expand_dims(l, -2)
            r = tf.expand_dims(tf.transpose(r, perm=[0, 1, 3, 2]), dim=-3)
            tmp = tf.math.reduce_logsumexp(l + r, axis=-1)
            return tmp

          srl_logits_transposed_mm_sampled = tf.squeeze(log_prod(tf.nn.log_softmax(z_prob, axis=-1),
                                                                 tf.nn.log_softmax(srl_logits_transposed_mm, axis=-1)),
                                                        axis=-2)
        else:
          if hparams.correct_vi_objective:
            srl_logits_transposed_mm_sampled = tf.squeeze(
              tf.matmul(tf.nn.softmax(z_prob), tf.nn.log_softmax(srl_logits_transposed_mm, axis=-1)), axis=-2)
          else:
            # z_prob = tf.Print(z_prob, [])
            raise NotImplementedError
          if hparams.sharpen_z_prob:
            z_prob_gathered = tf.gather_nd(predicate_mask, predicate_gather_indices)
            tokens_to_keep_gathered = tf.gather_nd(tokens_to_keep, predicate_gather_indices)
            z_prob_entropy = tf.nn.softmax_cross_entropy_with_logits(
              labels=tf.reshape(tf.nn.softmax(z_prob_gathered), [-1, num_clusters]),
              logits=tf.reshape(z_prob_gathered, [-1, num_clusters]))
            # z_prob_entropy = tf.Print(z_prob_entropy, [z_prob_entropy], "z_prob_entropy")
            z_prob_entropy_loss = tf.reduce_sum(z_prob_entropy * tokens_to_keep_gathered) / tf.reduce_sum(
              tokens_to_keep_gathered)
            additional_loss += tf.cond(tf.greater_equal(tf.train.get_global_step(), 8000),
                                       lambda: z_prob_entropy_loss, lambda: 0.)

            # additional_loss += z_prob_entropy_loss
            # srl_logits_transposed_mm_sampled_ns = tf.squeeze(tf.matmul(z_prob, tf.nn.softmax(srl_logits_transposed_mm, axis=-1)), axis=-2)
            # srl_logits_transposed_mm_sampled = tf.math.log(srl_logits_transposed_mm_sampled_ns)
        # z_prob = tf.nn.softmax(predicate_mask, axis=-2)
        # z_prob = tf.Print(z_prob, [tf.reduce_min(z_prob)], "z_prob")
      else:
        raise NotImplementedError
      pri_list.append(srl_logits_transposed_mm_sampled)

    if hparams.hard_pruning:
      # num_clusters = hparams.mixture_model
      # if not hparams.use_lstm:
      with tf.variable_scope("shared_pred_role_mlp_latent"):
        injected_hidden = nn_utils.MLP(inputs, hparams.latent_hidden_size,
                                       keep_prob=hparams.mlp_dropout)
      with tf.variable_scope("shared_pred_role_mlp"):
        predicate_role_mlp = nn_utils.MLP(inputs, predicate_mlp_size + role_mlp_size,
                                          keep_prob=hparams.mlp_dropout)
        predicate_mlp, role_mlp = predicate_role_mlp[:, :, :predicate_mlp_size], \
                                  predicate_role_mlp[:, :, predicate_mlp_size:]
      if not hparams.use_lstm:
        mm_mask, additional_loss = get_dep_transition_wfs(parse_gold,
                                                          tokens_to_keep=tokens_to_keep,
                                                          parse_labels=parse_label, use_lstm=hparams.use_lstm,
                                                          return_last=hparams.return_last,
                                                          mode=mode, num_clusters=num_clusters,
                                                          inference_mode=hparams.mixture_model_inference_mode,
                                                          gumbel_temperature=hparams.gumbel_temperature,
                                                          cluster_prob=hparams.cluster_prob,
                                                          propagation_depth=hparams.downward_propagation_depth,
                                                          pred_mlp=None,
                                                          role_mlp=None,
                                                          use_pred_role_feature=hparams.use_pred_role_feature,
                                                          predicate_gather_indices=predicate_gather_indices,
                                                          legacy_mode=hparams.legacy_mode,
                                                          layer_norm=hparams.wfs_layer_norm,
                                                          before_after_indicator=hparams.before_after_indicator,
                                                          max_up_depth=hparams.lstm_search_up_depth,
                                                          max_down_depth=hparams.lstm_search_down_depth,
                                                          use_gumbel_max=hparams.use_gumbel_max_on_input,
                                                          use_dep_label=hparams.latent_use_dep_label,

                                                          ignore_masks=hparams.ignore_mask)
      else:
        raise NotImplementedError

      # entropy = others[0]
      srl_logits_transposed_mm_list = []
      num_clusters = 2
      for scorer_id in range(1):
        with tf.variable_scope("biaffine_scorer_{}".format(scorer_id)):
          # srl_logits_transposed_mm = bilinear_scoring(external_role=role_mlp, external_predicate=predicate_mlp,
          #                                             external_role_size=role_mlp_size,
          #                                             external_predicate_size=predicate_mlp_size)
          srl_logits_transposed_mm = bilinear_scoring(external_role=role_mlp, external_predicate=predicate_mlp,
                                                      external_role_size=role_mlp_size,
                                                      external_predicate_size=predicate_mlp_size,
                                                      cluster_emb=(None,
                                                                   final_slot) if hparams.use_lstm and hparams.disc_final_slot else None,
                                                      concat_pred=False,
                                                      cluster_emb_only=False,
                                                      use_distance=hparams.bilinear_as_distance)
          srl_logits_transposed_mm_list.append(srl_logits_transposed_mm)
      always_ng_mtx_zeros = tf.zeros_like(srl_logits_transposed_mm)
      always_ng_mtx = tf.concat([always_ng_mtx_zeros[:, :, :1] + 100., always_ng_mtx_zeros[:, :, 1:]], axis=-1)
      srl_logits_transposed_mm_list = [always_ng_mtx] + srl_logits_transposed_mm_list
      if hparams.mixture_model_inference_mode == 'nll' or hparams.mixture_model_inference_mode == 'gumbel':
        #   raise NotImplementedError
        # elif hparams.mixture_model_inference_mode == 'gumbel':
        ##########################
        # z_prob @ normal scale
        # srl_logits_transposed_mm @ log-sacle
        ##########################
        # srl_logits_transposed_mm: (P, seq,num_cluster, srl_tags)
        srl_logits_transposed_mm = tf.stack(srl_logits_transposed_mm_list, axis=-2)
        # predicate_mask: (P, seq, num_clusters, 1)
        if hparams.legacy_mode:
          gathered_mm_mask = tf.gather_nd(mm_mask, predicate_gather_indices)
        else:
          gathered_mm_mask = mm_mask
        predicate_mask = tf.expand_dims(gathered_mm_mask, axis=-2)
        z_prob = predicate_mask
        srl_logits_transposed_mm_sampled_ns = tf.squeeze(
          tf.matmul(z_prob, tf.nn.softmax(srl_logits_transposed_mm, axis=-1)), axis=-2)
        srl_logits_transposed_mm_sampled = tf.math.log(srl_logits_transposed_mm_sampled_ns)
        # z_prob = tf.nn.softmax(predicate_mask, axis=-2)
        # z_prob = tf.Print(z_prob, [tf.reduce_min(z_prob)], "z_prob")
      else:
        raise NotImplementedError
      pri_list.append(srl_logits_transposed_mm_sampled)

    if hparams.mixture_model_embedding > 0:
      num_clusters = hparams.mixture_model_embedding
      # if not hparams.use_lstm:
      if hparams.share_pred_role_mlp:
        with tf.variable_scope("shared_pred_role_mlp"):
          predicate_mlp_size = 128
          role_mlp_size = 128
          predicate_role_mlp = nn_utils.MLP(inputs, predicate_mlp_size + role_mlp_size,
                                            keep_prob=hparams.mlp_dropout)
          shared_predicate_mlp, shared_role_mlp = predicate_role_mlp[:, :, :predicate_mlp_size], \
                                                  predicate_role_mlp[:, :, predicate_mlp_size:]
          if hparams.normalize_mlp_to_unit:
            shared_predicate_mlp, _ = tf.linalg.normalize(shared_predicate_mlp, axis=-1)
            shared_role_mlp, _ = tf.linalg.normalize(shared_role_mlp, axis=-1)
            tf.logging.log(tf.logging.INFO, "normalized tensor {} {}".format(shared_predicate_mlp, shared_role_mlp))
      else:
        with tf.variable_scope("shared_pred_role_mlp"):
          latent_mlp = nn_utils.MLP(inputs, hparams.latent_hidden_size,
                                    keep_prob=hparams.mlp_dropout)
      if not hparams.use_lstm:
        mm_mask, others = get_dep_transition_wfs(parse_gold,
                                                 tokens_to_keep=tokens_to_keep,
                                                 parse_labels=parse_label, use_lstm=hparams.use_lstm,
                                                 return_last=hparams.return_last,
                                                 mode=mode, num_clusters=num_clusters,
                                                 inference_mode=hparams.mixture_model_inference_mode,
                                                 gumbel_temperature=hparams.gumbel_temperature,
                                                 cluster_prob=hparams.cluster_prob,
                                                 propagation_depth=hparams.downward_propagation_depth,
                                                 pred_mlp=latent_mlp,
                                                 role_mlp=latent_mlp,
                                                 use_pred_role_feature=hparams.use_pred_role_feature,
                                                 predicate_gather_indices=predicate_gather_indices,
                                                 legacy_mode=hparams.legacy_mode,
                                                 layer_norm=hparams.wfs_layer_norm,
                                                 before_after_indicator=hparams.before_after_indicator,
                                                 use_gumbel_max=hparams.use_gumbel_max_on_input)

      else:
        if hparams.use_lexicons:
          if hparams.share_pred_role_mlp:
            inject_hidden = shared_role_mlp
          else:
            inject_hidden = latent_mlp
        else:
          inject_hidden = None
        _, others, mm_mask = get_dep_transition_wfs_dp(parse_gold,
                                                       tokens_to_keep=tokens_to_keep,
                                                       parse_labels=parse_label,
                                                       mode=mode, num_clusters=num_clusters,
                                                       inference_mode=hparams.mixture_model_inference_mode,
                                                       predicate_gather_indices=predicate_gather_indices,
                                                       batched_predicate_gather_indices=(
                                                       batched_predicate_gather_indices, unbatch_bpgi),
                                                       return_gathered_hiddens=not hparams.use_lstm_hiddens,
                                                       half_return_size=hparams.half_return_size,
                                                       l1_regularizer=hparams.l1_dense_regularizer,
                                                       layer_norm=hparams.wfs_layer_norm,
                                                       use_bai=hparams.before_after_indicator,
                                                       num_samples=num_samples, hiddens=inject_hidden,
                                                       latent_hidden_size=hparams.latent_hidden_size if not hparams.share_pred_role_mlp else role_mlp_size,
                                                       use_direction=hparams.latent_use_direction,
                                                       use_lr_dir=hparams.latent_use_lr_direction,
                                                       use_dep_label=hparams.latent_use_dep_label,
                                                       max_up_depth=hparams.lstm_search_up_depth,
                                                       max_down_depth=hparams.lstm_search_down_depth,
                                                       use_gumbel_max=hparams.use_gumbel_max_on_input,
                                                       returns_lstm_state=True, use_trigger_batch=True)
      # entropy = others[0]
      srl_logits_transposed_mm_list = []
      if hparams.add_exclusion_dist:
        num_clusters -= 1
      for scorer_id in range(1):
        with tf.variable_scope("biaffine_scorer_{}".format(1)):
          # pred_count = tf.shape(mm_mask)[0]
          # seq_len = tf.shape(mm_mask)[1]
          # cluster_hidden_pred = tf.matmul(mm_mask, cluster_emb)#tf.tile(tf.reshape(cluster_emb[scorer_id], [1, 1, 128]), [pred_count,1, 1])
          cluster_hidden_role = mm_mask
          if not hparams.share_pred_role_mlp:
            predicate_role_mlp = nn_utils.MLP(inputs, predicate_mlp_size + role_mlp_size,
                                              keep_prob=hparams.mlp_dropout)
            predicate_mlp, role_mlp = predicate_role_mlp[:, :, :predicate_mlp_size], \
                                      predicate_role_mlp[:, :,
                                      predicate_mlp_size:] if not hparams.pred_role_share_map else predicate_role_mlp[:,
                                                                                                   :,
                                                                                                   :predicate_mlp_size]
          else:
            predicate_mlp, role_mlp = shared_predicate_mlp, shared_role_mlp
          if hparams.normalize_mlp_to_unit:
            predicate_mlp, _ = tf.linalg.normalize(predicate_mlp, axis=-1)
            role_mlp, _ = tf.linalg.normalize(role_mlp, axis=-1)

          if hparams.separate_bilinear_cluster:
            srl_logits_transposed_mm = bilinear_scoring(external_role=role_mlp, external_predicate=predicate_mlp,
                                                        external_role_size=role_mlp_size,
                                                        external_predicate_size=predicate_mlp_size,
                                                        cluster_emb=None, concat_pred=False, cluster_emb_only=False,
                                                        use_distance=hparams.bilinear_as_distance)
            with tf.variable_scope("prior_scope"):
              dense_1 = tf.keras.layers.Dense(512, activation='relu')
              dense_2 = tf.keras.layers.Dense(256, activation='relu')
              dense_o = tf.keras.layers.Dense(num_labels)
              srl_logits_transposed_mm += dense_o(dense_2(dense_1(mm_mask)))
          else:
            srl_logits_transposed_mm = bilinear_scoring(external_role=role_mlp, external_predicate=predicate_mlp,
                                                        external_role_size=role_mlp_size,
                                                        external_predicate_size=predicate_mlp_size,
                                                        cluster_emb=(None, cluster_hidden_role), concat_pred=False,
                                                        cluster_emb_only=False,
                                                        use_distance=hparams.bilinear_as_distance)
          srl_logits_transposed_mm_list.append(srl_logits_transposed_mm)
      if hparams.mixture_model_inference_mode == 'gumbel':
        ##########################
        # z_prob @ normal scale
        # srl_logits_transposed_mm @ log-sacle
        ##########################
        # srl_logits_transposed_mm: (P, seq, srl_tags)
        srl_logits_transposed_mm = srl_logits_transposed_mm_list[0]  # tf.stack(srl_logits_transposed_mm_list, axis=-2)
        # predicate_mask: (P, seq, num_clusters, 1)
        # gathered_mm_mask = mm_mask
        # predicate_mask = tf.expand_dims(gathered_mm_mask, axis=-1)
        # z_prob = predicate_mask
      else:
        raise NotImplementedError
      pri_list.append(srl_logits_transposed_mm)

    with tf.variable_scope("weight"):
      if hparams.apply_weight_to_normal:
        pri_list.append(srl_logits_transposed_normal)
        weight = tf.get_variable("weight", [len(pri_list), 1, 1, 1])
        srl_logits_transposed = tf.reduce_sum(weight * tf.stack(pri_list, axis=0),
                                              axis=0)
      elif hparams.mixture_model > 0 or hparams.mixture_model_embedding > 0 or hparams.mixture_model_normal_embedding > 0:
        assert len(pri_list) == 1
        srl_logits_transposed = pri_list[0]
        if hparams.srl_score_layer_norm:
          with tf.variable_scope('layernorm_srl_score'):
            layernorm = tf.keras.layers.LayerNormalization(
              axis=-1, epsilon=0.001, center=True, scale=True,
              beta_initializer='zeros', gamma_initializer='ones',
              beta_regularizer=None, gamma_regularizer=None, beta_constraint=None,
              gamma_constraint=None
            )
            srl_logits_transposed = layernorm(srl_logits_transposed)
      elif hparams.apply_mean_weight:
        if hparams.exclude_specific_path and not hparams.mixture_model > 0:
          srl_logits_transposed_normal_exclude_path = tf.nn.relu(taken_mask) * srl_logits_transposed_normal
          # loss_weight_mtx += tf.nn.relu(taken_mask)[:, :, 0]
        if not hparams.remove_global_scorer:
          if not hparams.mixture_model > 0:
            pri_list.append(srl_logits_transposed_normal_exclude_path)
        else:
          default_score = tf.ones_like(srl_logits_transposed_normal)
          default_global_scorer = tf.concat([default_score[:, :, :1] * 100., default_score[:, :, 1:] * -100.], axis=-1)
          default_global_scorer *= tf.nn.relu(taken_mask)
          pri_list.append(default_global_scorer)
        tf.logging.log(tf.logging.INFO, "APPLY MEAN WEIGHT: using scorer list of {}".format(pri_list))
        if hparams.train_global_scorer_with_all_data and mode == ModeKeys.TRAIN:
          tran_global_scorer = tf.greater_equal(tf.random.uniform(shape=[], minval=-1., maxval=1.), 0)
          srl_logits_transposed = tf.cond(tran_global_scorer,
                                          lambda: srl_logits_transposed_normal,
                                          lambda: tf.reduce_sum(tf.stack(pri_list, axis=-1), axis=-1))
        else:
          srl_logits_transposed = tf.reduce_sum(tf.stack(pri_list, axis=-1), axis=-1)
      elif hparams.pattern_dominant:
        tf.logging.log(tf.logging.INFO, "using pattern dominant weight")
        weight = tf.get_variable("weight", [len(pri_list), 1, 1, 1], initializer=tf.constant_initializer(
          10.))
        if mode != ModeKeys.TRAIN:
          weight = tf.Print(weight, [weight], "aggregation weight", summarize=10)
          weight = tf.Print(weight, [srl_logits_transposed_normal], "scorer_details", summarize=40)
          weight = tf.Print(weight, [pri_list[0]], "scorer_details", summarize=40)
          weight = tf.Print(weight, [pri_list[-1]], "scorer_details", summarize=40)
          # weight = tf.Print(weight, [srl_logits_transposed_normal], "scorer_details", summarize=40)
        srl_logits_transposed = srl_logits_transposed_normal + tf.reduce_sum(weight * tf.stack(pri_list, axis=0),
                                                                             axis=0)
      else:
        if hparams.exclude_specific_path:
          srl_logits_transposed_normal *= tf.nn.relu(taken_mask)
        weight = tf.get_variable("weight", [len(pri_list), 1, 1, 1], initializer=tf.constant_initializer(
          1.) if hparams.aggregation_weight_one_init else None)
        if mode != ModeKeys.TRAIN:
          weight = tf.Print(weight, [weight], "aggregation weight", summarize=10)
          weight = tf.Print(weight, [srl_logits_transposed_normal], "scorer_details", summarize=40)
          weight = tf.Print(weight, [pri_list[0]], "scorer_details", summarize=40)
          weight = tf.Print(weight, [pri_list[-1]], "scorer_details", summarize=40)
          # weight = tf.Print(weight, [srl_logits_transposed_normal], "scorer_details", summarize=40)
        srl_logits_transposed = srl_logits_transposed_normal + tf.reduce_sum(weight * tf.stack(pri_list, axis=0),
                                                                             axis=0)

        # Here gather (num_predicates_in_batch, seq_len, num_classes) dep prior

    # (3) compute loss

    # now we have k sets of targets for the k frames
    # (p1) f1 f2 f3
    # (p2) f1 f2 f3

    # get all the tags for each token (which is the predicate for a frame), structuring
    # targets as follows (assuming p1 and p2 are predicates for f1 and f3, respectively):
    # (p1) f1 f1 f1
    # (p2) f3 f3 f3
    srl_targets_transposed = tf.transpose(targets, [0, 2, 1])

    gold_predicate_counts = tf.reduce_sum(tf.cast(bool_mask_where_predicates(predicate_targets), tf.int32), -1)
    srl_targets_indices = tf.where(tf.sequence_mask(tf.reshape(gold_predicate_counts, [-1])))

    # num_predicates_in_batch x seq_len
    srl_targets_gold_predicates = tf.gather_nd(srl_targets_transposed, srl_targets_indices)

    predicted_predicate_counts = tf.reduce_sum(tf.cast(bool_mask_where_predicates(predicate_preds), tf.int32), -1)
    srl_targets_pred_indices = tf.where(tf.sequence_mask(tf.reshape(predicted_predicate_counts, [-1])))
    srl_targets_predicted_predicates = tf.gather_nd(srl_targets_transposed, srl_targets_pred_indices)

    # num_predicates_in_batch x seq_len
    predictions = tf.cast(tf.argmax(srl_logits_transposed, axis=-1), tf.int32)

    seq_lens = tf.cast(tf.reduce_sum(mask, 1), tf.int32)
    print("srl_logit_length", srl_logits_transposed.get_shape().as_list())

    if False:  # hparams.conll09 and mode == ModeKeys.TRAIN and (hparams.mixture_model > 0) and \
      # (hparams.mixture_model_inference_mode == 'full' or hparams.mixture_model_inference_mode == 'gumbel'):
      srl_targets_onehot = tf.one_hot(indices=srl_targets_predicted_predicates, depth=num_labels, axis=-1)
      srl_targets_onehot = tf.expand_dims(srl_targets_onehot, axis=-2)
      srl_targets_onehot = tf.tile(srl_targets_onehot, multiples=[1, 1, num_clusters, 1])

      z_prob_reshaped = tf.reshape(z_prob, [-1])
      mask_tiled = tf.tile(tf.expand_dims(mask, -1), multiples=[1, 1, num_clusters])
      mask_tiled = tf.reshape(mask_tiled, [-1]) * z_prob_reshaped

      if hparams.boost_non_empty_loss > 1:
        bias_weight = tf.where(tf.not_equal(srl_targets_predicted_predicates, 0),
                               hparams.boost_non_empty_loss * tf.ones_like(srl_targets_predicted_predicates),
                               tf.ones_like(srl_targets_predicted_predicates))
        bias_weight = tf.tile(tf.expand_dims(bias_weight, -1), multiples=[1, 1, num_clusters])
        mask_tiled *= tf.reshape(tf.cast(bias_weight, tf.float32), [-1])

      loss_emission = tf.losses.softmax_cross_entropy(logits=tf.reshape(srl_logits_transposed, [-1, num_labels]),
                                                      onehot_labels=tf.reshape(srl_targets_onehot, [-1, num_labels]),
                                                      weights=tf.reshape(mask_tiled, [-1]),
                                                      label_smoothing=hparams.label_smoothing,
                                                      reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS)
      if hparams.mixture_model_inference_mode == 'gumbel':
        mask_count = tf.reduce_sum(tf.where(tf.math.greater(mask, 0), tf.ones_like(mask), tf.zeros_like(mask)))
        with_prob_count = tf.reduce_sum(
          tf.where(tf.math.greater(mask_tiled, 0), tf.ones_like(mask_tiled), tf.zeros_like(mask_tiled)))
        factor = with_prob_count / mask_count
        loss = loss_emission * factor
      else:
        loss = loss_emission * num_clusters
      if hparams.flooding:
        tf.logging.log(tf.logging.INFO, "performing loss flooding")
        loss = tf.abs(loss - 0.2) + 0.2
      loss += additional_loss
      # tf.logging.log(tf.logging.INFO, "adding entropy as loss")
      # loss += tf.reduce_mean(entropy * mask)

    if hparams.conll09 and mode == ModeKeys.TRAIN and hparams.mixture_model_normal_embedding > 0:
      srl_targets_onehot = tf.one_hot(indices=srl_targets_predicted_predicates, depth=num_labels, axis=-1)
      srl_targets_onehot = tf.tile(srl_targets_onehot, multiples=[num_samples, 1, 1])

      mask_tiled = tf.tile(mask, multiples=[num_samples, 1])
      mask_tiled = tf.reshape(mask_tiled, [-1])

      loss_emission = tf.losses.softmax_cross_entropy(logits=tf.reshape(srl_logits_transposed, [-1, num_labels]),
                                                      onehot_labels=tf.reshape(srl_targets_onehot, [-1, num_labels]),
                                                      weights=tf.reshape(mask_tiled, [-1]),
                                                      label_smoothing=hparams.label_smoothing,
                                                      reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS)
      loss = loss_emission + tf.reduce_mean(entropy)

    if False:  # hparams.conll09 and (mode == ModeKeys.PREDICT or mode == ModeKeys.EVAL) and (hparams.mixture_model > 0) and \
      # (hparams.mixture_model_inference_mode == 'full' or hparams.mixture_model_inference_mode == 'gumbel'):
      srl_logits_transposed = tf.reduce_sum(tf.nn.softmax(srl_logits_transposed, axis=-1) * z_prob, axis=-2)
      srl_logits_transposed = tf.math.log(srl_logits_transposed)
      predictions = tf.cast(tf.argmax(srl_logits_transposed, axis=-1), tf.int32)
      srl_targets_onehot = tf.one_hot(indices=srl_targets_predicted_predicates, depth=num_labels, axis=-1)
      loss_emission = tf.losses.softmax_cross_entropy(logits=tf.reshape(srl_logits_transposed, [-1, num_labels]),
                                                      onehot_labels=tf.reshape(srl_targets_onehot, [-1, num_labels]),
                                                      weights=tf.reshape(mask, [-1]),
                                                      # label_smoothing=hparams.label_smoothing,
                                                      reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS)
      if hparams.mixture_model_inference_mode == 'gumbel':
        loss = loss_emission
      else:
        loss = loss_emission * num_clusters

    if hparams.conll09 and (
        mode == ModeKeys.PREDICT or mode == ModeKeys.EVAL) and hparams.mixture_model_normal_embedding > 0:
      num_predicates = tf.cast(tf.shape(srl_logits_transposed)[0] / num_samples, tf.int32)
      seq_len = tf.shape(srl_logits_transposed)[1]
      srl_logits_transposed_reshaped = tf.reshape(tf.expand_dims(tf.nn.softmax(srl_logits_transposed, axis=-1), axis=0),
                                                  [num_samples, num_predicates, seq_len, num_labels])
      srl_logits_transposed = tf.reduce_mean(srl_logits_transposed_reshaped, axis=0)
      srl_logits_transposed = tf.math.log(srl_logits_transposed)
      predictions = tf.cast(tf.argmax(srl_logits_transposed, axis=-1), tf.int32)
      srl_targets_onehot = tf.one_hot(indices=srl_targets_predicted_predicates, depth=num_labels, axis=-1)
      loss_emission = tf.losses.softmax_cross_entropy(logits=tf.reshape(srl_logits_transposed, [-1, num_labels]),
                                                      onehot_labels=tf.reshape(srl_targets_onehot, [-1, num_labels]),
                                                      weights=tf.reshape(mask, [-1]),
                                                      # label_smoothing=hparams.label_smoothing,
                                                      reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS)
      if hparams.mixture_model_inference_mode == 'gumbel':
        loss = loss_emission
      else:
        loss = loss_emission * num_clusters

    if hparams.conll09 and (mode == ModeKeys.PREDICT or mode == ModeKeys.EVAL) and \
        not (hparams.mixture_model_normal_embedding > 0):  # hparams.mixture_model > 0 or
      srl_targets_onehot = tf.one_hot(indices=srl_targets_predicted_predicates, depth=num_labels, axis=-1)
      loss_emission = tf.losses.softmax_cross_entropy(logits=tf.reshape(srl_logits_transposed, [-1, num_labels]),
                                                      onehot_labels=tf.reshape(srl_targets_onehot, [-1, num_labels]),
                                                      weights=tf.reshape(mask, [-1]),
                                                      # label_smoothing=hparams.label_smoothing,
                                                      reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS)

      loss = loss_emission

    if hparams.conll09 and mode == ModeKeys.TRAIN and \
        not (hparams.mixture_model_normal_embedding > 0):  # hparams.mixture_model > 0 or
      srl_targets_onehot = tf.one_hot(indices=srl_targets_predicted_predicates, depth=num_labels, axis=-1)
      loss_emission = tf.losses.softmax_cross_entropy(logits=tf.reshape(srl_logits_transposed, [-1, num_labels]),
                                                      onehot_labels=tf.reshape(srl_targets_onehot, [-1, num_labels]),
                                                      weights=tf.reshape(mask, [-1]),
                                                      label_smoothing=hparams.label_smoothing,
                                                      reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS)
      loss = loss_emission
      # loss = tf.Print(loss, [loss], "loss")
      # if hparams.mixture_model_embedding > 0:
      #   tf.logging.log(tf.logging.INFO, "adding entropy as loss")
      #   loss += tf.reduce_mean(entropy*mask)

    if hparams.conll05 and mode == ModeKeys.TRAIN:
      print("check tensor shape", mask, srl_logits_transposed, srl_targets_predicted_predicates)
      assert hparams.mixture_model_inference_mode == 'full'
      assert transition_params is not None
      assert not tf_utils.is_trainable(transition_params)
      srl_targets_onehot = tf.one_hot(indices=srl_targets_predicted_predicates, depth=num_labels, axis=-1)
      srl_targets_onehot = tf.expand_dims(srl_targets_onehot, axis=-2)
      srl_targets_onehot = tf.tile(srl_targets_onehot, multiples=[1, 1, num_clusters, 1])

      z_prob_reshaped = tf.reshape(z_prob, [-1])
      mask = tf.tile(tf.expand_dims(mask, -1), multiples=[1, 1, num_clusters])
      mask = tf.reshape(mask, [-1]) * z_prob_reshaped
      # mask = tf.Print(mask, [mask, tf.reduce_min(mask), tf.reduce_min(z_prob)], "mask")
      loss_emission = tf.losses.softmax_cross_entropy(logits=tf.reshape(srl_logits_transposed, [-1, num_labels]),
                                                      onehot_labels=tf.reshape(srl_targets_onehot, [-1, num_labels]),
                                                      weights=tf.reshape(mask, [-1]),
                                                      label_smoothing=hparams.label_smoothing,
                                                      reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS)

      loss = loss_emission * num_clusters
      # loss = tf.Print(loss, [loss], "loss")

    if hparams.conll05 and (mode == ModeKeys.PREDICT or mode == ModeKeys.EVAL):
      srl_logits_transposed = tf.reduce_sum(tf.nn.softmax(srl_logits_transposed, axis=-1) * z_prob, axis=-2)
      srl_logits_transposed = tf.math.log(srl_logits_transposed)
      predictions, score = tf.contrib.crf.crf_decode(srl_logits_transposed, transition_params, seq_lens)
      log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(tf.stop_gradient(srl_logits_transposed),
                                                                            srl_targets_predicted_predicates,
                                                                            seq_lens,
                                                                            tf.stop_gradient(transition_params))
      loss = tf.reduce_mean(-log_likelihood)

    output = {
      'loss': loss + additional_loss,
      'predictions': predictions,
      'scores': srl_logits_transposed,
      'targets': srl_targets_gold_predicates,
      'probabilities': tf.nn.softmax(srl_logits_transposed, -1)
    }

    return output


dispatcher = {
  'srl_bilinear': srl_bilinear,
  'srl_bilinear_dep_prior': srl_bilinear_dep_prior,
  'srl_bilinear_dep_prior_pos': srl_bilinear_dep_prior_pos,
  'srl_bilinear_sm': srl_bilinear_sm,
  'srl_bilinear_dep_prior_bilinear': srl_bilinear_dep_prior_bilinear,
  'joint_softmax_classifier': joint_softmax_classifier,
  'joint_softmax_classifier_wsm': joint_softmax_classifier_wsm,
  'joint_softmax_classifier_ls': joint_softmax_classifier_ls,
  'softmax_classifier': softmax_classifier,
  'softmax_classifier_2': softmax_classifier_2,
  'parse_bilinear': parse_bilinear,
  'parse_bilinear_msm': parse_bilinear_msm,
  'parse_bilinear_ls': parse_bilinear_ls,
  'conditional_bilinear': conditional_bilinear,
  'conditional_bilinear_ls': conditional_bilinear_ls,
  'parse_bilinear_with_decedents': parse_bilinear_with_decedents,
  'parse_bilinear_sigmoid': parse_bilinear_sigmoid
}


def dispatch(fn_name):
  try:
    return dispatcher[fn_name]
  except KeyError:
    print('Undefined output function `%s' % fn_name)
    exit(1)


# need to decide shape/form of train_outputs!
def get_params(mode, model_config, task_map, train_outputs, features, labels, current_outputs, task_labels, num_labels,
               joint_lookup_maps, tokens_to_keep, transition_params, hparams):
  params = {'mode': mode, 'model_config': model_config, 'inputs': current_outputs, 'targets': task_labels,
            'tokens_to_keep': tokens_to_keep, 'num_labels': num_labels, 'transition_params': transition_params,
            'hparams': hparams}
  params_map = task_map['params'] if 'params' in task_map else {}
  for param_name, param_values in params_map.items():
    print(param_name, ": ", param_values)

    # if this is a map-type param, do map lookups and pass those through
    if 'joint_maps' in param_values:
      params[param_name] = {map_name: joint_lookup_maps[map_name] for map_name in param_values['joint_maps']}
    elif 'label' in param_values:
      params[param_name] = labels[param_values['label']]
    elif 'feature' in param_values:
      params[param_name] = features[param_values['feature']]
    # otherwise, this is a previous-prediction-type param, look those up and pass through
    elif 'transformation' == param_name:
      params['targets'] = transformation_fn.dispatch(param_values)(params['targets'])
      # raise NotImplementedError
    elif 'layer' in param_values:
      # print("debug <train outputs>: ", train_outputs)
      outputs_layer = train_outputs[param_values['layer']]
      params[param_name] = outputs_layer[param_values['output']]
    else:
      params[param_name] = param_values['value']
    # print("debug <get params, returned params>:", params)
  return params
