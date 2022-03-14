from functools import partial

import tensorflow as tf
import constants
import nn_utils

# import transformation_fn

head_gen_library = {
  "labeled_div": partial(nn_utils.generating_head_mtx_from_head_label_dist, head_label_aggregation_fn=tf.math.truediv),
  "labeled_mul": partial(nn_utils.generating_head_mtx_from_head_label_dist, head_label_aggregation_fn=tf.math.multiply),
}
decedent_gen_library = {
  "labeled_div": partial(nn_utils.generating_decedent_mtx_from_head_label_dist, head_label_aggregation_fn=tf.math.truediv),
  "labeled_mul": partial(nn_utils.generating_decedent_mtx_from_head_label_dist, head_label_aggregation_fn=tf.math.multiply),
}



def roll(matrix, shift):
  # Assuming it to be a 3d-tensor of shape (batch, seq_len, seq_len)
  if shift > 0: #which means it's rightward shifting
    shift = tf.abs(shift)
    return tf.concat([ matrix[:, shift:, :], tf.zeros_like(matrix)[:, :shift,  :]], axis = 1) # In implementation it's actually up-rolling
  elif shift < 0:
    shift = tf.abs(shift)
    return tf.concat([matrix[:, :, shift:], tf.zeros_like(matrix)[:, :, :shift]], axis= 2)
  else:
    return matrix

def get_decedent_mtx(heads):
  # heads: (B, S)
  heads = tf.cast(heads, tf.int32)
  seq_len = tf.shape(heads)[1]
  # print(seq_len)
  array_idx = tf.range(400, dtype=tf.int32)[:seq_len]
  mtx_idxer = tf.tile(tf.reshape(array_idx, [-1, 1]), [1, seq_len])
  idxer = tf.tile(tf.reshape(array_idx, [1, -1]), [seq_len, 1])
  heads = tf.tile(tf.expand_dims(heads, 1), [1, seq_len, 1])
  one_mtx = tf.ones_like(mtx_idxer, dtype=tf.float32)
  zero_mtx = tf.zeros_like(mtx_idxer, dtype=tf.float32)
  decedent_mtx = tf.map_fn(lambda x: tf.where(tf.equal(mtx_idxer, x),  one_mtx, zero_mtx), heads, dtype=tf.float32)
  return decedent_mtx

def get_decedent_mtx_from_score(heads):
  heads_dist = tf.nn.softmax(heads)
  dependent_scores = tf.transpose(heads_dist, perm=[0, 2, 1])
  dependent_dist = tf.nn.softmax(dependent_scores)
  return dependent_dist


def get_k_adjacent_mtx(heads, chain):
  def multiplication(head, decedent):
    tmp = tf.eye(tf.shape(head)[1], batch_shape=[tf.shape(head)[0]])
    for direction in chain:
      if direction == "up":
        tmp = tf.linalg.matmul(tmp, head)
      elif direction == "down":
        tmp = tf.linalg.matmul(tmp, decedent)
    return tmp

  # with tf.device('/cpu:0'):
  if len(heads.get_shape()) < 3:
    heads = tf.one_hot(heads, tf.shape(heads)[-1], off_value=constants.VERY_SMALL, on_value=constants.VERY_LARGE)
    decedents = tf.transpose(heads, perm=[0, 2, 1])
  else:
    heads = heads
    decedents = tf.transpose(heads, perm=[0, 2, 1])
  head_adjacent_mtx = tf.nn.softmax(heads)
  decedent_adjacent_mtx = tf.nn.softmax(decedents)
  # k_dist_mtx = order_dist(head_adjacent_mtx)#
  result = multiplication(head_adjacent_mtx, decedent_adjacent_mtx)
  return result#tf.clip_by_value(tf.math.log(result), constants.VERY_SMALL*2, 0, name=None)+constants.VERY_LARGE

def get_k_adjacent_mtx_hard(heads, chain, tokens_to_keep):
  raise NotImplementedError


def get_hsdp_adjacent_mtx(heads, chain, tokens_to_keep):
  raise NotImplementedError


def remove_self_recurrent(input):
  mask = tf.eye(tf.shape(input)[-1], batch_shape=[tf.shape(input)[0]]) * constants.VERY_SMALL
  return input + mask


def adjacent_mtx_normalization(mtx, mode="none"):
  raise NotImplementedError


def adding_head_noise(heads, dropout_rate = 1., off_value = -10e0):
  ones = tf.ones_like(heads, dtype=tf.float32)
  noise = tf.nn.dropout(ones, keep_prob=dropout_rate, noise_shape=tf.concat([tf.shape(ones)[:-1], [1]], axis=0)) * dropout_rate
  dropouted_heads = ones * off_value * (1-noise) + (noise) * heads
  return dropouted_heads

def head_replacing_fn(heads, tokens_to_keep, replacing_rate = 9e-1):
  # Note: heads here is of shape [b, seq]
  seq_lens = tf.cast(tf.expand_dims(tf.reduce_sum(tokens_to_keep, -1), axis=-1), dtype = tf.float32)
  random_list = tf.cast(tf.round(tf.random.uniform(shape=tf.shape(heads)) * seq_lens), dtype=tf.int32)
  ones = tf.ones_like(heads, dtype=tf.float32)
  noise = tf.cast(tf.nn.dropout(ones, keep_prob=replacing_rate, noise_shape=tf.shape(ones)) * replacing_rate, dtype=tf.int32)

  replaced_heads = random_list * (1-noise) + noise * heads
  return replaced_heads#tf.cast(replaced_heads, dtype=tf.int32)

def label_replacing_fn(labels, num_labels, replacing_rate = 9e-1):
  # Note: heads here is of shape [b, seq]
  #seq_lens = tf.cast(tf.expand_dims(tf.reduce_sum(tokens_to_keep, -1), axis=-1), dtype = tf.float32
  random_list = tf.random.uniform(shape=tf.shape(labels), maxval=num_labels, dtype=tf.int32)
  ones = tf.ones_like(labels, dtype=tf.float32)
  noise = tf.cast(tf.nn.dropout(ones, keep_prob=replacing_rate, noise_shape=tf.shape(ones)) * replacing_rate, dtype=tf.int32)

  replaced_heads = random_list * (1-noise) + noise * labels
  return replaced_heads#tf.cast(replaced_heads, dtype=tf.int32)


def normalize_prob_gt_1(mtx):
  raise NotImplementedError

def get_labeled_adjacent_mtx(parse_gold, parse_label, chain, num_labels, tokens_to_keep, smoothing_softmax = False,
                             head_gen_fn_train=head_gen_library["labeled_div"],
                             head_gen_fn_eval=head_gen_library["labeled_div"],
                             ignore_recurrent_head = True, normalization = "none", using_log_prob = True, allow_intermediate_nodes=False, on_prob = 9e-1,
                             selective_gating = False, head_dropout = False, head_dropout_rate =1., head_replacing = False, head_replacing_rate = 1.0, prod_mode = "eye-end_node-noop", ls_multiplier = 1.,
                             label_replacing=False, label_replacing_rate = 1., new_masking = False, extreme_value = False):
  """
    heads: head-dependent distribution of shape (B, seq_len, seq_len)
    labels: label distribution for each head-dependent choice, shape: (B, seq_len, labels)
  """
  heads = parse_gold
  labels = parse_label
  # token_mask = tf.expand_dims(tf.cast(1 - tokens_to_keep, tf.float32) * constants.VERY_SMALL, 1)
  token_mask_row = tf.expand_dims(tf.cast(1 - tokens_to_keep, tf.float32) * constants.VERY_SMALL, 1)
  token_mask_col = tf.expand_dims(tf.cast(1 - tokens_to_keep, tf.float32) * constants.VERY_SMALL, 2)

  tf.logging.log(tf.logging.INFO, "Using labeled k-adjacency matrix gen with chain {}".format(chain))
  tf.logging.log(tf.logging.INFO, "Using head_gen_fn_train {}".format(head_gen_fn_train))
  tf.logging.log(tf.logging.INFO, "Using head_gen_fn_eval {}".format(head_gen_fn_eval))
  tf.logging.log(tf.logging.INFO, "Using extreme_value {} ".format(extreme_value))
  tf.logging.log(tf.logging.INFO, "prod mode {}".format(prod_mode))
  tf.logging.log(tf.logging.INFO, "Head replacing mode {} with rate {}".format(head_replacing, head_replacing_rate))
  tf.logging.log(tf.logging.INFO, "Label replacing mode {} with rate {}".format(label_replacing, label_replacing_rate))

  tf.logging.log(tf.logging.INFO, "Head dropping out mode {} with rate {}".format(head_dropout, head_dropout_rate))
  tf.logging.log(tf.logging.INFO, "Receiving head/labels {}/{}".format(heads, labels))

  direction_seq = []
  for chain_item in chain:
    direction = chain_item[0]
    step = int(chain_item[1])
    direction_seq += [True if direction == "u" else False for _ in range(step)]

  def multiplication_v2(head, chain):
    print("head, direction_seq: ", head, direction_seq)
    def normalize_prob_gt_1(mtx):
      prob_mass = tf.reduce_logsumexp(mtx, -1, keep_dims=True)
      ind = tf.cast(tf.greater(prob_mass, 0.), dtype=tf.float32)
      # ind = tf.Print(ind, [prob_mass, ind], "prob mass & gt1 indicator")
      mtx -= ind * prob_mass
      mtx = tf.Print(mtx, [prob_mass, ind, tf.reduce_logsumexp(mtx, -1, keep_dims=True)], "prob mass & gt1 indicator")
      return mtx


    def update_activation(next_step, current_activation):
      return tf.math.maximum(next_step, current_activation)
    init_activation_mode, output_mode, init_intermediate_mode = prod_mode.split("-")
    def masking(activation, next_step):
      small_value = tf.ones(tf.shape(next_step), dtype=tf.float32) * constants.VERY_SMALL
      # small_value = -1e2
      return tf.where(tf.greater(activation, next_step), small_value, next_step)
    def masking_self_loop(next_step):
      small_value = tf.eye(tf.shape(next_step)[-1], batch_shape=[tf.shape(next_step)[0]]) * constants.VERY_SMALL
      return next_step + small_value
    def prod(l, r, activation_map):
      l = tf.expand_dims(l, 2)
      r = tf.expand_dims(tf.transpose(r, perm=[0, 2, 1]), dim=1)
      tmp = tf.math.reduce_logsumexp(l + r, axis=-1)

      return masking(activation_map, tmp)

    if len(direction_seq) == 1 and direction_seq[0] and output_mode == "end_node" and init_activation_mode == "noop":
      tf.logging.log(tf.logging.INFO, "Using shortcut for NEN, returning the first element {} of {}".format(head[0], head))
      return head[0]
    if len(direction_seq) == 1 and direction_seq[0] and output_mode == "end_node" and init_activation_mode == "eye":
      tf.logging.log(tf.logging.INFO, "Using shortcut for EEN, returning the first element {} of {}".format(head[0], head))
      return masking_self_loop(head[0])
    raise NotImplementedError

  if len(heads.get_shape()) < 3:
    # if head_replacing:
    #   heads = head_replacing_fn(tf.cast(heads, dtype=tf.int32), tokens_to_keep, head_replacing_rate)
    max_seq_len = tf.cast(tf.shape(heads)[-1], dtype=tf.float32)
    on_value = -1e-1
    off_value = tf.math.log(tf.math.exp(on_value) * (1 - on_prob) / (max_seq_len - 1))
    off_value_label = tf.math.log(tf.math.exp(on_value) * (1 - on_prob) / (43))
    if extreme_value:
      on_value = constants.VERY_LARGE
      off_value = constants.VERY_SMALL
      off_value_label = constants.VERY_SMALL

    heads = tf.one_hot(heads, tf.shape(heads)[-1], off_value=off_value, on_value=on_value)
    # heads = tf.Print(heads, [heads], "heads", summarize=60)

    heads = heads + token_mask_row
    if new_masking:
      heads = heads + token_mask_col
    if parse_label is not None:
      # if label_replacing:
      #   labels = label_replacing_fn(tf.cast(labels, dtype=tf.int32), 44, label_replacing_rate)
      labels = tf.one_hot(labels, num_labels, off_value=off_value_label, on_value=on_value)
      masked_heads = head_gen_fn_train(heads, labels, num_labels, chain=chain,
                                       tokens_to_keep=tokens_to_keep)
      if not new_masking:
        masked_heads = [_ + token_mask_col for _ in masked_heads]
      # if head_dropout:
      #   masked_heads = [adding_head_noise(_, dropout_rate=head_dropout_rate) for _ in masked_heads]
    else:
      masked_heads = tf.nn.log_softmax(heads)
      if not new_masking:
        heads += token_mask_col
      if head_dropout:
        masked_heads = adding_head_noise(heads, dropout_rate=head_dropout_rate)

    set_print_result = False
  else:

    heads = heads + token_mask_row
    if new_masking:
      heads = heads + token_mask_col
    if parse_label is not None:
      masked_heads = head_gen_fn_eval(heads,
                                      labels,
                                      num_labels, chain=chain, tokens_to_keep=tokens_to_keep)
      if not new_masking:
        masked_heads = [_ + token_mask_col for _ in masked_heads]
    else:
      masked_heads = tf.nn.log_softmax(heads)
      if not new_masking:
        masked_heads += token_mask_col

  tf.logging.log(tf.logging.INFO, "label masked heads {}".format(masked_heads))
  head_adjacent_mtx = multiplication_v2(masked_heads, chain)
  tf.logging.log(tf.logging.INFO, "root masked heads {}".format(head_adjacent_mtx))
  if len(direction_seq) > 1:
    tf.logging.log(tf.logging.INFO, "normalize transition probability > 1 to 1")
    head_adjacent_mtx = normalize_prob_gt_1(head_adjacent_mtx)
  result = head_adjacent_mtx
  return result




  # # todo: accomodate different down strategies
  # if len(heads.get_shape()) < 3:
  #   # if using_log_prob:
  #   if head_replacing:
  #     heads = head_replacing_fn(heads, tokens_to_keep, head_replacing_rate)
  #   max_seq_len = tf.cast(tf.shape(heads)[-1], dtype=tf.float32)
  #   # off_value = tf.math.log(tf.math.exp(1e-1)*0.1/0.9/max_seq_len)
  #   on_value = 1e-1
  #   off_value = tf.math.log(tf.math.exp(on_value) * (1-on_prob) / (max_seq_len-1))
  #   heads = tf.one_hot(heads, tf.shape(heads)[-1], off_value=off_value, on_value=on_value)
  #   masked_heads = heads
  #   labels = tf.one_hot(labels, num_labels, off_value=off_value, on_value=on_value)
  #   if head_dropout:
  #     masked_heads = adding_head_noise(masked_heads, dropout_rate=head_dropout_rate, off_value=off_value)
  #   masked_heads = masked_heads + token_mask # necessary as both out-of-range token & in-range tokens would be slightly open
  #   labeled_heads = head_gen_fn_train(tf.nn.log_softmax(masked_heads), tf.nn.log_softmax(labels), num_labels, chain=chain, tokens_to_keep=tokens_to_keep)
  # else:
  #   if allow_intermediate_nodes:
  #     masked_heads = heads + token_mask
  #     labeled_heads = head_gen_fn_eval(tf.nn.log_softmax(masked_heads),
  #                                      tf.nn.log_softmax(labels),
  #                                      num_labels, chain=chain, tokens_to_keep=tokens_to_keep)
  #   else:
  #     tf.logging.log(tf.logging.ERROR, "Not supporting distribution yet")
  #     raise NotImplementedError
  # head_adjacent_mtx = multiplication_v2(labeled_heads, chain)
  # result = head_adjacent_mtx
  # if selective_gating:
  #   result = nn_utils.selective_gating(result)
  #   return result
  # else:
  #   return result

# Simple up, with residual prob concentrating on root, using ROOT scheme
def get_labeled_up_mtx(parse_gold, parse_label, chain, num_labels, tokens_to_keep, smoothing_softmax = False,
                             head_gen_fn_train=head_gen_library["labeled_div"],
                             head_gen_fn_eval=head_gen_library["labeled_div"],
                             ignore_recurrent_head = True, normalization = "none", using_log_prob = True, allow_intermediate_nodes=False, on_prob = 9e-1,
                             selective_gating = False, head_dropout = False, head_dropout_rate =1., head_replacing = False, head_replacing_rate = 1.0, prod_mode = "eye-end_node-noop", ls_multiplier = 1.,
                             label_replacing=False, label_replacing_rate = 1., new_masking = False, extreme_value = False):
  """
    heads: head-dependent distribution of shape (B, seq_len, seq_len)
    labels: label distribution for each head-dependent choice, shape: (B, seq_len, labels)
  """
  heads = parse_gold
  labels = parse_label
  token_mask_row = tf.expand_dims(tf.cast(1 - tokens_to_keep, tf.float32) * constants.VERY_SMALL, 1)

  tf.logging.log(tf.logging.INFO, "Using simple up direction, concentrating non-head prob to ROOT")

  if len(heads.get_shape()) < 3:
    if extreme_value:
      on_value = constants.VERY_LARGE
      off_value = constants.VERY_SMALL
      off_value_label = constants.VERY_SMALL
    else:
      on_value = -1
      off_value = -10
      off_value_label = -10

    heads = tf.one_hot(heads, tf.shape(heads)[-1], off_value=off_value, on_value=on_value)
    labels = tf.one_hot(labels, num_labels, off_value=off_value_label, on_value=on_value)
    heads = heads + token_mask_row

  else:

    heads = heads + token_mask_row
    heads = tf.concat([heads[:, :1, :] + 100, heads[:, 1:, :]], axis=1)


  masked_heads = head_gen_fn_train(heads, labels, num_labels, chain=['u1'],
                                     tokens_to_keep=tokens_to_keep)[0]
  hold_prob = tf.reduce_sum(tf.math.exp(masked_heads), -1, keep_dims=True)
  to_fill = tf.concat([tf.log(1-hold_prob),
                         tf.fill([tf.shape(heads)[0], tf.shape(heads)[1], tf.shape(heads)[2] - 1], constants.VERY_SMALL)
                         ], axis=-1)
  masked_heads = tf.reduce_logsumexp(tf.stack([masked_heads, to_fill], axis=-1), axis=-1)
  masked_heads = tf.Print(masked_heads, [tf.nn.softmax(masked_heads[:, 1:])], "masked_head", summarize=40)
  return masked_heads
# Simple up, with prob void, using vallina shceme
def get_labeled_een_up_mtx(parse_gold, parse_label, chain, num_labels, tokens_to_keep, smoothing_softmax = False,
                             head_gen_fn_train=head_gen_library["labeled_div"],
                             head_gen_fn_eval=head_gen_library["labeled_div"],
                             ignore_recurrent_head = True, normalization = "none", using_log_prob = True, allow_intermediate_nodes=False, on_prob = 9e-1,
                             selective_gating = False, head_dropout = False, head_dropout_rate =1., head_replacing = False, head_replacing_rate = 1.0, prod_mode = "eye-end_node-noop", ls_multiplier = 1.,
                             label_replacing=False, label_replacing_rate = 1., new_masking = False, extreme_value = False):
  """
    heads: head-dependent distribution of shape (B, seq_len, seq_len)
    labels: label distribution for each head-dependent choice, shape: (B, seq_len, labels)
  """
  heads = parse_gold
  labels = parse_label
  token_mask_row = tf.expand_dims(tf.cast(1 - tokens_to_keep, tf.float32) * constants.VERY_SMALL, 1)
  token_mask_col = tf.expand_dims(tf.cast(1 - tokens_to_keep, tf.float32) * constants.VERY_SMALL, 2)

  tf.logging.log(tf.logging.INFO, "Using labeled k-adjacency matrix gen with chain {}".format(chain))
  tf.logging.log(tf.logging.INFO, "Using head_gen_fn_train {}".format(head_gen_fn_train))
  tf.logging.log(tf.logging.INFO, "Using head_gen_fn_eval {}".format(head_gen_fn_eval))
  tf.logging.log(tf.logging.INFO, "Using extreme_value {} ".format(extreme_value))
  tf.logging.log(tf.logging.INFO, "Using new masking mechanism {} ".format(new_masking))
  tf.logging.log(tf.logging.INFO, "prod mode {}".format(prod_mode))
  tf.logging.log(tf.logging.INFO, "Head replacing mode {} with rate {}".format(head_replacing, head_replacing_rate))
  tf.logging.log(tf.logging.INFO, "Label replacing mode {} with rate {}".format(label_replacing, label_replacing_rate))

  tf.logging.log(tf.logging.INFO, "Head dropping out mode {} with rate {}".format(head_dropout, head_dropout_rate))
  tf.logging.log(tf.logging.INFO, "Receiving head/labels {}/{}".format(heads, labels))

  direction_seq = []
  for chain_item in chain:
    direction = chain_item[0]
    step = int(chain_item[1])
    direction_seq += [True if direction == "u" else False for _ in range(step)]

  def multiplication_v2(head, chain):
    print("head, direction_seq: ", head, direction_seq)

    def normalize_prob_gt_1(mtx):
      prob_mass = tf.reduce_logsumexp(mtx, -1, keep_dims=True)
      ind = tf.cast(tf.greater(prob_mass, 0.), dtype=tf.float32)
      # ind = tf.Print(ind, [prob_mass, ind], "prob mass & gt1 indicator")
      mtx -= ind * prob_mass
      mtx = tf.Print(mtx, [prob_mass, ind, tf.reduce_logsumexp(mtx, -1, keep_dims=True)], "prob mass & gt1 indicator")
      return mtx

    def update_activation(next_step, current_activation):
      return tf.math.maximum(next_step, current_activation)

    init_activation_mode, output_mode, init_intermediate_mode = prod_mode.split("-")

    def masking(activation, next_step):
      small_value = tf.ones(tf.shape(next_step), dtype=tf.float32) * constants.VERY_SMALL
      # small_value = -1e2
      return tf.where(tf.greater(activation, next_step), small_value, next_step)

    def masking_self_loop(next_step):
      small_value = tf.eye(tf.shape(next_step)[-1], batch_shape=[tf.shape(next_step)[0]]) * constants.VERY_SMALL
      return next_step + small_value

    def prod(l, r, activation_map):
      l = tf.expand_dims(l, 2)
      r = tf.expand_dims(tf.transpose(r, perm=[0, 2, 1]), dim=1)
      tmp = tf.math.reduce_logsumexp(l + r, axis=-1)

      return masking(activation_map, tmp)

    if len(direction_seq) == 1 and direction_seq[0] and output_mode == "end_node" and init_activation_mode == "noop":
      tf.logging.log(tf.logging.INFO,
                     "Using shortcut for NEN, returning the first element {} of {}".format(head[0], head))
      return head[0]
    if len(direction_seq) == 1 and direction_seq[0] and output_mode == "end_node" and init_activation_mode == "eye":
      tf.logging.log(tf.logging.INFO,
                     "Using shortcut for EEN, returning the first element {} of {}".format(head[0], head))
      return masking_self_loop(head[0])
    raise NotImplementedError

  if len(heads.get_shape()) < 3:
    # if head_replacing:
    #   heads = head_replacing_fn(tf.cast(heads, dtype=tf.int32), tokens_to_keep, head_replacing_rate)
    max_seq_len = tf.cast(tf.shape(heads)[-1], dtype=tf.float32)
    on_value = -1e-1
    off_value = tf.math.log(tf.math.exp(on_value) * (1 - on_prob) / (max_seq_len - 1))
    off_value_label = tf.math.log(tf.math.exp(on_value) * (1 - on_prob) / (43))
    if extreme_value:
      on_value = constants.VERY_LARGE
      off_value = constants.VERY_SMALL
      off_value_label = constants.VERY_SMALL

    heads = tf.one_hot(heads, tf.shape(heads)[-1], off_value=off_value, on_value=on_value)
    # heads = tf.Print(heads, [heads], "heads", summarize=60)

    heads = heads + token_mask_row
    if new_masking:
      heads = heads + token_mask_col
    if parse_label is not None:
      # if label_replacing:
      #   labels = label_replacing_fn(tf.cast(labels, dtype=tf.int32), 44, label_replacing_rate)
      labels = tf.one_hot(labels, num_labels, off_value=off_value_label, on_value=on_value)
      masked_heads = head_gen_fn_train(heads, labels, num_labels, chain=chain,
                                       tokens_to_keep=tokens_to_keep)
      if not new_masking:
        masked_heads = [_ + token_mask_col for _ in masked_heads]
      # if head_dropout:
      #   masked_heads = [adding_head_noise(_, dropout_rate=head_dropout_rate) for _ in masked_heads]
    else:
      masked_heads = tf.nn.log_softmax(heads)
      if not new_masking:
        heads += token_mask_col
      if head_dropout:
        masked_heads = adding_head_noise(heads, dropout_rate=head_dropout_rate)

    set_print_result = False
  else:

    heads = heads + token_mask_row
    if new_masking:
      heads = heads + token_mask_col
    if parse_label is not None:
      masked_heads = head_gen_fn_eval(heads,
                                      labels,
                                      num_labels, chain=chain, tokens_to_keep=tokens_to_keep)
      if not new_masking:
        masked_heads = [_ + token_mask_col for _ in masked_heads]
    else:
      masked_heads = tf.nn.log_softmax(heads)
      if not new_masking:
        masked_heads += token_mask_col

  tf.logging.log(tf.logging.INFO, "label masked heads {}".format(masked_heads))
  head_adjacent_mtx = multiplication_v2(masked_heads, chain)
  tf.logging.log(tf.logging.INFO, "root masked heads {}".format(head_adjacent_mtx))
  if len(direction_seq) > 1:
    tf.logging.log(tf.logging.INFO, "normalize transition probability > 1 to 1")
    head_adjacent_mtx = normalize_prob_gt_1(head_adjacent_mtx)
  result = head_adjacent_mtx
  return result


# Simple up, with prob void, using ROOT shceme
def get_dist_labeled_up_mtx(parse_gold, parse_label, chain, num_labels, tokens_to_keep, smoothing_softmax = False,
                             head_gen_fn_train=head_gen_library["labeled_div"],
                             head_gen_fn_eval=head_gen_library["labeled_div"],
                             ignore_recurrent_head = True, normalization = "none", using_log_prob = True, allow_intermediate_nodes=False, on_prob = 9e-1,
                             selective_gating = False, head_dropout = False, head_dropout_rate =1., head_replacing = False, head_replacing_rate = 1.0, prod_mode = "eye-end_node-noop", ls_multiplier = 1.,
                             label_replacing=False, label_replacing_rate = 1., new_masking = False, extreme_value = False):
  """
    heads: head-dependent distribution of shape (B, seq_len, seq_len)
    labels: label distribution for each head-dependent choice, shape: (B, seq_len, labels)
  """
  heads = parse_gold
  labels = parse_label
  # token_mask = tf.expand_dims(tf.cast(1 - tokens_to_keep, tf.float32) * constants.VERY_SMALL, 1)
  token_mask_row = tf.expand_dims(tf.cast(1 - tokens_to_keep, tf.float32) * constants.VERY_SMALL, 1)
  token_mask_col = tf.expand_dims(tf.cast(1 - tokens_to_keep, tf.float32) * constants.VERY_SMALL, 2)

  tf.logging.log(tf.logging.INFO, "Using simple up direction, removing all prob to ROOT")

  if len(heads.get_shape()) < 3:
    if extreme_value:
      on_value = constants.VERY_LARGE
      off_value = constants.VERY_SMALL
      off_value_label = constants.VERY_SMALL
    else:
      on_value = -1
      off_value = -10
      off_value_label = -10

    heads = tf.one_hot(heads, tf.shape(heads)[-1], off_value=off_value, on_value=on_value)
    labels = tf.one_hot(labels, num_labels, off_value=off_value_label, on_value=on_value)
    heads = heads + token_mask_row

  else:
    heads = heads + token_mask_row
    heads = tf.concat([heads[:, :1, :] + 100, heads[:, 1:, :]], axis=1)


  masked_heads = head_gen_fn_train(heads, labels, num_labels, chain=['u1'],
                                     tokens_to_keep=tokens_to_keep)[0]
  root_mask = tf.concat([tf.fill([tf.shape(heads)[0], tf.shape(heads)[1], 1], constants.VERY_SMALL),
                         tf.zeros([tf.shape(heads)[0], tf.shape(heads)[1], tf.shape(heads)[2] - 1], dtype=tf.float32)
                         ], axis=-1)
  masked_heads = masked_heads + root_mask
  return masked_heads
# Simple up, with prob void, using ROOT shceme
def get_PreColMaskDist_labeled_up_mtx(parse_gold, parse_label, chain, num_labels, tokens_to_keep, smoothing_softmax = False,
                             head_gen_fn_train=head_gen_library["labeled_div"],
                             head_gen_fn_eval=head_gen_library["labeled_div"],
                             ignore_recurrent_head = True, normalization = "none", using_log_prob = True, allow_intermediate_nodes=False, on_prob = 9e-1,
                             selective_gating = False, head_dropout = False, head_dropout_rate =1., head_replacing = False, head_replacing_rate = 1.0, prod_mode = "eye-end_node-noop", ls_multiplier = 1.,
                             label_replacing=False, label_replacing_rate = 1., new_masking = False, extreme_value = False):
  """
    heads: head-dependent distribution of shape (B, seq_len, seq_len)
    labels: label distribution for each head-dependent choice, shape: (B, seq_len, labels)
  """
  heads = parse_gold
  labels = parse_label
  # token_mask = tf.expand_dims(tf.cast(1 - tokens_to_keep, tf.float32) * constants.VERY_SMALL, 1)
  token_mask_row = tf.expand_dims(tf.cast(1 - tokens_to_keep, tf.float32) * constants.VERY_SMALL, 1)
  token_mask_col = tf.expand_dims(tf.cast(1 - tokens_to_keep, tf.float32) * constants.VERY_SMALL, 2)

  tf.logging.log(tf.logging.INFO, "Using simple up direction, removing all prob to ROOT")

  if len(heads.get_shape()) < 3:
    if extreme_value:
      on_value = constants.VERY_LARGE
      off_value = constants.VERY_SMALL
      off_value_label = constants.VERY_SMALL
    else:
      on_value = -1
      off_value = -10
      off_value_label = -10

    heads = tf.one_hot(heads, tf.shape(heads)[-1], off_value=off_value, on_value=on_value)
    labels = tf.one_hot(labels, num_labels, off_value=off_value_label, on_value=on_value)
    heads = heads + token_mask_row
    heads += token_mask_col

  else:

    heads = heads + token_mask_row
    heads = tf.concat([heads[:, :1, :] + 100, heads[:, 1:, :]], axis=1)
    heads += token_mask_col


  masked_heads = head_gen_fn_train(heads, labels, num_labels, chain=['u1'],
                                     tokens_to_keep=tokens_to_keep)[0]
  root_mask = tf.concat([tf.fill([tf.shape(heads)[0], tf.shape(heads)[1], 1], constants.VERY_SMALL),
                         tf.zeros([tf.shape(heads)[0], tf.shape(heads)[1], tf.shape(heads)[2] - 1], dtype=tf.float32)
                         ], axis=-1)
  masked_heads = masked_heads + root_mask
  return masked_heads

def get_een_kup1down_mtx(parse_gold, parse_label, chain, num_labels, tokens_to_keep, smoothing_softmax = False,
                             head_gen_fn_train=head_gen_library["labeled_div"],
                             head_gen_fn_eval=head_gen_library["labeled_div"],
                             ignore_recurrent_head = True, normalization = "none", using_log_prob = True, allow_intermediate_nodes=False, on_prob = 9e-1,
                             selective_gating = False, head_dropout = False, head_dropout_rate =1., head_replacing = False, head_replacing_rate = 1.0, prod_mode = "eye-end_node-noop", ls_multiplier = 1.,
                             label_replacing=False, label_replacing_rate = 1., new_masking = False, extreme_value = False):
  """
    heads: head-dependent distribution of shape (B, seq_len, seq_len)
    labels: label distribution for each head-dependent choice, shape: (B, seq_len, labels)
  """
  heads = parse_gold
  labels = parse_label
  length = int(chain[0])
  # token_mask = tf.expand_dims(tf.cast(1 - tokens_to_keep, tf.float32) * constants.VERY_SMALL, 1)
  token_mask_row = tf.expand_dims(tf.cast(1 - tokens_to_keep, tf.float32) * constants.VERY_SMALL, 1)
  token_mask_col = tf.expand_dims(tf.cast(1 - tokens_to_keep, tf.float32) * constants.VERY_SMALL, 2)

  tf.logging.log(tf.logging.INFO, "Using simple {}up1down direction, removing all prob to ROOT".format(chain))

  def prod(l, r):
    l = tf.expand_dims(l, 2)
    r = tf.expand_dims(tf.transpose(r, perm=[0, 2, 1]), dim=1)
    tmp = tf.math.reduce_logsumexp(l + r, axis=-1)
    return tmp


  if len(heads.get_shape()) < 3:
    # max_seq_len = tf.cast(tf.shape(heads)[-1], dtype=tf.float32)

    if extreme_value:
      on_value = constants.VERY_LARGE
      off_value = constants.VERY_SMALL
      off_value_label = constants.VERY_SMALL
    else:
      on_value = -1
      off_value = -10
      off_value_label = -10

    heads = tf.one_hot(heads, tf.shape(heads)[-1], off_value=off_value, on_value=on_value)
    labels = tf.one_hot(labels, num_labels, off_value=off_value_label, on_value=on_value)
    heads = heads + token_mask_row
  else:

    heads = heads + token_mask_row


  masked_heads = head_gen_fn_train(heads, labels, num_labels, chain=['u{}'.format(length+1)],
                                     tokens_to_keep=tokens_to_keep)
  print("masked_heads, ", masked_heads)
  masked_heads = [masked_heads[idx] +token_mask_col for idx in range(length+1)]
  print("masked_heads, ", masked_heads)
  masked_heads = [masked_heads[idx] +token_mask_col for idx in range(length+1)]
  # masked_heads = tf.map_fn(lambda mtx: mtx+token_mask_col, masked_heads, dtype=tf.float32)[0]
  # First, we make a up/down matrix, then remove prob on root node
  up, down = masked_heads[-2:]
  updown = prod(up, tf.transpose(down, perm=[0, 2, 1])) + tf.eye(tf.shape(heads)[1], batch_shape=[tf.shape(heads)[0]]) * constants.VERY_SMALL
  tmp = updown
  for idx in reversed(range(length-1)):
    tmp = prod(masked_heads[idx], tmp)
  tmp = tf.Print(tmp, [tmp], "merged masked_head")
  return tmp

# Simple up, for creating prior bias based on dependency graph
def get_lprior_up_mtx(parse_gold, parse_label, chain, num_labels, tokens_to_keep, smoothing_softmax = False,
                             head_gen_fn_train=head_gen_library["labeled_div"],
                             head_gen_fn_eval=head_gen_library["labeled_div"],
                             ignore_recurrent_head = True, normalization = "none", using_log_prob = True, allow_intermediate_nodes=False, on_prob = 9e-1,
                             selective_gating = False, head_dropout = False, head_dropout_rate =1., head_replacing = False, head_replacing_rate = 1.0, prod_mode = "eye-end_node-noop", ls_multiplier = 1.,
                             label_replacing=False, label_replacing_rate = 1., new_masking = False, extreme_value = False):
  """
    heads: head-dependent distribution of shape (B, seq_len, seq_len)
    labels: label distribution for each head-dependent choice, shape: (B, seq_len, labels)
  """
  heads = parse_gold
  labels = parse_label
  # assert  head_gen_fn_train == nn_utils.generating_prior_mtx_for_srl
  token_mask_row = tf.expand_dims(tf.cast(1 - tokens_to_keep, tf.float32) * constants.VERY_SMALL, 1)

  tf.logging.log(tf.logging.INFO, "Using simple up direction, concentrating non-head prob to ROOT")

  if len(heads.get_shape()) < 3:
    if extreme_value:
      on_value = constants.VERY_LARGE
      off_value = constants.VERY_SMALL
      off_value_label = constants.VERY_SMALL
    else:
      on_value = -1
      off_value = -10
      off_value_label = -10

    heads = tf.one_hot(heads, tf.shape(heads)[-1], off_value=off_value, on_value=on_value)
    labels = tf.one_hot(labels, num_labels, off_value=off_value_label, on_value=on_value)
    heads = heads + token_mask_row

  else:

    heads = heads + token_mask_row
    # heads = tf.concat([heads[:, :1, :] + 100, heads[:, 1:, :]], axis=1)


  masked_heads = nn_utils.generating_prior_mtx_for_srl(heads, labels, num_labels, chain=['u1'],
                                     tokens_to_keep=tokens_to_keep)[0]
  masked_heads = tf.Print(masked_heads, [masked_heads], "priors inferred from dependency graph", summarize=40)
  return masked_heads



def get_adjacent_mtx_2(parse_gold, chain, tokens_to_keep, use_hard_decedent=False, ignore_recurrent_head = True, normalization = "softmax", using_log_prob = False, allow_intermediate_nodes=False, head_dropout = 0.1):
  """
    heads: head-dependent distribution of shape (B, seq_len, seq_len)
    labels: label distribution for each head-dependent choice, shape: (B, seq_len, seq_len, labels)
  """
  heads = parse_gold
  tf.logging.log(tf.logging.INFO, "Using unlabeled k-adjacency matrix gen with chain {}".format(chain))

  def multiplication_v2(head, chain):
    def prod(l, r, activation_map):
      l = tf.expand_dims(l, 2)
      r = tf.expand_dims(tf.transpose(r, perm=[0, 2, 1]), dim=1)
      tmp = tf.math.reduce_logsumexp(l + r, axis=-1)
      activated_mask = tf.math.sigmoid(activation_map + 1e5) * constants.VERY_SMALL
      tmp += activated_mask
      return tmp

    # activation_map = tf.ones_like(head) * constants.VERY_SMALL
    if len(chain) == 1 and chain[0] == "u1":
      result = head
      return result
    else: #length of chain > 1
      activation_map = tf.ones_like(head) * constants.VERY_SMALL
      no_op_map = tf.ones_like(activation_map) * constants.VERY_SMALL
      self_recurrent_map = (1 - tf.eye(tf.shape(heads)[-1], batch_shape=[tf.shape(heads)[0]])) * constants.VERY_SMALL
      activation_map = tf.reduce_logsumexp(tf.stack([activation_map, self_recurrent_map], -1), -1)
      self_recurrent_activated_mask = tf.math.sigmoid(activation_map + 1e5) * constants.VERY_SMALL
      head_without_recurrent = head + self_recurrent_activated_mask
      # True: upward, False: Downward
      direction_seq = []
      for chain_item in chain:
        direction = chain_item[0]
        step = int(chain_item[1])
        direction_seq += [True if direction == "u" else False for _ in range(step)]
      # print(direction_seq)
      direction_cond = tf.constant(direction_seq)
      materials = tf.math.log(tf.eye(tf.shape(heads)[-1], batch_shape=[tf.shape(heads)[0]]))
      int_materials = tf.math.log(tf.eye(tf.shape(heads)[-1], batch_shape=[tf.shape(heads)[0]]))
      loop_cond = lambda materials, int_materials, activation_map, direction_cond, step: tf.less(step, len(direction_seq))
      loop_cond_no_restriction = lambda materials, direction_cond, step: tf.less(step, len(direction_seq))
      def loop_body_no_restriction(materials, direction_cond, step):
        next_step = tf.cond(direction_cond[step],
                            lambda: prod(materials, head, no_op_map),
                            lambda: prod(materials, tf.transpose(head, perm=[0, 2, 1]),
                                         no_op_map),
                            )
        # print("next step", next_step)
        return next_step, direction_cond, step+1
      def loop_body(materials, int_materials, activation_map, direction_cond, step):
        next_step = tf.cond(direction_cond[step],
                lambda: prod(materials, head_without_recurrent, activation_map),
                lambda: prod(materials, tf.transpose(head_without_recurrent, perm=[0, 2, 1]), activation_map),
                )
        # next_step = tf.Print(next_step, [next_step], summarize=30)
        int_materials = tf.reduce_logsumexp(tf.stack([next_step, int_materials], -1), -1)
        # materials.append(next_step)
        materials = next_step
        activation_map_next_step = tf.reduce_logsumexp(tf.stack([next_step, activation_map], -1), -1)
        return next_step, int_materials, activation_map_next_step, direction_cond, step+1

      # loop_vars = tf.Print(loop_vars, [materials], "start_looping")
      if allow_intermediate_nodes:
        loop_vars = [materials, direction_cond, 0]
        materials, _, _ = tf.while_loop(loop_cond_no_restriction, loop_body_no_restriction, loop_vars)
        return materials
      else:
        loop_vars = [materials, int_materials, activation_map, direction_cond, 0]
        materials, int_materials, activation_map, _, _ = tf.while_loop(loop_cond, loop_body, loop_vars)
        return materials


  if len(heads.get_shape()) < 3:
    # if using_log_prob:
    heads = tf.one_hot(heads, tf.shape(heads)[-1], off_value=constants.VERY_SMALL, on_value=-1e-6)
    masked_heads = heads
    masked_heads = masked_heads

  else:
    if allow_intermediate_nodes:
      mask = tf.cast(1 - tokens_to_keep, tf.float32) * constants.VERY_SMALL
      masked_heads = heads + tf.expand_dims(mask, 1)
      masked_heads = tf.nn.log_softmax(masked_heads) # would it be safe to cap at 0?
      # masked_heads = heads
    else:
      tf.logging.log(tf.logging.ERROR, "Not supporting distribution yet at intermediate node mode yet")
      raise NotImplementedError


  result = multiplication_v2(masked_heads, chain)
  head_adjacent_mtx = result
  result = head_adjacent_mtx
  # print("result", result)
  # adjacent_mtx_normalization(result, normalization)
  mask = tf.cast(1 - tokens_to_keep, tf.float32) * constants.VERY_SMALL
  result = result + tf.expand_dims(mask, 1)
  # result = result + root_booster
  # result_root = tf.concat(
  #   [tf.clip_by_value(result[:, :1, :1], clip_value_min=constants.VERY_SMALL, clip_value_max=constants.VERY_LARGE),
  #   tf.clip_by_value(result[:, 1:, :1], clip_value_min=-1e7, clip_value_max=constants.VERY_LARGE)], axis =1)
  # result_cont = tf.clip_by_value(result[:, :, 1:], clip_value_min=constants.VERY_SMALL, clip_value_max=constants.VERY_LARGE)
  # result = tf.concat([result_root, result_cont], axis=-1)

  return result

# todo implement direction-specific dependency e.g. <up, up, down>


def one_hot(heads):
  return tf.one_hot(heads, tf.shape(heads)[-1], on_value=constants.VERY_LARGE,
             off_value=constants.VERY_SMALL)
def local_window_balanced(input, strip_width):
  strip_width = int(strip_width)
  # tf.roll()
  diag = tf.reduce_sum([roll(tf.linalg.diag(tf.where(tf.not_equal(input, constants.PAD_VALUE), tf.ones_like(input), tf.zeros_like(input))), shift=k) for k in range(-strip_width, strip_width+1)], axis=0)
  diag = tf.where(tf.greater(diag, 0), constants.VERY_LARGE * tf.ones_like(diag, dtype=tf.float32),
                  constants.VERY_SMALL * tf.ones_like(diag, dtype=tf.float32))
  return tf.cast(diag, tf.float32)

def local_window_rtilted(input, strip_width):
  strip_width = int(strip_width)
  diag = tf.reduce_sum([roll(tf.linalg.diag(tf.where(tf.not_equal(input, constants.PAD_VALUE), tf.ones_like(input), tf.zeros_like(input))), shift=k) for k in range(-(strip_width-1), strip_width+1)], axis=0)
  diag = tf.where(tf.greater(diag, 0), constants.VERY_LARGE * tf.ones_like(diag, dtype=tf.float32),
                  constants.VERY_SMALL * tf.ones_like(diag, dtype=tf.float32))
  return tf.cast(diag, tf.float32)

def local_window_ltilted(input, strip_width):
  strip_width = int(strip_width)
  diag = tf.reduce_sum([roll(tf.linalg.diag(tf.where(tf.not_equal(input, constants.PAD_VALUE), tf.ones_like(input), tf.zeros_like(input))), shift=k) for k in range(-strip_width, strip_width)], axis=0)
  diag = tf.where(tf.greater(diag, 0), constants.VERY_LARGE * tf.ones_like(diag, dtype=tf.float32), constants.VERY_SMALL * tf.ones_like(diag, dtype=tf.float32))
  return tf.cast(diag, tf.float32)

def gen_block_by_line(idx, len, size, offset):
  array_idx = tf.range(len, dtype=tf.int32)
  array_location =tf.math.logical_and(tf.greater_equal(array_idx, idx-offset), tf.less(array_idx, idx+size-offset))
  # line = tf.where(array_location, tf.ones_like(array_idx, dtype=tf.float32),
  #                 tf.zeros_like(array_idx, dtype=tf.float32))
  line = tf.where(array_location, constants.VERY_LARGE * tf.ones_like(array_idx, dtype=tf.float32), constants.VERY_SMALL * tf.ones_like(array_idx, dtype=tf.float32))
  # line = tf.Print(line, [line], "line")
  return line
def gen_block_by_instance(input, idxer):
  seq_len = input.get_shape()[0]
  size = tf.cast(input / 12, dtype=tf.int32)
  offset = tf.cast(input % 12, dtype=tf.int32)
  mtx = tf.map_fn(lambda inp: gen_block_by_line(idx = inp[0], len = seq_len, size = inp[1], offset = inp[2]), (idxer, size, offset), dtype=tf.float32)
  return tf.cast(mtx, tf.float32)
def chunk_to_block_diag(input):
  seq_len = input.get_shape()[1]
  idxer = tf.range(seq_len, dtype=tf.int32)
  batch = tf.map_fn(lambda inp: gen_block_by_instance(inp, idxer), elems=input, dtype=tf.float32)
  return tf.cast(batch, tf.float32)



dispatcher = {
  'one_hot': one_hot,
  'local_window_balanced': local_window_balanced,
  'local_window_ltilted': local_window_ltilted,
  'local_window_rtilted': local_window_rtilted,
  'chunk_to_block_diag': chunk_to_block_diag,
  'get_labeled_adjacent_mtx': get_labeled_adjacent_mtx,
  'get_root_up_mtx': get_labeled_up_mtx,
  'get_een_up_mtx': get_labeled_een_up_mtx,
  'get_void_up_mtx': get_dist_labeled_up_mtx,
  'get_PreColMaskVoid_up_mtx': get_PreColMaskDist_labeled_up_mtx,
  'get_een_kup1down_mtx': get_een_kup1down_mtx,
  'get_labeled_adjacent_mtx_softmax_log_prob': partial(get_labeled_adjacent_mtx,
                                          head_gen_fn_train=head_gen_library["labeled_div"],
                                          head_gen_fn_eval=head_gen_library["labeled_div"],
                                                       using_log_prob=True, normalization="softmax"
                                          ),
  'get_labeled_adjacent_mtx_mvnorm': partial(get_labeled_adjacent_mtx,
                                             head_gen_fn_train=head_gen_library["labeled_mul"],
                                             head_gen_fn_eval=head_gen_library["labeled_mul"],
                                             normalization="mean_var_norm"),
  'get_labeled_adjacent_mtx_nonenorm': partial(get_labeled_adjacent_mtx,
                                             head_gen_fn_train=head_gen_library["labeled_mul"],
                                             head_gen_fn_eval=head_gen_library["labeled_mul"],
                                             normalization="none"),
  'get_labeled_adjacent_mtx_nonenorm_log_prob': partial(get_labeled_adjacent_mtx,
                                                        head_gen_fn_train=head_gen_library["labeled_div"],
                                                        head_gen_fn_eval=head_gen_library["labeled_div"],
                                                        # decedent_gen_fn_train=decedent_gen_library["labeled_div"],
                                                        # decedent_gen_fn_eval=decedent_gen_library["labeled_div"],
                                             normalization="none", using_log_prob=True),
  'get_hsdp_labeled_adjacent_mtx': partial(get_labeled_adjacent_mtx, decedent_gen_fn_eval=nn_utils.generating_hard_decedent_mtx_from_head_label_dist),
  'get_adjacent_mtx':get_adjacent_mtx_2,
  'get_adjacent_mtx_nonenorm_logprob':partial(get_adjacent_mtx_2, normalization="none", using_log_prob=True),
  'pass_through': lambda x: x}


def dispatch(fn_name):
  try:
    return dispatcher[fn_name]
  except KeyError:
    print('Undefined transformation function %s' % fn_name)
    exit(1)

def get_params(input, transformation_name, tokens_to_keep=None):
  transformation_diag_width_in_name = ['local_window_balanced',
                                 'local_window_ltilted',
                                 'local_window_rtilted',
                                       ]
  transformation_k_dist_in_name = ['get_k_adjacent_mtx_2', 'get_k_adjacent_mtx_1', 'get_k_adjacent_mtx_3']
  transformation_pass_through = ['one_hot', 'get_decedent_mtx', 'pass_through']
  if transformation_name in transformation_pass_through:
    return {'heads': input}
  # elif transformation_name in transformation_diag_width_in_name:
  #   return {'input': input, 'strip_width': src_name.split('_')[-1]}
  elif transformation_name.startswith('get_k_adjacent_mtx_hard'):
    assert  tokens_to_keep is not None
    return {'heads': input, 'chain': transformation_name.split('_')[5:], 'tokens_to_keep': tokens_to_keep}
  elif transformation_name.startswith('get_k_adjacent_mtx'):
    assert  tokens_to_keep is not None
    return {'heads': input, 'chain': transformation_name.split('_')[4:], 'tokens_to_keep': tokens_to_keep}
  elif transformation_name.startswith('get_hsdp_adjacent_mtx'):
    assert tokens_to_keep is not None
    return {'heads': input, 'chain': transformation_name.split('_')[4:], 'tokens_to_keep': tokens_to_keep}
  elif transformation_name.startswith('get_labeled_hsdp_adjacent_mtx'):
    assert tokens_to_keep is not None
    # todo hard coding of num of dep labels
    return {'heads': input, 'chain': transformation_name.split('_')[4:], 'tokens_to_keep': tokens_to_keep, 'num_labels': 44}
  else:
    print('Undefined transformation param format')
    raise NotImplementedError
