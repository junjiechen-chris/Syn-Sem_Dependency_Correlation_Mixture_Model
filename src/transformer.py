import tensorflow as tf
import constants
import nn_utils

'''
Much of this code is adapted from the Tensor2Tensor Transformer implementation:
    https://github.com/tensorflow/tensor2tensor
'''


def add_timing_signal_1d(x, min_timescale=1.0, max_timescale=1.0e4):
  """Adds a bunch of sinusoids of different frequencies to a Tensor.
  Each channel of the input Tensor is incremented by a sinusoid of a different
  frequency and phase.
  This allows attention to learn to use absolute and relative positions.
  Timing signals should be added to some precursors of both the query and the
  memory inputs to attention.
  The use of relative position is possible because sin(x+y) and cos(x+y) can be
  expressed in terms of y, sin(x) and cos(x).
  In particular, we use a geometric sequence of timescales starting with
  min_timescale and ending with max_timescale.  The number of different
  timescales is equal to channels / 2. For each timescale, we
  generate the two sinusoidal signals sin(timestep/timescale) and
  cos(timestep/timescale).  All of these sinusoids are concatenated in
  the channels dimension.
  Args:
    x: a Tensor with shape [batch, length, channels]
    min_timescale: a float
    max_timescale: a float
  Returns:
    a Tensor the same shape as x.
  """
  length = tf.shape(x)[1]
  channels = tf.shape(x)[2]
  position = tf.to_float(tf.range(length))
  num_timescales = channels // 2
  log_timescale_increment = (
      tf.log(float(max_timescale) / float(min_timescale)) /
      (tf.to_float(num_timescales) - 1))
  inv_timescales = min_timescale * tf.exp(
      tf.to_float(tf.range(num_timescales)) * -log_timescale_increment)
  scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(inv_timescales, 0)
  signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
  signal = tf.pad(signal, [[0, 0], [0, tf.mod(channels, 2)]])
  signal = tf.reshape(signal, [1, length, channels])
  return x + signal


def attention_bias_ignore_padding(tokens_to_keep):
  """Create a bias tensor to be added to attention logits.
  Args:
    tokens_to_keep: an int Tensor with shape [batch, batch_seq_len].
  Returns:
    A `Tensor` with shape [batch, 1, 1, batch_seq_len].
  """
  # mask = tf.sequence_mask(lengths, tf.reduce_max(lengths))
  mask = tf.cast(1 - tokens_to_keep, tf.float32) * constants.VERY_SMALL
  return tf.expand_dims(tf.expand_dims(mask, axis=1), axis=1)


def split_last_dimension(x, n):
  """Reshape x so that the last dimension becomes two dimensions.
  The first of these two dimensions is n.
  Args:
    x: a Tensor with shape [..., m]
    n: an integer.
  Returns:
    a Tensor with shape [..., n, m/n]
  """
  old_shape = x.get_shape().dims
  last = old_shape[-1]
  new_shape = old_shape[:-1] + [n] + [last // n if last else None]
  ret = tf.reshape(x, tf.concat([tf.shape(x)[:-1], [n, -1]], 0))
  ret.set_shape(new_shape)
  return ret


def combine_last_two_dimensions(x):
  """Reshape x so that the last two dimension become one.
  Args:
    x: a Tensor with shape [..., a, b]
  Returns:
    a Tensor with shape [..., ab]
  """
  old_shape = x.get_shape().dims
  a, b = old_shape[-2:]
  new_shape = old_shape[:-2] + [a * b if a and b else None]
  ret = tf.reshape(x, tf.concat([tf.shape(x)[:-2], [-1]], 0))
  ret.set_shape(new_shape)
  return ret


def split_heads(x, num_heads):
  """Split channels (dimension 3) into multiple heads (becomes dimension 1).
  Args:
    x: a Tensor with shape [batch, length, channels]
    num_heads: an integer
  Returns:
    a Tensor with shape [batch, num_heads, length, channels / num_heads]
  """
  return tf.transpose(split_last_dimension(x, num_heads), [0, 2, 1, 3])


def combine_heads(x):
  """Inverse of split_heads.
  Args:
    x: a Tensor with shape [batch, num_heads, length, channels / num_heads]
  Returns:
    a Tensor with shape [batch, length, channels]
  """
  return combine_last_two_dimensions(tf.transpose(x, [0, 2, 1, 3]))


def conv_hidden_relu(inputs,
                     hidden_size,
                     output_size,
                     dropout):
  """Hidden layer with RELU activation followed by linear projection."""
  with tf.variable_scope("conv_hidden_relu", [inputs]):
    inputs = tf.expand_dims(inputs, 1)
    in_size = inputs.get_shape().as_list()[-1]
    params1 = tf.get_variable("ff1", [1, 1, in_size, hidden_size])
    params2 = tf.get_variable("ff2", [1, 1, hidden_size, hidden_size])
    params3 = tf.get_variable("ff3", [1, 1, hidden_size, output_size])
    h = tf.nn.conv2d(inputs, params1, [1, 1, 1, 1], "SAME")
    h = nn_utils.leaky_relu(h)
    h = tf.nn.dropout(h, dropout)
    h = tf.nn.conv2d(h, params2, [1, 1, 1, 1], "SAME")
    h = nn_utils.leaky_relu(h)
    h = tf.nn.dropout(h, dropout)
    ret = tf.nn.conv2d(h, params3, [1, 1, 1, 1], "SAME")
    ret = tf.squeeze(ret, 1)
    return ret


def dot_product_attention(q, k, v,
                          bias,
                          special_attention,
                          dropout_rate=1.0):
  """dot-product attention.
  Args:
    q: a Tensor with shape [batch, heads, length_q, depth_k]
    k: a Tensor with shape [batch, heads, length_kv, depth_k]
    v: a Tensor with shape [batch, heads, length_kv, depth_v]
    bias: bias Tensor (see attention_bias())
    dropout_rate: a floating point number
    name: an optional string
  Returns:
    A Tensor.
  """
  with tf.variable_scope("dot_product_attention", values=[q, k, v]):
    special_attention = special_attention[0]
    tf.logging.log(tf.logging.INFO, "Using LISA attention with {} heads".format(len(special_attention)))
    #todo change it ! because things are hard coded
    logits = tf.matmul(q, k, transpose_b=True)
    if special_attention:
      logits = tf.concat([logits] + list(map(lambda x: tf.expand_dims(x, 1), special_attention)), axis=1)
    # logits = tf.Print(logits, [tf.nn.softmax(tf.stack(special_attention, 1)[0, 0])], "LISA special attn", summarize=40)
    # concat special_attention to end of logits
    if bias is not None:
      logits += bias

    weights = tf.nn.softmax(logits, -1)
    weights_drop = tf.nn.dropout(weights, dropout_rate)
    return tf.matmul(weights_drop, v), logits

def discounting_dot_product_attention(q, k, v,
                          bias,
                          special_attentions,
                          dropout_rate=1.0):
  """dot-product attention.
  Args:
    q: a Tensor with shape [batch, heads, length_q, depth_k]
    k: a Tensor with shape [batch, heads, length_kv, depth_k]
    v: a Tensor with shape [batch, heads, length_kv, depth_v]
    discounters: a Tensor with shape [batch, heads, length_q, length_kv]
    bias: bias Tensor (see attention_bias())
    dropout_rate: a floating point number
    name: an optional string
  Returns:
    A Tensor.
  """


  with tf.variable_scope("discounting_dot_product_attention", values=[q, k, v]):
    # [batch, num_heads, query_length, memory_length]
    # todo assure the shape of logit and discounter are equal!
    injections = special_attentions[0]
    discounters = special_attentions[1]

    logits = tf.matmul(q, k, transpose_b=True)

    if injections:
      assert not discounters
      logits = tf.concat([logits] + list(map(lambda x: tf.expand_dims(x, 1), injections)), axis=1)

    if bias is not None:
      logits += bias

    num_attn_to_discount = len(discounters)
    tf.logging.log(tf.logging.INFO, "Entering Transformer with {} discounters and {} injectors".format(num_attn_to_discount, 0))


    if discounters:
      tf.logging.log(tf.logging.INFO, "Performing discounting")
      discounters = tf.stack(discounters, 1)
      masked_logits = logits[:, -num_attn_to_discount:]
      gate = tf.reduce_sum(tf.math.exp(discounters), -1, keep_dims=True)
      discounters = tf.Print(discounters, [discounters], "discounters @ transformer")
      discounters_prob = tf.math.exp(discounters)

      discounted_weights = discounters_prob + (1 - gate) * tf.nn.softmax(masked_logits)
      discounted_weights = tf.Print(discounted_weights, [discounted_weights, gate], "Prob to discount", summarize=10)
      weights = tf.concat([tf.nn.softmax(logits[:, :-num_attn_to_discount]), discounted_weights], axis=1)
    else:
      weights = tf.nn.softmax(logits, -1)

    weights_drop = tf.nn.dropout(weights, dropout_rate)
    # raise NotImplementedError
    return tf.matmul(weights_drop, v), logits

def discounting_dot_product_attention_gt1p(q, k, v,
                          bias,
                          special_attentions,
                          dropout_rate=1.0):
  """dot-product attention.
  Args:
    q: a Tensor with shape [batch, heads, length_q, depth_k]
    k: a Tensor with shape [batch, heads, length_kv, depth_k]
    v: a Tensor with shape [batch, heads, length_kv, depth_v]
    discounters: a Tensor with shape [batch, heads, length_q, length_kv]
    bias: bias Tensor (see attention_bias())
    dropout_rate: a floating point number
    name: an optional string
  Returns:
    A Tensor.
  """


  with tf.variable_scope("discounting_dot_product_attention", values=[q, k, v]):
    # [batch, num_heads, query_length, memory_length]
    # todo assure the shape of logit and discounter are equal!
    injections = special_attentions[0]
    discounters = special_attentions[1]

    logits = tf.matmul(q, k, transpose_b=True)

    if injections:
      assert not discounters
      logits = tf.concat([logits] + list(map(lambda x: tf.expand_dims(x, 1), injections)), axis=1)

    if bias is not None:
      logits += bias

    num_attn_to_discount = len(discounters)
    tf.logging.log(tf.logging.INFO, "Entering Transformer with {} discounters and {} injectors".format(num_attn_to_discount, 0))


    if discounters:
      tf.logging.log(tf.logging.INFO, "Performing discounting")
      discounters = tf.stack(discounters, 1)
      masked_logits = logits[:, -num_attn_to_discount:]
      gate = tf.reduce_sum(tf.math.exp(discounters), -1, keep_dims=True)
      discounters = tf.Print(discounters, [discounters], "discounters @ transformer")
      discounters_prob = tf.math.exp(discounters)

      discounted_weights = discounters_prob + tf.nn.relu((1 - gate)) * tf.nn.softmax(masked_logits)
      discounted_weights = tf.Print(discounted_weights, [discounted_weights, tf.reduce_max(gate)], "Prob to discount", summarize=10)
      weights = tf.concat([tf.nn.softmax(logits[:, :-num_attn_to_discount]), discounted_weights], axis=1)
    else:
      weights = tf.nn.softmax(logits, -1)

    weights_drop = tf.nn.dropout(weights, dropout_rate)
    # raise NotImplementedError
    return tf.matmul(weights_drop, v), logits

def discounting_dot_product_attention_gt1p_norext_val(q, k, v,
                          bias,
                          special_attentions,
                          dropout_rate=1.0):
  """dot-product attention.
  Args:
    q: a Tensor with shape [batch, heads, length_q, depth_k]
    k: a Tensor with shape [batch, heads, length_kv, depth_k]
    v: a Tensor with shape [batch, heads, length_kv, depth_v]
    discounters: a Tensor with shape [batch, heads, length_q, length_kv]
    bias: bias Tensor (see attention_bias())
    dropout_rate: a floating point number
    name: an optional string
  Returns:
    A Tensor.
  """


  with tf.variable_scope("discounting_dot_product_attention", values=[q, k, v]):
    # [batch, num_heads, query_length, memory_length]
    # todo assure the shape of logit and discounter are equal!
    injections = special_attentions[0]
    discounters = special_attentions[1]

    logits = tf.matmul(q, k, transpose_b=True)

    if injections:
      assert not discounters
      logits = tf.concat([logits] + list(map(lambda x: tf.expand_dims(x, 1), injections)), axis=1)

    if bias is not None:
      logits += bias

    num_attn_to_discount = len(discounters)
    tf.logging.log(tf.logging.INFO, "Entering Transformer with {} discounters and {} injectors".format(num_attn_to_discount, 0))


    if discounters:
      tf.logging.log(tf.logging.INFO, "Performing discounting")
      discounters = tf.stack(discounters, 1)
      masked_logits = logits[:, -num_attn_to_discount:]
      gate = tf.reduce_sum(tf.math.exp(discounters), -1, keep_dims=True)
      discounters = tf.Print(discounters, [discounters], "discounters @ transformer")
      discounters_prob = tf.math.exp(discounters)

      discounted_weights = (1/tf.maximum(gate, 1)) * discounters_prob + tf.nn.relu((1 - gate)) * tf.nn.softmax(masked_logits)
      discounted_weights = tf.Print(discounted_weights, [discounted_weights, tf.reduce_max(tf.reduce_sum(discounted_weights, -1))], "Prob to discount", summarize=10)
      weights = tf.concat([tf.nn.softmax(logits[:, :-num_attn_to_discount]), discounted_weights], axis=1)
    else:
      weights = tf.nn.softmax(logits, -1)

    weights_drop = tf.nn.dropout(weights, dropout_rate)
    # raise NotImplementedError
    return tf.matmul(weights_drop, v), logits

def discounting_dot_product_attention_mul(q, k, v,
                          bias,
                          special_attentions,
                          dropout_rate=1.0):
  """dot-product attention.
  Args:
    q: a Tensor with shape [batch, heads, length_q, depth_k]
    k: a Tensor with shape [batch, heads, length_kv, depth_k]
    v: a Tensor with shape [batch, heads, length_kv, depth_v]
    discounters: a Tensor with shape [batch, heads, length_q, length_kv]
    bias: bias Tensor (see attention_bias())
    dropout_rate: a floating point number
    name: an optional string
  Returns:
    A Tensor.
  """


  with tf.variable_scope("discounting_dot_product_attention", values=[q, k, v]):
    # [batch, num_heads, query_length, memory_length]
    # todo assure the shape of logit and discounter are equal!
    injections = special_attentions[0]
    discounters = special_attentions[1]
    assert  not injections
    logits = tf.matmul(q, k, transpose_b=True)

    # if injections:
    #   assert not discounters
    #   logits = tf.concat([logits] + list(map(lambda x: tf.expand_dims(x, 1), injections)), axis=1)

    if bias is not None:
      logits += bias

    num_attn_to_discount = len(discounters)
    tf.logging.log(tf.logging.INFO, "Entering Transformer with {} discounters and {} injectors".format(num_attn_to_discount, 0))


    if discounters:
      tf.logging.log(tf.logging.INFO, "Performing discounting")
      discounters = tf.stack(discounters, 1)
      masked_logits = logits[:, -num_attn_to_discount:]
      gate = tf.reduce_sum(tf.math.exp(discounters), -1, keep_dims=True)

      discounters_prob = tf.nn.log_softmax(gate * (discounters-gate))
      discounters_prob = tf.Print(discounters_prob, [discounters_prob], "discounters_prob @ transformer")
      discounted_weights = tf.nn.softmax(discounters_prob+tf.nn.log_softmax(masked_logits))
      discounted_weights = tf.Print(discounted_weights, [discounted_weights, gate], "Prob to discount", summarize=10)
      weights = tf.concat([tf.nn.softmax(logits[:, :-num_attn_to_discount]), discounted_weights], axis=1)
    else:
      weights = tf.nn.softmax(logits, -1)

    weights_drop = tf.nn.dropout(weights, dropout_rate)
    # raise NotImplementedError
    return tf.matmul(weights_drop, v), logits

def discounting_dot_product_attention_add(q, k, v,
                          bias,
                          special_attentions,
                          dropout_rate=1.0):
  """dot-product attention.
  Args:
    q: a Tensor with shape [batch, heads, length_q, depth_k]
    k: a Tensor with shape [batch, heads, length_kv, depth_k]
    v: a Tensor with shape [batch, heads, length_kv, depth_v]
    discounters: a Tensor with shape [batch, heads, length_q, length_kv]
    bias: bias Tensor (see attention_bias())
    dropout_rate: a floating point number
    name: an optional string
  Returns:
    A Tensor.
  """


  with tf.variable_scope("discounting_dot_product_attention", values=[q, k, v]):
    # [batch, num_heads, query_length, memory_length]
    # todo assure the shape of logit and discounter are equal!
    injections = special_attentions[0]
    discounters = special_attentions[1]
    assert  not injections
    logits = tf.matmul(q, k, transpose_b=True)

    # if injections:
    #   assert not discounters
    #   logits = tf.concat([logits] + list(map(lambda x: tf.expand_dims(x, 1), injections)), axis=1)

    if bias is not None:
      logits += bias

    num_attn_to_discount = len(discounters)
    tf.logging.log(tf.logging.INFO, "Entering Transformer with {} discounters and {} injectors".format(num_attn_to_discount, 0))


    if discounters:
      tf.logging.log(tf.logging.INFO, "Performing discounting")
      discounters = tf.stack(discounters, 1)
      masked_logits = logits[:, -num_attn_to_discount:]
      gate = tf.reduce_sum(tf.math.exp(discounters), -1, keep_dims=True)

      discounters_prob = tf.nn.softmax(gate * (discounters-gate))
      discounters_prob = tf.Print(discounters_prob, [discounters_prob], "discounters_prob @ transformer")
      discounted_weights = (discounters_prob+tf.nn.softmax(masked_logits))/2
      discounted_weights = tf.Print(discounted_weights, [discounted_weights, gate], "Prob to discount", summarize=10)
      weights = tf.concat([tf.nn.softmax(logits[:, :-num_attn_to_discount]), discounted_weights], axis=1)
    else:
      weights = tf.nn.softmax(logits, -1)

    weights_drop = tf.nn.dropout(weights, dropout_rate)
    # raise NotImplementedError
    return tf.matmul(weights_drop, v), logits

def injection_dot_product_attention(q, k, v,
                          bias,
                          special_attentions,
                          dropout_rate=1.0):
  """dot-product attention.
  Args:
    q: a Tensor with shape [batch, heads, length_q, depth_k]
    k: a Tensor with shape [batch, heads, length_kv, depth_k]
    v: a Tensor with shape [batch, heads, length_kv, depth_v]
    discounters: a Tensor with shape [batch, heads, length_q, length_kv]
    bias: bias Tensor (see attention_bias())
    dropout_rate: a floating point number
    name: an optional string
  Returns:
    A Tensor.
  """


  with tf.variable_scope("discounting_dot_product_attention", values=[q, k, v]):
    # [batch, num_heads, query_length, memory_length]
    # todo assure the shape of logit and discounter are equal!
    injections = special_attentions[0]
    discounters = special_attentions[1]
    assert  not discounters
    logits = tf.matmul(q, k, transpose_b=True)
    num_attn_to_inject = len(injections)

    if injections:
      injections = tf.stack(injections, 1)
      gate = tf.reduce_sum(tf.math.exp(injections), -1, keep_dims=True)

      injections_prob = gate * (injections - gate)
      injections_prob = tf.Print(injections_prob, [tf.nn.softmax(injections_prob), gate], "discounters_prob @ transformer", summarize=10)
      logits = tf.concat([logits, injections_prob], axis=1)

    if bias is not None:
      logits += bias


    tf.logging.log(tf.logging.INFO, "Entering Transformer with {} discounters and {} injectors".format(num_attn_to_inject, 0))

    weights = tf.nn.softmax(logits, -1)
    weights_drop = tf.nn.dropout(weights, dropout_rate)
    return tf.matmul(weights_drop, v), logits

def discounting_dot_product_attention_ns(q, k, v,
                          bias,
                          special_attentions,
                          dropout_rate=1.0):

  with tf.variable_scope("discounting_dot_product_attention", values=[q, k, v]):
    # [batch, num_heads, query_length, memory_length]
    # todo assure the shape of logit and discounter are equal!
    injections = special_attentions[0]
    discounters = special_attentions[1]

    logits = tf.matmul(q, k, transpose_b=True)

    if injections:
      assert not discounters
      logits = tf.concat([logits] + list(map(lambda x: tf.expand_dims(x, 1), injections)), axis=1)

    if bias is not None:
      logits += bias

    weights = tf.nn.softmax(logits, -1)
    num_attn_to_discount = len(discounters)
    tf.logging.log(tf.logging.INFO, "Entering Transformer with {} discounters and {} injectors".format(num_attn_to_discount, 0))
    if discounters:
      tf.logging.log(tf.logging.INFO, "Performing discounting")
      discounters = tf.stack(discounters, 1)
      masked_logits = weights[:, -num_attn_to_discount:]
      gate = tf.reduce_sum(discounters, -1, keep_dims=True)
      discounted_weights = discounters + (1 - gate) * masked_logits
      discounted_weights = tf.Print(discounted_weights, [discounted_weights, gate], "Prob to discount", summarize=10)
      weights = tf.concat([tf.nn.softmax(logits[:, :-num_attn_to_discount]), discounted_weights], axis=1)


    weights_drop = tf.nn.dropout(weights, dropout_rate)
    # raise NotImplementedError
    return tf.matmul(weights_drop, v), logits

def okazaki_discounting_dot_product_attention(q, k, v,
                          bias,
                          special_attentions,
                          dropout_rate=1.0):
  """dot-product attention.
  This version is presented
  Args:
    q: a Tensor with shape [batch, heads, length_q, depth_k]
    k: a Tensor with shape [batch, heads, length_kv, depth_k]
    v: a Tensor with shape [batch, heads, length_kv, depth_v]
    discounters: a Tensor with shape [batch, heads, length_q, length_kv]
    bias: bias Tensor (see attention_bias())
    dropout_rate: a floating point number
    name: an optional string
  Returns:
    A Tensor.
  """
  with tf.variable_scope("okazaki_discounting_dot_product_attention", values=[q, k, v, special_attentions]):
    # [batch, num_heads, query_length, memory_length]
    # todo assure the shape of logit and discounter are equal!
    injections = special_attentions[0]
    discounters = special_attentions[1]

    logits = tf.matmul(q, k, transpose_b=True)

    if injections:
      logits = tf.concat([logits] + list(map(lambda x: tf.expand_dims(x, 1), injections)), axis=1)

    if bias is not None:
      logits += bias

    num_attn_to_discount = len(discounters)
    num_attn_to_inject = len(injections)


    if discounters:
      discounters = tf.stack(discounters, 1)
      # discounted_logits = tf.concat([logits[:, :-num_attn_to_discount], logits[:, -num_attn_to_discount:] * discounters], axis=1)
      if num_attn_to_inject == 0:
        discounted_weights = [tf.nn.softmax(logits[:, -num_attn_to_discount:] * discounters, -1)]
      else:
        discounted_weights = [tf.nn.softmax(logits[:, -num_attn_to_discount-num_attn_to_inject:-num_attn_to_inject] * discounters, -1)]
    else:
      discounted_weights = []
    if injections:
      injected_weights = list(map(lambda x: tf.expand_dims(x, 1), injections))
    else:
      injected_weights = []
    if num_attn_to_inject+num_attn_to_discount>0 :
      weights = tf.concat(
        [tf.nn.softmax(logits[:, :-num_attn_to_discount - num_attn_to_inject], -1)] + discounted_weights +
        injected_weights, axis=1)
    else:
      weights = tf.nn.softmax(logits, -1)
    weights_drop = tf.nn.dropout(weights, dropout_rate)
    return tf.matmul(weights_drop, v), logits





def compute_qkv(antecedent, input_depth, total_key_depth, total_value_depth):
  """Computes query, key and value.
  Args:
    total_key_depth: num_heads * key_dim
    total_value_depth: num_heads * value_dim
  Returns:
    q, k, v : [batch, length, depth] tensors
  """
  params = tf.get_variable("qkv_transform", [1, 1, input_depth, 2*total_key_depth + total_value_depth])
  antecedent = tf.expand_dims(antecedent, 1)
  qkv_combined = tf.nn.conv2d(antecedent, params, [1, 1, 1, 1], "SAME")
  # qkv_combined = tf.Print(qkv_combined, [tf.shape(qkv_combined)])
  qkv_combined = tf.squeeze(qkv_combined, 1)
  q, k, v = tf.split(qkv_combined, [total_key_depth, total_key_depth, total_value_depth], axis=2)
  return q, k, v


def multihead_attention(antecedent,
                        bias,
                        num_heads,
                        head_size,
                        dropout_rate,
                        special_attention,
                        special_values,
                        special_attention_mode):
  """Multihead scaled-dot-product attention with input/output transformations.
  Args:
    bias: bias Tensor (see attention_bias())
    total_key_depth: an integer
    total_value_depth: an integer
    output_depth: an integer
    num_heads: an integer dividing total_key_depth and total_value_depth
    dropout_rate: a floating point number
    name: an optional string
  Returns:
    A Tensor.
  Raises:
    ValueError: if the key depth or value depth are not divisible by the
      number of attention heads.
  """
  # if total_key_depth % num_heads != 0:
  #   raise ValueError("Key depth (%d) must be divisible by the number of "
  #                    "attention heads (%d)." % (total_key_depth, num_heads))
  # if total_value_depth % num_heads != 0:
  #   raise ValueError("Value depth (%d) must be divisible by the number of "
  #                    "attention heads (%d)." % (total_value_depth, num_heads))
  with tf.variable_scope("multihead_attention", values=[antecedent]):

    input_size = antecedent.get_shape()[-1]

    total_output_size = head_size * num_heads

    num_basic_attention_heads = num_heads - len(special_attention[0])
    num_basic_value_heads = num_heads - len(special_values)


    total_basic_key_size = num_basic_attention_heads * head_size
    total_basic_value_size = num_basic_value_heads * head_size

    q, k, v = compute_qkv(antecedent, input_size, total_basic_key_size, total_basic_value_size)
    q = split_heads(q, num_basic_attention_heads)
    k = split_heads(k, num_basic_attention_heads)
    v = split_heads(v, num_basic_value_heads)

    # concat special_values to beginning of values; first k attention heads
    # will attend to k special values
    special_values = list(map(lambda x: tf.expand_dims(x, 1), special_values))
    v = tf.concat(special_values + [v], axis=1)

    # key_depth_per_head = total_key_depth // num_heads
    q *= head_size**-0.5
    if special_attention_mode == 'my_discounting':
      x, attn_weights = discounting_dot_product_attention(q, k, v, bias, special_attention, dropout_rate)
    elif special_attention_mode == 'my_discounting_ns':
      x, attn_weights = discounting_dot_product_attention_ns(q, k, v, bias, special_attention, dropout_rate)
    elif special_attention_mode == 'my_discounting_mul':
      x, attn_weights = discounting_dot_product_attention_mul(q, k, v, bias, special_attention, dropout_rate)
    elif special_attention_mode == 'my_discounting_add':
      x, attn_weights = discounting_dot_product_attention_add(q, k, v, bias, special_attention, dropout_rate)
    elif special_attention_mode == 'my_discounting_gt1p':
      x, attn_weights = discounting_dot_product_attention_gt1p(q, k, v, bias, special_attention, dropout_rate)
    elif special_attention_mode == 'my_discounting_gt1p_norext':
      x, attn_weights = discounting_dot_product_attention_gt1p_norext_val(q, k, v, bias, special_attention, dropout_rate)
    elif special_attention_mode == 'my_injection':
      x, attn_weights = injection_dot_product_attention(q, k, v, bias, special_attention, dropout_rate)
    elif special_attention_mode == 'lisa_attn':
      x, attn_weights = dot_product_attention(q, k, v, bias, special_attention, dropout_rate)
    elif special_attention_mode == 'okazaki_discounting':
      x, attn_weights = okazaki_discounting_dot_product_attention(q, k, v, bias, special_attention, dropout_rate)
    else:
      tf.logging.log(tf.logging.FATAL, "Special attention mode {} do not exist".format(special_attention_mode))
      raise NotImplementedError
    x = combine_heads(x)
    params = tf.get_variable("final_proj", [1, 1, total_output_size, total_output_size])
    x = tf.expand_dims(x, 1)
    x = tf.nn.conv2d(x, params, [1, 1, 1, 1], "SAME")
    x = tf.squeeze(x, 1)
    return x, attn_weights


def transformer(inputs, seq_lengths, head_size, num_heads, attn_dropout, ff_dropout, prepost_dropout,
                relu_hidden_size, special_attention, special_values, special_attention_mode = 'injection'):

  # todo deal with special_attention, special_values
  # Note that the current input of special attn is of [[injection], [discounting]]
  with tf.name_scope('transformer_layer'):
    mask = attention_bias_ignore_padding(seq_lengths)

    with tf.variable_scope("self_attention"):
      x = nn_utils.layer_norm(inputs)
      y, attn_weights = multihead_attention(x, mask, num_heads, head_size, attn_dropout, special_attention,
                                            special_values, special_attention_mode)
      x = tf.add(x, tf.nn.dropout(y, prepost_dropout))

    with tf.variable_scope("ffnn"):
      x = nn_utils.layer_norm(x)
      y = conv_hidden_relu(x, relu_hidden_size, num_heads * head_size, ff_dropout)
      x = tf.add(x, tf.nn.dropout(y, prepost_dropout))
    # x = tf.Print(x, ["transformer proceeding", x])

    return x