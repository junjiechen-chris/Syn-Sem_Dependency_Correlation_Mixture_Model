import tensorflow as tf
import numpy as np

import constants
histogram_output = {}


gradient_to_watch = []

def graph_aggregation(dep_graph_list):
  num_dep_grap = dep_graph_list.get_shape()[1]
  with tf.variable_scope("aggregation_weight"):
    weight = tf.get_variable("weight", shape=[num_dep_grap])
    aggregated_graph = tf.math.reduce_sum(dep_graph_list * tf.reshape(tf.nn.softmax(weight), shape=[1, num_dep_grap, 1, 1]), axis=1)
  return aggregated_graph

def linear_graph_aggregation(dep_graph_list, reduction_mode = "sum"):
  num_dep_grap = dep_graph_list.get_shape()[1]
  with tf.variable_scope("aggregation_weight", reuse=tf.AUTO_REUSE):
    weight = tf.get_variable("weight", shape=[num_dep_grap])
    if reduction_mode == "sum":
      aggregated_graph = tf.math.reduce_sum(dep_graph_list * tf.reshape(
      tf.nn.softmax(weight), shape=[1, num_dep_grap, 1, 1]), axis=1)
    elif reduction_mode == "logsumexp":
      # In this case, if the softmaxed weight is small -> insignificant -> throw the activation to -inf
      aggregated_graph = tf.math.reduce_logsumexp(
        tf.clip_by_value(dep_graph_list / tf.reshape(
        tf.nn.softmax(weight), shape=[1, num_dep_grap, 1, 1]), clip_value_min=constants.VERY_SMALL, clip_value_max=constants.VERY_LARGE), axis=1)
    elif reduction_mode == "logsumexp_add":
      aggregated_graph = tf.math.reduce_logsumexp(
        tf.clip_by_value(dep_graph_list + tf.reshape(
          weight, shape=[1, num_dep_grap, 1, 1]), clip_value_min=constants.VERY_SMALL,
                         clip_value_max=constants.VERY_LARGE), axis=1)
    elif reduction_mode == "softmax":
      aggregated_graph = tf.math.reduce_logsumexp(
        tf.clip_by_value(tf.nn.softmax(dep_graph_list, dim=-1) + tf.reshape(
          weight, shape=[1, num_dep_grap, 1, 1]), clip_value_min=constants.VERY_SMALL,
                         clip_value_max=constants.VERY_LARGE), axis=1)
    else:
      tf.logging.log(tf.logging.ERROR, "Linear graph aggregation: reduction mode {} is not supported".format(reduction_mode))
      raise ValueError
    if "{}".format(weight.name.replace(":","_")) not in histogram_output:
      summary = tf.summary.histogram("weight", tf.math.sigmoid(weight))
      heatmap = tf.reshape(weight, shape=[1, 1, -1, 1])
      # heatmap_minus = tf.math.maximum(-weight, 0)/tf.reduce_max(-weight)*255
      # heat_map = tf.expand_dims(tf.expand_dims(tf.stack([heatmap_minus, heatmap_pls], 0), 0), -1)
      summary_heatmap = tf.summary.image("weight_map", heatmap)
      histogram_output["{}".format(weight.name.replace(":", "_"))] = summary
      histogram_output["{}_heatmap".format(weight.name.replace(":", "_"))] = summary_heatmap

    # tf.summary.histogram("{}_aggregation_distribution".format(weight.name), tf.nn.softmax(weight))
    return aggregated_graph, tf.nn.softmax(weight)


def graph_aggregation_softmax_done(dep_graph_list, parser_keep_rate = 0.9):
  num_dep_grap = dep_graph_list.get_shape()[1]
  with tf.variable_scope("aggregation_weight"):
    weight = tf.get_variable("weight", shape=[num_dep_grap], trainable=True)
    aggregated_graph = tf.math.reduce_sum(tf.nn.softmax(dep_graph_list, dim=-1) * tf.reshape(tf.nn.dropout(tf.nn.softmax(weight), keep_prob=parser_keep_rate), shape=[1, num_dep_grap, 1, 1]), axis=1)
    return aggregated_graph, tf.nn.softmax(weight)

def graph_mean_aggregation(dep_graph_list, parser_keep_rate = 0.9):
  num_dep_grap = dep_graph_list.get_shape()[1]
  with tf.variable_scope("aggregation_weight"):
    # Be careful as it'd break previous checkpoints
    weight = tf.constant([1.] * num_dep_grap)
    if num_dep_grap> 1:
      aggregated_graph = tf.math.reduce_sum(tf.nn.softmax(dep_graph_list, dim=-1) * tf.reshape(
      tf.nn.dropout(tf.nn.softmax(weight), keep_prob=parser_keep_rate), shape=[1, num_dep_grap, 1, 1]), axis=1)
    else:
      aggregated_graph = tf.math.reduce_sum(tf.nn.softmax(dep_graph_list, dim=-1) * tf.reshape(
        tf.nn.softmax(weight), shape=[1, num_dep_grap, 1, 1]), axis=1)

    #aggregated_graph = tf.nn.softmax(tf.math.reduce_sum(dep_graph_list, axis=0), dim=-1)
  return aggregated_graph

def graph_mean_aggregation_prob(dep_graph_list, parser_keep_rate = 0.9):
  num_dep_grap = dep_graph_list.get_shape()[1]
  # print("dep_graph", dep_graph_list)
  # print("graph_mean_aggregation, dep_graph_list", dep_graph_list)
  with tf.variable_scope("aggregation_weight"):
    # Be careful as it'd break previous checkpoints
    weight = tf.constant([1.] * num_dep_grap)
    if num_dep_grap> 1:
      aggregated_graph = tf.math.reduce_sum(dep_graph_list * tf.reshape(
      tf.nn.dropout(tf.nn.softmax(weight), keep_prob=parser_keep_rate), shape=[1, num_dep_grap, 1, 1]), axis=1)
    else:
      aggregated_graph = tf.math.reduce_sum(dep_graph_list * tf.reshape(
        tf.nn.softmax(weight), shape=[1, num_dep_grap, 1, 1]), axis=1)

    #aggregated_graph = tf.nn.softmax(tf.math.reduce_sum(dep_graph_list, axis=0), dim=-1)
  return aggregated_graph


def graph_mlp_aggregation(dep_graph_list, v, mlp_dropout, projection_dim, parser_dkeep_rate=0.9, batch_norm = False):
  #graph
  # dep_graph_list = tf.transpose(dep_graph_list, )
  num_dep_grap = dep_graph_list.get_shape()[1]
  with tf.variable_scope('MLP'):
    mlp = MLP(v, projection_dim, keep_prob=mlp_dropout, n_splits=1)
    mlp = tf.nn.leaky_relu(mlp)
  if batch_norm:
    tf.logging.log(tf.logging.INFO, "Using batch normalization @ {}".format(tf.get_variable_scope().name))
    mean, variance = tf.nn.moments(mlp, [-1], keep_dims=True)
    with tf.variable_scope('BatchNorm'):
      beta = tf.get_variable('offset', [1, projection_dim])
      gamma = tf.get_variable('scale', [1])
    mlp = tf.nn.batch_normalization(mlp, mean, variance, offset=beta, scale=gamma, variance_epsilon=1e-6)
  with tf.variable_scope('Classifier'):
    logits = MLP(mlp, num_dep_grap, keep_prob=mlp_dropout, n_splits=1)
  aggregated_graph = tf.math.reduce_sum(
      tf.nn.softmax(dep_graph_list, dim=-1) * tf.nn.dropout(tf.nn.softmax(logits), keep_prob=parser_dkeep_rate)[:, :, tf.newaxis, tf.newaxis],
      axis=1)
  return aggregated_graph, tf.nn.softmax(logits)

def leaky_relu(x): return tf.maximum(0.1 * x, x)


def int_to_str_lookup_table(inputs, lookup_map):
  # todo order of map.values() is probably not guaranteed; should prob sort by keys first
  # print("int_to_str_lookup: {}".format(np.array(list(lookup_map.values()))))
  return tf.nn.embedding_lookup(np.array(list(lookup_map.values())), inputs)


def set_vars_to_moving_average(moving_averager):
  moving_avg_variables = tf.get_collection(tf.GraphKeys.MOVING_AVERAGE_VARIABLES)
  return tf.group(*[tf.assign(x, moving_averager.average(x)) for x in moving_avg_variables])


def layer_norm(inputs, epsilon=1e-6):
  """Applies layer normalization.

  Args:
    inputs: A tensor with 2 or more dimensions, where the first dimension has
      `batch_size`.
    epsilon: A floating number. A very small number for preventing ZeroDivision Error.

  Returns:
    A tensor with the same shape and data dtype as `inputs`.
  """
  with tf.variable_scope("layer_norm"):
    inputs_shape = inputs.get_shape()
    params_shape = inputs_shape[-1:]

    mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
    beta = tf.get_variable("beta", params_shape, initializer=tf.zeros_initializer())
    gamma = tf.get_variable("gamma", params_shape, initializer=tf.ones_initializer())
    normalized = (inputs - mean) * tf.rsqrt(variance + epsilon)
    outputs = gamma * normalized + beta
  return outputs


# def orthonormal_initializer(input_size, output_size):
#   """"""
#
#   if not tf.get_variable_scope().reuse:
#     print(tf.get_variable_scope().name)
#     I = np.eye(output_size)
#     lr = .1
#     eps = .05 / (output_size + input_size)
#     success = False
#     while not success:
#       Q = np.random.randn(input_size, output_size) / np.sqrt(output_size)
#       for i in range(100):
#         QTQmI = Q.T.dot(Q) - I
#         loss = np.sum(QTQmI ** 2 / 2)
#         Q2 = Q ** 2
#         Q -= lr * Q.dot(QTQmI) / (np.abs(Q2 + Q2.sum(axis=0, keepdims=True) + Q2.sum(axis=1, keepdims=True) - 1) + eps)
#         if np.isnan(Q[0, 0]):
#           lr /= 2
#           break
#       if np.isfinite(loss) and np.max(Q) < 1e6:
#         success = True
#       eps *= 2
#     print('Orthogonal pretrainer loss: %.2e' % loss)
#   else:
#     print('Orthogonal pretrainer failed, using non-orthogonal random matrix')
#     Q = np.random.randn(input_size, output_size) / np.sqrt(output_size)
#   return Q.astype(np.float32)


def linear_layer(inputs, output_size, add_bias=True, n_splits=1, initializer=None):
  """"""

  if not isinstance(inputs, (list, tuple)):
    inputs = [inputs]
  output_size *= n_splits

  with tf.variable_scope('Linear'):
    # Reformat the input
    total_input_size = 0
    shapes = [a.get_shape().as_list() for a in inputs]
    for shape in shapes:
      total_input_size += shape[-1]
    input_shape = tf.shape(inputs[0])
    output_shape = []
    for i in range(len(shapes[0])):
      output_shape.append(input_shape[i])
    output_shape[-1] = output_size
    output_shape = tf.stack(output_shape)
    for i, (input_, shape) in enumerate(zip(inputs, shapes)):
      inputs[i] = tf.reshape(input_, [-1, shape[-1]])
    concatenation = tf.concat(axis=1, values=inputs)

    # Get the matrix
    if initializer is None:
      initializer = tf.initializers.orthogonal
      # mat = orthonormal_initializer(total_input_size, output_size // n_splits)
      # mat = np.concatenate([mat] * n_splits, axis=1)
      # initializer = tf.constant_initializer(mat)
    matrix = tf.get_variable('Weights', [total_input_size, output_size], initializer=initializer)
    # tf.add_to_collection('Weights', matrix)

    # Get the bias
    if add_bias:
      bias = tf.get_variable('Biases', [output_size], initializer=tf.zeros_initializer())
    else:
      bias = 0

    # Do the multiplication
    new = tf.matmul(concatenation, matrix) + bias
    new = tf.reshape(new, output_shape)
    new.set_shape([tf.Dimension(None) for _ in range(len(shapes[0]) - 1)] + [tf.Dimension(output_size)])
    if n_splits > 1:
      return tf.split(axis=len(new.get_shape().as_list()) - 1, num_or_size_splits=n_splits, value=new)
    else:
      return new


# TODO clean this up
def MLP(inputs, output_size, func=leaky_relu, keep_prob=1.0, n_splits=1):
  """"""
  # tf.Print(inputs, ["within MLP 0", inputs])
  input_shape = inputs.get_shape().as_list()
  n_dims = len(input_shape)
  batch_size = tf.shape(inputs)[0]
  input_size = tf.shape(inputs)[-1]
  shape_to_set = [tf.Dimension(None)] * (n_dims - 1) + [tf.Dimension(output_size)]
  # input_shape = tf.Print(input_shape, [input_shape])
  # tf.Print(inputs, ["within MLP 1", inputs])
  if keep_prob < 1:
    noise_shape = tf.stack([batch_size] + [1] * (n_dims - 2) + [input_size])
    inputs = tf.nn.dropout(inputs, keep_prob, noise_shape=noise_shape)
  # tf.Print(inputs, ["within MLP 2", inputs])
  linear = linear_layer(inputs,
                        output_size,
                        n_splits=n_splits,
                        add_bias=True)
  if n_splits == 1:
    linear = [linear]
  for i, split in enumerate(linear):
    split = func(split)
    split.set_shape(shape_to_set)
    linear[i] = split
  if n_splits == 1:
    return linear[0]
  else:
    return linear


def nearPSD(A, epsilon=0):
  n = A.shape[0]
  eigval, eigvec = tf.linalg.eig(A)
  val = tf.maximum(eigval, epsilon)
  vec = np.matrix(eigvec)
  T = 1 / (np.multiply(vec, vec) * val.T)
  T = np.matrix(np.sqrt(np.diag(np.array(T).reshape((n)))))
  B = T * vec * np.diag(np.array(np.sqrt(val)).reshape((n)))
  out = B * B.T
  return (out)

def bilinear(inputs1, inputs2, output_size, add_bias2=True, add_bias1=True, add_bias=False, initializer=None):
  """"""

  with tf.variable_scope('Bilinear'):
    # Reformat the inputs
    ndims = len(inputs1.get_shape().as_list())
    inputs1_shape = tf.shape(inputs1)
    inputs1_bucket_size = inputs1_shape[ndims - 2]
    inputs1_size = inputs1.get_shape().as_list()[-1]

    inputs2_shape = tf.shape(inputs2)
    inputs2_bucket_size = inputs2_shape[ndims - 2]
    inputs2_size = inputs2.get_shape().as_list()[-1]
    # output_shape = []
    batch_size1 = 1
    batch_size2 = 1
    for i in range(ndims - 2):
      batch_size1 *= inputs1_shape[i]
      batch_size2 *= inputs2_shape[i]
      # output_shape.append(inputs1_shape[i])
    # output_shape.append(inputs1_bucket_size)
    # output_shape.append(output_size)
    # output_shape.append(inputs2_bucket_size)
    # output_shape = tf.stack(output_shape)
    inputs1 = tf.reshape(inputs1, tf.stack([batch_size1, inputs1_bucket_size, inputs1_size]))
    inputs2 = tf.reshape(inputs2, tf.stack([batch_size2, inputs2_bucket_size, inputs2_size]))
    if add_bias1:
      inputs1 = tf.concat(axis=2, values=[inputs1, tf.ones(tf.stack([batch_size1, inputs1_bucket_size, 1]))])
    if add_bias2:
      inputs2 = tf.concat(axis=2, values=[inputs2, tf.ones(tf.stack([batch_size2, inputs2_bucket_size, 1]))])

    # Get the matrix
    if initializer is None:
      # mat = orthonormal_initializer(inputs1_size + add_bias1, inputs2_size + add_bias2)[:, None, :]
      # mat = np.concatenate([mat] * output_size, axis=1)
      # initializer = tf.constant_initializer(mat)
      initializer = tf.initializers.orthogonal
    weights = tf.get_variable('Weights', [inputs1_size + add_bias1, output_size, inputs2_size + add_bias2],
                              initializer=initializer)
    # tf.add_to_collection('Weights', weights)

    # inputs1: num_triggers_in_batch x 1 x self.trigger_mlp_size
    # inputs2: batch x seq_len x self.role_mlp_size

    # Do the multiplications
    # (bn x d) (d x rd) -> (bn x rd)
    lin = tf.matmul(tf.reshape(inputs1, [-1, inputs1_size + add_bias1]), tf.reshape(weights, [inputs1_size + add_bias1, -1]))
    # (b x nr x d) (b x n x d)T -> (b x nr x n)
    lin_reshape = tf.reshape(lin, tf.stack([batch_size1, inputs1_bucket_size * output_size, inputs2_size + add_bias2]))
    bilin = tf.matmul(lin_reshape, inputs2, adjoint_b=True)
    # (bn x r x n)
    bilin = tf.reshape(bilin, tf.stack([-1, output_size, inputs2_bucket_size]))

    # Get the bias
    if add_bias:
      bias = tf.get_variable('Biases', [output_size], initializer=tf.zeros_initializer())
      bilin += tf.expand_dims(bias, 1)

    return bilin


def bilinear_t4(inputs1, inputs2, output_size, add_bias2=True, add_bias1=True, add_bias=False, initializer=None):
  """"""

  with tf.variable_scope('Bilinear'):
    # Reformat the inputs
    ndims = len(inputs1.get_shape().as_list())
    inputs1_shape = tf.shape(inputs1)
    inputs1_bucket_size = inputs1_shape[ndims - 2]
    inputs1_size = inputs1.get_shape().as_list()[-1]

    inputs2_shape = tf.shape(inputs2)
    inputs2_bucket_size = inputs2_shape[ndims - 2]
    inputs2_size = inputs2.get_shape().as_list()[-1]
    # output_shape = []
    batch_size1 = 1
    batch_size2 = 1
    for i in range(ndims - 2):
      batch_size1 *= inputs1_shape[i]
      batch_size2 *= inputs2_shape[i]
      # output_shape.append(inputs1_shape[i])
    # output_shape.append(inputs1_bucket_size)
    # output_shape.append(output_size)
    # output_shape.append(inputs2_bucket_size)
    # output_shape = tf.stack(output_shape)
    inputs1 = tf.reshape(inputs1, tf.stack([batch_size1, inputs1_bucket_size, inputs1_size]))
    inputs2 = tf.reshape(inputs2, tf.stack([batch_size2, inputs2_bucket_size, inputs2_size]))
    if add_bias1:
      inputs1 = tf.concat(axis=2, values=[inputs1, tf.ones(tf.stack([batch_size1, inputs1_bucket_size, 1]))])
    if add_bias2:
      inputs2 = tf.concat(axis=2, values=[inputs2, tf.ones(tf.stack([batch_size2, inputs2_bucket_size, 1]))])

    # Get the matrix
    if initializer is None:
      # mat = orthonormal_initializer(inputs1_size + add_bias1, inputs2_size + add_bias2)[:, None, :]
      # mat = np.concatenate([mat] * output_size, axis=1)
      # initializer = tf.constant_initializer(mat)
      initializer = tf.initializers.orthogonal
    weights = tf.get_variable('Weights', [inputs1_size + add_bias1, output_size, inputs2_size + add_bias2],
                              initializer=initializer)
    # tf.add_to_collection('Weights', weights)

    # inputs1: num_triggers_in_batch x 1 x self.trigger_mlp_size
    # inputs2: batch x seq_len x self.role_mlp_size

    # Do the multiplications
    # (bn x d) (d x rd) -> (bn x rd)
    lin = tf.matmul(tf.reshape(inputs1, [-1, inputs1_size + add_bias1]), tf.reshape(weights, [inputs1_size + add_bias1, -1]))
    # (b x nr x d) (b x n x d)T -> (b x nr x n)
    lin_reshape = tf.reshape(lin, tf.stack([batch_size1, inputs1_bucket_size, output_size, inputs2_size + add_bias2]))
    bilin = tf.matmul(lin_reshape, inputs2, adjoint_b=True)
    # (bn x r x n)
    bilin = tf.reshape(bilin, tf.stack([-1, output_size, inputs2_bucket_size]))

    # Get the bias
    if add_bias:
      bias = tf.get_variable('Biases', [output_size], initializer=tf.zeros_initializer())
      bilin += tf.expand_dims(bias, 1)

    return bilin




def bilinear_classifier(inputs1, inputs2, keep_prob, add_bias1=True, add_bias2=False):
  """"""

  input_shape = tf.shape(inputs1)
  batch_size = input_shape[0]
  bucket_size = input_shape[1]
  input_size = inputs1.get_shape().as_list()[-1]

  if keep_prob < 1:
    noise_shape = [batch_size, 1, input_size]
    inputs1 = tf.nn.dropout(inputs1, keep_prob, noise_shape=noise_shape)
    inputs2 = tf.nn.dropout(inputs2, keep_prob, noise_shape=noise_shape)

  bilin = bilinear(inputs1, inputs2, 1,
                   add_bias1=add_bias1,
                   add_bias2=add_bias2,
                   initializer=tf.zeros_initializer())
  output = tf.reshape(bilin, [batch_size, bucket_size, bucket_size])
  # output = tf.squeeze(bilin)
  return output

def bilinear_classifier_nary_single_prior(inputs1, inputs2, n_classes, keep_prob, add_bias1=True, add_bias2=False):
  # To check whether there's a need to add prior to predicate?
  # Since predicate has been fixed...
  """"""

  input_shape1 = tf.shape(inputs1)
  input_shape2 = tf.shape(inputs2)

  batch_size1 = input_shape1[0]
  batch_size2 = input_shape2[0]

  # with tf.control_dependencies([tf.assert_equal(input_shape1[1], input_shape2[1])]):
  bucket_size1 = input_shape1[1]
  bucket_size2 = input_shape2[1]
  input_size1 = inputs1.get_shape().as_list()[-1]
  input_size2 = inputs2.get_shape().as_list()[-1]

  input_shape_to_set1 = [tf.Dimension(None), tf.Dimension(None), input_size1 + 1]
  input_shape_to_set2 = [tf.Dimension(None), tf.Dimension(None), input_size2 ]

  if isinstance(keep_prob, tf.Tensor) or keep_prob < 1:
    noise_shape1 = tf.stack([batch_size1, 1, input_size1])
    noise_shape2 = tf.stack([batch_size2, 1, input_size2])

    inputs1 = tf.nn.dropout(inputs1, keep_prob, noise_shape=noise_shape1)
    inputs2 = tf.nn.dropout(inputs2, keep_prob, noise_shape=noise_shape2)

  inputs1 = tf.concat(axis=2, values=[inputs1, tf.ones(tf.stack([batch_size1, bucket_size1, 1]))])
  inputs1.set_shape(input_shape_to_set1)
  # inputs2 = tf.concat(axis=2, values=[inputs2, tf.#ones(tf.stack([batch_size2, bucket_size2, 1]))])
  # inputs2.set_shape(input_shape_to_set2)

  bilin = bilinear(inputs1, inputs2,
                   n_classes,
                   add_bias1=add_bias1,
                   add_bias2=add_bias2,
                   initializer=tf.zeros_initializer())

  return bilin
def bilinear_classifier_nary(inputs1, inputs2, n_classes, keep_prob, add_bias1=True, add_bias2=True):
  """"""

  input_shape1 = tf.shape(inputs1)
  input_shape2 = tf.shape(inputs2)

  batch_size1 = input_shape1[0]
  batch_size2 = input_shape2[0]

  # with tf.control_dependencies([tf.assert_equal(input_shape1[1], input_shape2[1])]):
  bucket_size1 = input_shape1[1]
  bucket_size2 = input_shape2[1]
  input_size1 = inputs1.get_shape().as_list()[-1]
  input_size2 = inputs2.get_shape().as_list()[-1]

  input_shape_to_set1 = [tf.Dimension(None), tf.Dimension(None), input_size1 + 1]
  input_shape_to_set2 = [tf.Dimension(None), tf.Dimension(None), input_size2 + 1]

  if isinstance(keep_prob, tf.Tensor) or keep_prob < 1:
    noise_shape1 = tf.stack([batch_size1, 1, input_size1])
    noise_shape2 = tf.stack([batch_size2, 1, input_size2])

    inputs1 = tf.nn.dropout(inputs1, keep_prob, noise_shape=noise_shape1)
    inputs2 = tf.nn.dropout(inputs2, keep_prob, noise_shape=noise_shape2)

  inputs1 = tf.concat(axis=2, values=[inputs1, tf.ones(tf.stack([batch_size1, bucket_size1, 1]))])
  inputs1.set_shape(input_shape_to_set1)
  inputs2 = tf.concat(axis=2, values=[inputs2, tf.ones(tf.stack([batch_size2, bucket_size2, 1]))])
  inputs2.set_shape(input_shape_to_set2)

  bilin = bilinear(inputs1, inputs2,
                   n_classes,
                   add_bias1=add_bias1,
                   add_bias2=add_bias2,
                   initializer=tf.zeros_initializer())

  return bilin

def bilinear_classifier_nary_t4(inputs1, inputs2, n_classes, keep_prob, add_bias1=True, add_bias2=True):
  """"""

  input_shape1 = tf.shape(inputs1)
  input_shape2 = tf.shape(inputs2)

  batch_size1 = input_shape1[0]
  batch_size2 = input_shape2[0]

  # with tf.control_dependencies([tf.assert_equal(input_shape1[1], input_shape2[1])]):
  bucket_size1 = input_shape1[1]
  bucket_size2 = input_shape2[1]
  input_size1 = inputs1.get_shape().as_list()[-1]
  input_size2 = inputs2.get_shape().as_list()[-1]

  input_shape_to_set1 = [tf.Dimension(None), tf.Dimension(None), input_size1 + 1]
  input_shape_to_set2 = [tf.Dimension(None), tf.Dimension(None), tf.Dimension(None), input_size2 + 1]

  if isinstance(keep_prob, tf.Tensor) or keep_prob < 1:
    noise_shape1 = tf.stack([batch_size1, 1, input_size1])
    noise_shape2 = tf.stack([batch_size2, 1, 1, input_size2])

    inputs1 = tf.nn.dropout(inputs1, keep_prob, noise_shape=noise_shape1)
    inputs2 = tf.nn.dropout(inputs2, keep_prob, noise_shape=noise_shape2)

  inputs1 = tf.concat(axis=2, values=[inputs1, tf.ones(tf.stack([batch_size1, bucket_size1, 1]))])
  inputs1.set_shape(input_shape_to_set1)
  inputs2 = tf.concat(axis=2, values=[inputs2, tf.ones(tf.stack([batch_size2, bucket_size2, bucket_size2, 1]))])
  inputs2.set_shape(input_shape_to_set2)

  bilin = bilinear_t4(inputs1, inputs2,
                   n_classes,
                   add_bias1=add_bias1,
                   add_bias2=add_bias2,
                   initializer=tf.zeros_initializer())

  return bilin


def conditional_bilinear_classifier(inputs1, inputs2, n_classes, probs, keep_prob, add_bias1=True, add_bias2=True):
  """"""
  # prob : head probability


  input_shape = tf.shape(inputs1)
  batch_size = input_shape[0]
  bucket_size = input_shape[1]
  input_size = inputs1.get_shape().as_list()[-1]
  input_shape_to_set = [tf.Dimension(None), tf.Dimension(None), input_size + 1]
  # output_shape = tf.stack([batch_size, bucket_size, n_classes, bucket_size])
  if len(probs.get_shape().as_list()) == 2:
    probs = tf.to_float(tf.one_hot(tf.to_int64(probs), bucket_size, 1, 0))
  else:
    probs = tf.stop_gradient(probs)

  # print("debug <probability shape>: ", probs.shape)

  if keep_prob < 1:
    noise_shape = tf.stack([batch_size, 1, input_size])
    inputs1 = tf.nn.dropout(inputs1, keep_prob, noise_shape=noise_shape)
    inputs2 = tf.nn.dropout(inputs2, keep_prob, noise_shape=noise_shape)

  inputs1 = tf.concat(axis=2, values=[inputs1, tf.ones(tf.stack([batch_size, bucket_size, 1]))])
  inputs1.set_shape(input_shape_to_set)
  inputs2 = tf.concat(axis=2, values=[inputs2, tf.ones(tf.stack([batch_size, bucket_size, 1]))])
  inputs2.set_shape(input_shape_to_set)

  bilin = bilinear(inputs1, inputs2,
                   n_classes,
                   add_bias1=add_bias1,
                   add_bias2=add_bias2,
                   initializer=tf.zeros_initializer())
  bilin = tf.reshape(bilin, [batch_size, bucket_size, n_classes, bucket_size])
  # print("debug <shape of bilin vs. probs>", bilin, " ", tf.expand_dims(probs, 3))
  # probs = tf.Print(probs, ['probs:', tf.shape(probs)])
  # print("debug <tf.matmul(bilin, tf.expand_dims(probs, 3))@bilinear classifier>: ", tf.matmul(bilin, tf.expand_dims(probs, 3)))
  weighted_bilin = tf.squeeze(tf.matmul(bilin, tf.expand_dims(probs, 3)), -1)

  return weighted_bilin, bilin


def generating_head_mtx_from_head_label_dist(heads, labels, num_labels, tokens_to_keep, chain, softmax=tf.nn.softmax, score_multiplier = 1., head_label_aggregation_fn = tf.math.truediv, use_strength_bias = False, head_label_aggregation = "log_multi", label_score_aggregation = "expectation", ls_multiplier = 10.):
  """
    heads: head-dependent distribution of shape (B, seq_len, seq_len), must be masked!
    labels: label distribution for each head-dependent choice, shape: (B, seq_len, labels)
  """

  head_strength = []
  for chain_item, cnt in zip(chain, range(len(chain))):
    tmp_head_strength = []
    direction = chain_item[0]
    steps = int(chain_item[1])
    for step in range(steps):
      with tf.variable_scope("dependency_label_strength_{}_{}_{}".format(cnt, direction, step), ):
        if label_score_aggregation == "logsumexp":
          s = tf.get_variable("label_strength", shape=[1, 1, num_labels],
                              initializer=tf.random_uniform_initializer(minval=0.1, maxval=1.0))
          # s = tf.print(s, [s], "label_strength", output_stream=sys.stdout)
          label_score = tf.math.reduce_logsumexp(tf.math.log(s) + tf.math.log_softmax(labels), -1, keep_dims=True)
        elif label_score_aggregation == "expectation":
          s = tf.get_variable("label_strength", shape=[1, 1, num_labels])
          gamma = tf.get_variable("gamma", shape=[1, 1, 1], initializer=tf.ones_initializer())
          beta = tf.get_variable("beta", shape=[1, 1, 1],
                                 trainable=use_strength_bias)
          s = gamma * s + beta
          label_score = tf.math.reduce_sum(s * tf.nn.softmax(labels), -1, keep_dims=True)
        elif label_score_aggregation == "expectationp3":
          s = tf.get_variable("label_strength", shape=[1, 1, num_labels])

          gamma = tf.get_variable("gamma", shape=[1, 1, 1])
          beta = tf.get_variable("beta", shape=[1, 1, 1], initializer=tf.zeros_initializer(),
                                 trainable=use_strength_bias)
          s = gamma * s + beta + 3
          # s = tf.Print(s, [tf.sigmoid(s)])
          label_score = tf.math.reduce_sum(s * tf.nn.softmax(labels), -1, keep_dims=True)
        elif label_score_aggregation == "expectationp_zero_bias":
          s = tf.get_variable("label_strength", shape=[1, 1, num_labels], initializer=tf.zeros_initializer())
          gamma = tf.get_variable("gamma", shape=[1, 1, 1], initializer=tf.ones_initializer())
          beta = tf.get_variable("beta", shape=[1, 1, 1], initializer=tf.zeros_initializer(),
                                 trainable=use_strength_bias)
          s = gamma * s + beta
          label_score = tf.math.reduce_sum(s * tf.nn.softmax(labels), -1, keep_dims=True)
        elif label_score_aggregation == "expectationm3":
          s = tf.get_variable("label_strength", shape=[1, 1, num_labels])
          gamma = tf.get_variable("gamma", shape=[1, 1, 1])
          beta = tf.get_variable("beta", shape=[1, 1, 1], initializer=tf.zeros_initializer(),
                                 trainable=use_strength_bias)
          s = gamma * s + beta - 3
          label_score = tf.math.reduce_sum(s * tf.nn.softmax(labels), -1, keep_dims=True)
        elif label_score_aggregation == "bilinear":
          # Not effective
          s = tf.tile(s, [tf.shape(labels)[0], 1, 1])
          bilin = bilinear(s, labels, 1,
                           add_bias1=use_strength_bias,
                           add_bias2=False,
                           initializer=tf.zeros_initializer())
          label_score = bilin

        else:
          tf.logging.log(tf.logging.ERROR,
                         "label_score_aggregation function is not defined {}".format(head_label_aggregation))
          raise ValueError
        # unnormalized_head_strength = heads + tf.math.log(label_score)
        if head_label_aggregation == "log_multi":
          unnormalized_head_strength = heads + label_score
        elif head_label_aggregation == "log_sigmoid":
          unnormalized_head_strength = tf.nn.log_softmax(heads) + tf.math.log_sigmoid(label_score)
        elif head_label_aggregation == "gating":
          self_looping_map = tf.eye(tf.shape(heads)[-1], batch_shape=[tf.shape(heads)[0]])*10
          gate = tf.math.log_sigmoid(label_score)
          unnormalized_head_strength = tf.reduce_logsumexp(tf.stack(
            [gate + heads, tf.math.log(1-tf.math.exp(gate)) + tf.nn.log_softmax(self_looping_map)], -1
          ), -1)
        elif head_label_aggregation == "gating_on_prob":
          # self_looping_map = tf.eye(tf.shape(heads)[-1], batch_shape=[tf.shape(heads)[0]])*10
          gate = tf.nn.sigmoid(label_score)
          unnormalized_head_strength = gate * tf.nn.softmax(heads)
        elif head_label_aggregation == "masking_with_gate":
          gate = tf.nn.sigmoid(label_score)
          gate = tf.Print(gate, [gate, gate * tf.nn.softmax(heads)], "gate @ transformation fn")
          unnormalized_head_strength = tf.math.log(tf.maximum(gate * tf.nn.softmax(heads), 1e-10))
        elif head_label_aggregation == "masking_with_gate_full_log":
          gate = tf.log_sigmoid(label_score)
          unnormalized_head_strength = gate + tf.nn.log_softmax(heads)
        elif head_label_aggregation == "linear":
          unnormalized_head_strength = tf.math.log_sigmoid(heads) + ls_multiplier * label_score
          max_component = tf.reduce_max(ls_multiplier * label_score, -2, keepdims=True)
          # max_component = tf.print(max_component, [max_component], "max_component", output_stream= sys.stdout )
          unnormalized_head_strength -= max_component
        else:
          tf.logging.log(tf.logging.ERROR, "head_label_aggregation function is not defined {}".format(head_label_aggregation))
          raise ValueError
        head_strength.append(unnormalized_head_strength)
  return head_strength

def generating_prior_mtx_for_srl(heads, labels, num_srl_labels, tokens_to_keep, chain, softmax=tf.nn.softmax, score_multiplier = 1., head_label_aggregation_fn = tf.math.truediv, use_strength_bias = False, head_label_aggregation = "log_multi", label_score_aggregation = "expectation", ls_multiplier = 10., memory_efficient = False):
  """
    heads: head-dependent distribution of shape (B, seq_len, seq_len), must be masked!
    labels: label distribution for each head-dependent choice, shape: (B, seq_len, labels)
  """

  head_strength = []
  seq_len = tf.shape(labels)[1]
  if memory_efficient:
    labels = tf.expand_dims(labels, 2)
    heads = tf.expand_dims(tf.nn.softmax(heads, -1), 3)
  else:
    labels = tf.tile(tf.expand_dims(labels, 2), [1, 1, seq_len, 1])
    heads = tf.tile(tf.expand_dims(tf.nn.softmax(heads, -1), 3), [1, 1, 1, num_srl_labels])
  for chain_item, cnt in zip(chain, range(len(chain))):
    tmp_head_strength = []
    direction = chain_item[0]
    steps = int(chain_item[1])
    for step in range(steps):
      with tf.variable_scope("dependency_label_strength_{}_{}_{}".format(cnt, direction, step), ):

        dense = tf.keras.layers.Dense(
          num_srl_labels, activation=None, use_bias=True,
          kernel_initializer='glorot_uniform',
          bias_initializer='zeros'
        )
        tf.logging.log(tf.logging.INFO, "labeled dep -> srl prior: {}".format(dense))
        output = dense(labels)
        output = tf.Print(output, [output], "labeled dep -> srl prior, logit form")
        output = tf.sigmoid(output) # treat this one as prior
        output = tf.Print(output, [output], "labeled dep -> srl prior, log_sigmoid form")

        combined_head_label = heads*output
        tf.logging.log(tf.logging.INFO, "labeled dep -> srl prior output: {}".format(combined_head_label))
        head_strength.append(combined_head_label)
  return head_strength

def generating_prior_mtx_with_pos(pos_tag, num_srl_labels):
  """
    heads: head-dependent distribution of shape (B, seq_len, seq_len), must be masked!
    labels: label distribution for each head-dependent choice, shape: (B, seq_len, labels)
  """
  seq_len = tf.shape(pos_tag)[1]
  batch_size = tf.shape(pos_tag)[0]

  labels = tf.expand_dims(pos_tag, 2)
  heads = tf.expand_dims(tf.eye(seq_len, batch_shape=[batch_size]), 3)#tf.expand_dims(tf.nn.softmax(heads, -1), 3)
  with tf.variable_scope("pos_tag_related"):

    dense = tf.keras.layers.Dense(
      num_srl_labels, activation=None, use_bias=True,
      kernel_initializer='glorot_uniform',
      bias_initializer='zeros'
    )
    tf.logging.log(tf.logging.INFO, "labeled dep -> srl prior: {}".format(dense))
    output = dense(labels)
    output = tf.Print(output, [output], "labeled dep -> srl prior, logit form")
    output = tf.sigmoid(output)  # treat this one as prior
    output = tf.Print(output, [output], "labeled dep -> srl prior, log_sigmoid form")

    combined_head_label = heads * output
    tf.logging.log(tf.logging.INFO, "labeled dep -> srl prior output: {}".format(combined_head_label))

  return combined_head_label



def generating_prior_mtx_for_srl_relu(heads, labels, num_srl_labels, tokens_to_keep, chain, softmax=tf.nn.softmax, score_multiplier = 1., head_label_aggregation_fn = tf.math.truediv, use_strength_bias = False, head_label_aggregation = "log_multi", label_score_aggregation = "expectation", ls_multiplier = 10., memory_efficient = False):
  """
    heads: head-dependent distribution of shape (B, seq_len, seq_len), must be masked!
    labels: label distribution for each head-dependent choice, shape: (B, seq_len, labels)
  """

  head_strength = []
  seq_len = tf.shape(labels)[1]
  if memory_efficient:
    labels = tf.expand_dims(labels, 2)
    heads = tf.expand_dims(tf.nn.softmax(heads, -1), 3)
  else:
    labels = tf.tile(tf.expand_dims(labels, 2), [1, 1, seq_len, 1])
    heads = tf.tile(tf.expand_dims(tf.nn.softmax(heads, -1), 3), [1, 1, 1, num_srl_labels])
  for chain_item, cnt in zip(chain, range(len(chain))):
    tmp_head_strength = []
    direction = chain_item[0]
    steps = int(chain_item[1])
    for step in range(steps):
      with tf.variable_scope("dependency_label_strength_{}_{}_{}".format(cnt, direction, step), ):

        dense = tf.keras.layers.Dense(
          num_srl_labels, activation=tf.nn.relu, use_bias=True,
          kernel_initializer='glorot_uniform',
          bias_initializer='zeros'
        )
        tf.logging.log(tf.logging.INFO, "labeled dep -> srl prior: {}".format(dense))
        output = dense(labels)
        output = tf.Print(output, [output], "labeled dep -> srl prior, logit form")
        # output = (output) # treat this one as prior
        # output = tf.Print(output, [output], "labeled dep -> srl prior, log_sigmoid form")

        combined_head_label = heads*output
        tf.logging.log(tf.logging.INFO, "labeled dep -> srl prior output: {}".format(combined_head_label))
        head_strength.append(combined_head_label)
  return head_strength
def selective_gating(transition_mtx):
  with tf.variable_scope("transition_mtx_selective_gating"):
    max_transition_score = tf.reduce_logsumexp(transition_mtx, -1, keep_dims=True)
    bias = tf.get_variable("beta", shape=[1, 1, 1])
    gate = tf.nn.sigmoid(max_transition_score+bias)
    self_looping_map = tf.eye(tf.shape(transition_mtx)[-1], batch_shape=[tf.shape(transition_mtx)[0]]) * 10
    gated_transition_mtx = tf.math.log(gate * tf.nn.softmax(transition_mtx) + (1 - gate) * tf.nn.softmax(self_looping_map))
    return gated_transition_mtx




def generating_decedent_mtx_from_head_label_dist(decedents, labels, num_labels, tokens_to_keep, softmax=tf.nn.softmax, chain_length = 1, score_multiplier = 1., head_label_aggregation_fn = tf.math.multiply, use_strength_bias = False):
  pass
#   """
#     heads: head-dependent distribution of shape (B, seq_len, seq_len), must be masked!
#     labels: label distribution for each head-dependent choice, shape: (B, seq_len, seq_len, labels)
#   """
#   # raise NotImplementedError
#   decedent_strength = []
#   # softmaxed_head = softmax(heads)
#   # decedent_labels = tf.matmul(decedents, labels)
#
#
#   for l in range(chain_length):
#     beta = 0.
#     with tf.variable_scope("dependency_label_strength_{}".format(l), reuse=tf.AUTO_REUSE):
#       s = tf.get_variable("strength", shape=[1, 1, num_labels])
#       if use_strength_bias:
#         beta = tf.get_variable("beta", shape=[1, 1, 1], initializer=tf.zeros_initializer())
#       tf.logging.log(tf.logging.INFO, "strength scope {}".format(s.name))
#     # label_score = tf.nn.sigmoid(tf.math.reduce_sum(s * tf.nn.softmax(decedent_labels), -1, keep_dims=True) + beta)
#     unnormalized_decedent_strength = head_label_aggregation_fn(decedents, 1.)
#                                                            # label_score * score_multiplier)  # (1 - label_score) * 2 * constants.VERY_SMALL
#     unnormalized_decedent_strength = tf.clip_by_value(unnormalized_decedent_strength, clip_value_max=constants.VERY_LARGE,
#                                                   clip_value_min=constants.VERY_SMALL)
#
#     decedent_strength.append(unnormalized_decedent_strength)
#   return decedent_strength
def generating_hard_decedent_mtx_from_head_label_dist(heads, labels, num_labels, tokens_to_keep, softmax=tf.nn.softmax, chain_length = 1):
  pass
#   """
#     heads: head-dependent distribution of shape (B, seq_len, seq_len), must be masked!
#     labels: label distribution for each head-dependent choice, shape: (B, seq_len, seq_len, labels)
#   """
#   # raise NotImplementedError
#   heads_tmp = tf.argmax(heads, -1)
#   heads_tmp = tf.one_hot(heads_tmp, tf.shape(heads_tmp)[-1], off_value=constants.VERY_SMALL,
#                          on_value=constants.VERY_LARGE)
#
#   decedents = tf.transpose(heads_tmp, perm=[0, 2, 1])
#   decedent_labels = tf.transpose(labels, perm = [0, 2, 1, 3])
#   # softmaxed_decedents = softmax(decedents)
#   decedent_strength = []
#   for l in range(chain_length):
#     with tf.variable_scope("dependency_label_strength_{}".format(l), reuse=tf.AUTO_REUSE):
#       s = tf.get_variable("strength", shape=[1, 1, 1, num_labels])
#       tf.logging.log(tf.logging.INFO, "strength scope {}".format(s.name))
#     softmaxed_labels = softmax(decedent_labels)
#     # label_discounting_factor = tf.reduce_sum(s * softmaxed_labels, axis=-1)
#     unnormalized_decedent_strength = decedents + 0.
#
#     mask = tf.cast(1 - tokens_to_keep, tf.float32) * constants.VERY_SMALL
#     unnormalized_decedent_strength = unnormalized_decedent_strength + tf.expand_dims(mask, 1)
#
#     decedent_strength.append(unnormalized_decedent_strength)
#   return decedent_strength

def softmax_with_smoothing(logits, axis=-1, alpha = 0.1):
  # logits_rank = len(logits.get_shape().as_list())
  num_labels = tf.shape(logits)[-1:]
  # print(num_labels)
  # logits = tf.Print(logits, [logits], "logits")
  dist = tf.nn.softmax(logits, axis=axis)
  # dist = tf.Print(dist, [dist], "dist")
  bias = tf.ones_like(dist) * alpha/tf.cast(num_labels[0], dtype=tf.float32)
  # print(1/num_labels)
  # bias *=
  dist *= (1-alpha)
  dist += bias
  return dist
