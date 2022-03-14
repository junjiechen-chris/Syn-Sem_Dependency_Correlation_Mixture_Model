from collections import OrderedDict

import tensorflow as tf
import json
import re
import sys
import dataset
import constants
from pathlib import Path
import numpy as np


def load_hparams(args, model_config, neptune_handler = None):
  # Create a HParams object specifying the names and values of the model hyperparameters
  skip_list = [
    'use_labeled_adjacency_mtx_hparams_option'
  ]



  hparams = tf.contrib.training.HParams(**constants.hparams)

  # First get default hyperparams from the model config
  if 'hparams' in model_config:
    hparams.override_from_dict(model_config['hparams'])

  if args.debug:
    hparams.set_hparam('shuffle_buffer_multiplier', 1)
    hparams.set_hparam('eval_throttle_secs', 60)
    hparams.set_hparam('eval_every_steps', 1000)
    hparams.set_hparam('learning_rate', 0.06)
    hparams.set_hparam('batch_size', 4000)
    hparams.set_hparam('num_train_epochs', 3)

    # hparams.set_hparam('debug', True)

  if args.okazaki_discounting:
    hparams.set_hparam('special_attention_mode', 'okazaki_discounting')

  if args.output_attention_weight:
    hparams.set_hparam('output_attention_weight', True)

  if args.parser_dropout:
    hparams.set_hparam('parser_dropout', 0.9)

  if args.aggregator_mlp_bn:
    hparams.set_hparam('aggregator_mlp_bn', True)

  # Override those with command line hyperparams
  if neptune_handler is not None:
    neptune_handler["sys/tags"].add(["universal" if hparams.is_ud else "stanford"])


  # neptune_handler["other_hparams"] = hparams
  if args.hparams:
    if neptune_handler is not None:
      list_hp = args.hparams.split(',')
      if list_hp[-1] == "":
        list_hp = list_hp[-1]
      list_hp_nv = [item.split('=') for item in list_hp]
      for item, nv in zip(list_hp, list_hp_nv):
        name, value = nv
        if name in skip_list:
          continue
        neptune_handler["sys/tags"].add(item)
    hparams.parse(args.hparams)

  tf.logging.log(tf.logging.INFO, "Using hyperparameters: %s" % str(hparams.values()))

  return hparams


def get_input_fn(vocab, data_config, data_files, batch_size, num_epochs, shuffle, is_token_based_batching,
                 shuffle_buffer_multiplier=1, embedding_files=None):
  # this needs to be created from here (lazily) so that it ends up in the same tf.Graph as everything else
  vocab_lookup_ops = vocab.create_vocab_lookup_ops(embedding_files)
  # print("debug <create vocab_lookup ops>: ", vocab_lookup_ops)
  return dataset.get_data_iterator(data_files, data_config, vocab_lookup_ops, batch_size, num_epochs, shuffle,
                                   shuffle_buffer_multiplier, is_token_based_batching = is_token_based_batching)


def load_json_configs(config_file_list, args=None):
  """
  Loads a list of json configuration files into one combined map. Configuration files
  at the end of the list take precedence over earlier configuration files (so they will
  overwrite earlier configs!)

  If args is passed, then this function will attempt to replace entries surrounded with
  the special tokens ## ## with an entry from args with the same name.

  :param config_file_list: list of json configuration files to load
  :param args: command line args to replace special strings in json
  :return: map containing combined configurations
  """
  combined_config = OrderedDict({})
  if config_file_list:
    config_files = config_file_list.split(',')
    print(config_files)
    # print("debug <config file list>: ", config_files)
    for config_file in config_files:
      if args:
        # read the json in as a string so that we can run a replace on it
        json_str = Path(config_file).read_text()
        matches = re.findall(r'.*##(.*)##.*', json_str)
        for match in matches:
          try:
            value = getattr(args, match)
            json_str = json_str.replace('##%s##' % match, value)
          except AttributeError:
            tf.logging.log(tf.logging.ERROR, 'Could not find "%s" attribute in command line args when parsing: %s' %
                           (match, config_file))
            sys.exit(1)
        try:
          config = json.loads(json_str)
        except json.decoder.JSONDecodeError as e:
          tf.logging.log(tf.logging.ERROR, 'Error reading json: "%s"' % config_file)
          tf.logging.log(tf.logging.ERROR, e.msg)
          sys.exit(1)
      else:
        with open(config_file) as f:
          try:
            config = json.load(f, object_pairs_hook=OrderedDict)
          except json.decoder.JSONDecodeError as e:
            tf.logging.log(tf.logging.ERROR, 'Error reading json: "%s"' % config_file)
            tf.logging.log(tf.logging.ERROR, e.msg)
            sys.exit(1)
      # print("debug <input configs>:", config)
      combined_config.update(config)#OrderedDict({**combined_config, **config})
    # print("debug <combined_config>:", combined_config)
  # return OrderedDict(sorted(combined_config.items(), key=lambda x: x[1]['conll_idx'] if isinstance(x[1]['conll_idx'], int) else x[1]['conll_idx'][0]))
    return combined_config

def copy_without_dropout(hparams):
  new_hparams = {k: (1.0 if 'dropout' in k else v) for k, v in hparams.values().items()}
  return tf.contrib.training.HParams(**new_hparams)


def get_vars_for_moving_average(average_norms):
  vars_to_average = tf.trainable_variables()
  if not average_norms:
    vars_to_average = [v for v in tf.trainable_variables() if 'norm' not in v.name]
  tf.logging.log(tf.logging.INFO, "Creating moving averages for %d variables." % len(vars_to_average))
  return vars_to_average


def learning_rate(hparams, global_step):
  # print("<debug global step>:", global_step)
  lr = hparams.learning_rate
  warmup_steps = hparams.warmup_steps
  decay_rate = hparams.decay_rate
  if warmup_steps > 0:

    # add 1 to global_step so that we start at 1 instead of 0
    global_step_float = tf.cast(global_step, tf.float32) + 1.
    lr *= tf.minimum(tf.rsqrt(global_step_float),
                     tf.multiply(global_step_float, warmup_steps ** -decay_rate))
    return lr
  else:
    decay_steps = hparams.decay_steps
    if decay_steps > 0:
      return lr * decay_rate ** (global_step / decay_steps)
    else:
      return lr


def best_model_compare_fn(best_eval_result, current_eval_result, key):
  """Compares two evaluation results and returns true if the second one is greater.
    Both evaluation results should have the value for key, used for comparison.
    Args:
      best_eval_result: best eval metrics.
      current_eval_result: current eval metrics.
      key: key to value used for comparison.
    Returns:
      True if the loss of current_eval_result is smaller; otherwise, False.
    Raises:
      ValueError: If input eval result is None or no loss is available.
    """

  if not best_eval_result or key not in best_eval_result:
    raise ValueError('best_eval_result cannot be empty or key "%s" is not found.' % key)

  if not current_eval_result or key not in current_eval_result:
    raise ValueError('best_eval_result cannot be empty or key "%s" is not found.' % key)

  return best_eval_result[key] < current_eval_result[key]


def serving_input_receiver_fn():
  inputs = tf.placeholder(tf.int32, [None, None, None])
  return tf.estimator.export.TensorServingInputReceiver(inputs, inputs)

def load_dep_pattern(fn, num_labels = 69):
  path = "config/dep_pattern_config/{}".format(fn)
  with open(path) as f:
    lines = f.readlines()
    lines = [map(int, line.split(',')) for line in lines]
  np_weights = []
  for line in lines:
    weight = np.zeros(num_labels)
    for item in line:
      weight[item] = 1.
    np_weights.append(weight)
  return np.stack(np_weights,axis=0)

