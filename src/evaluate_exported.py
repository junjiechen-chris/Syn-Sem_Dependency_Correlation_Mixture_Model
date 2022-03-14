import json
from collections import OrderedDict

import tensorflow as tf
import numpy as np
import argparse

from tensorflow.estimator import WarmStartSettings

import train_utils
from vocab import Vocab
import sys
from tensorflow.contrib import predictor
import evaluation_fns_np as eval_fns
import constants
import os
import util
from model import LISAModel


arg_parser = argparse.ArgumentParser(description='')
arg_parser.add_argument('--test_files',
                        help='Comma-separated list of test data files')
arg_parser.add_argument('--dev_files',
                        help='Comma-separated list of development data files')
arg_parser.add_argument('--save_dir', required=True,
                        help='Directory containing saved model')
# todo load this more generically, so that we can have diff stats per task
arg_parser.add_argument('--transition_stats',
                        help='Transition statistics between labels')
arg_parser.add_argument('--debug', dest='debug', action='store_true',
                        help='Whether to run in debug mode: a little faster and smaller')
arg_parser.add_argument('--data_config', required=True,
                        help='Path to data configuration json')
arg_parser.add_argument('--hparams', type=str,
                        help='Comma separated list of "name=value" hyperparameter settings.')
# todo: are these necessary?
arg_parser.add_argument('--model_configs', required=True,
                        help='Comma-separated list of paths to model configuration json.')
arg_parser.add_argument('--task_configs', required=True,
                        help='Comma-separated list of paths to task configuration json.')
arg_parser.add_argument('--layer_configs', required=True,
                        help='Comma-separated list of paths to layer configuration json.')
arg_parser.add_argument('--attention_configs',
                        help='Comma-separated list of paths to attention configuration json.')

arg_parser.add_argument('--ensemble', dest='ensemble', action='store_true',
                        help='Whether to ensemble models in save dir.')
arg_parser.add_argument('--okazaki_discounting', dest='okazaki_discounting', action='store_true',
                        help='whether to use okazaki style of discounting method')

arg_parser.add_argument('--output_attention_weight', dest='output_attention_weight', action='store_true',
                        help='whether to print out attention weight')
arg_parser.add_argument('--parser_dropout', dest='parser_dropout', action='store_true',
                        help='whether to add a dropout layer for parser aggregation')
arg_parser.add_argument('--aggregator_mlp_bn', dest='aggregator_mlp_bn', action='store_true',
                        help='whether to use batch normalization on aggregator mlp')
arg_parser.add_argument('--eval_with_transformation', dest='eval_with_transformation', action='store_true',
                        help='whether to evaluate with result transformation')

arg_parser.add_argument('--output_predictions', dest='output_predictions', action='store_true',
                        help='whether to evaluate with result transformation')



arg_parser.set_defaults(debug=False, ensemble=False)

args, leftovers = arg_parser.parse_known_args()

util.init_logging(tf.logging.INFO)

if not os.path.isdir(args.save_dir):
  util.fatal_error("save_dir not found: %s" % args.save_dir)

# Load all the various configurations
# todo: validate json
data_config = train_utils.load_json_configs(args.data_config)
data_config = OrderedDict(sorted(data_config.items(), key=lambda x: x[1]['conll_idx'] if isinstance(x[1]['conll_idx'], int) else x[1]['conll_idx'][0]))
model_config = train_utils.load_json_configs(args.model_configs)

# if args.eval_with_transformation:
#   args.task_configs = args.task_configs.replace('srl.json', 'srl_only.json')
#   tf.logging.log(tf.logging.INFO, "Now using {} as tasks".format(args.task_configs) )
task_config = train_utils.load_json_configs(args.task_configs, args)
layer_config = train_utils.load_json_configs(args.layer_configs)
attention_config = train_utils.load_json_configs(args.attention_configs)


## Forcing srl_eval to srl_eval_with_transformation



# attention_config = {}
# if args.attention_configs and args.attention_configs != '':
#   attention_config = train_utils.load_json_configs(args.attention_configs)
layer_task_config, layer_attention_config = util.combine_attn_maps(layer_config, attention_config, task_config)

hparams = train_utils.load_hparams(args, model_config)
# hparams.batch_size = 1280

dev_filenames = args.dev_files.split(',')
test_filenames = args.test_files.split(',') if args.test_files and not args.eval_with_transformation else []

vocab = Vocab(data_config, args.save_dir)
vocab.update(test_filenames)

hparams.mode = 'predict'
# print("debug <hparams>:", hparams)


embedding_files = [embeddings_map['pretrained_embeddings'] for embeddings_map in model_config['embeddings'].values()
                   if 'pretrained_embeddings' in embeddings_map]

# Generate mappings from feature/label names to indices in the model_fn inputs
# feature_idx_map = {}
# label_idx_map = {}
feature_idx_map, label_idx_map = util.load_feat_label_idx_maps(data_config)

# create transition parameters if training or decoding with crf/viterbi
# need to load these here for ensembling (they're also loaded by the model)
transition_params = util.load_transition_params(layer_task_config, vocab)

def constrcut_predictor(path):
    # warm_start_from=tf.estimator.WarmStartSettings(
    #     ckpt_to_initialize_from=str(path),
    #    vars_to_warm_start=".*" # everything in TRAINABLE_VARIABLES - excluding optimiser params
    #     # vars_to_warm_start=[".*"], # everything in GLOBAL_VARIABLES - including optimiser params
    # )
    model = LISAModel(hparams, model_config, layer_task_config, layer_attention_config, feature_idx_map, label_idx_map,
                      vocab)
    # ws = WarmStartSettings(ckpt_to_initialize_from=path,
    #                   vars_to_warm_start=".*")
    # print("debug <loading from>:", path)
    estimator = tf.estimator.Estimator(model_fn=model.model_fn, model_dir=path)
    return estimator

if args.ensemble:
  predict_fns = [predictor.from_saved_model("%s/%s" % (args.save_dir, subdir))
                 for subdir in util.get_immediate_subdirectories(args.save_dir)]
else:
  # predict_fns = [predictor.from_saved_model(args.save_dir)]
  estimator = constrcut_predictor(args.save_dir)
  # print("debug <converting estimator to predictor>")
  predict_fns = [predictor.from_estimator(estimator, serving_input_receiver_fn=train_utils.serving_input_receiver_fn)]


def dev_input_fn():
  return train_utils.get_input_fn(vocab, data_config, dev_filenames, hparams.batch_size, num_epochs=1, shuffle=False,
                                  embedding_files=embedding_files, is_token_based_batching = hparams.is_token_based_batching)


def eval_fn(input_op, sess, input_source):
  if args.eval_with_transformation:
    task_config['srl']['eval_fns']['srl_f1']['name'] = 'conll_srl_eval_with_transformation'
    pass

  eval_accumulators = eval_fns.get_accumulators(task_config)
  eval_results = OrderedDict({})
  i = 0
  while True:
    i += 1
    try:
      # input_np = sess.run(dev_input_fn())
      input_np = sess.run(input_op)
      # print("<debug input_np>: {}".format(input_np))
      predictor_input = {'input': input_np}
      predictions = [predict_fn(predictor_input) for predict_fn in predict_fns]

      shape = input_np.shape
      batch_size = shape[0]
      batch_seq_len = shape[1]

      feats = {f: input_np[:, :, idx] for f, idx in feature_idx_map.items()}

      # print(feats.keys())
      tokens_to_keep = np.where(feats['word'] == constants.PAD_VALUE, 0, 1)

      combined_predictions = predictions[0]
      # print(combined_predictions)
      # print([i for i in combined_predictions])

      # todo: implement ensembling
      combined_scores = {k: v for k, v in combined_predictions.items() if k.endswith("_scores")}
      combined_probabilities = {k: v for k, v in combined_predictions.items() if k.endswith("_probabilities")}
      combined_loss = {k: v for k, v in combined_predictions.items() if k.endswith("loss")}

      # for model_outputs in predictions:
      #   for key, val in model_outputs.items():
      #     if key.endswith("_probabilities"):
      #       if key not in combined_probabilities:
      #         print("init", key)
      #         combined_probabilities[key] = val
      #       else:
      #         print("adding ", key)
      #         # product of experts ensembling
      #         if val.shape == combined_probabilities[key].shape:
      #           combined_scores[key] = np.multiply(combined_probabilities[key], val)

      combined_predictions.update({k.replace('scores', 'predictions'): np.argmax(v, axis=-1) for k, v in combined_scores.items()})
      combined_predictions.update({k.replace('probabilities', 'predictions'): np.argmax(v, axis=-1) for k, v in combined_probabilities.items()})

      labels = {}
      for l, idx in label_idx_map.items():
        these_labels = input_np[:, :, idx[0]:idx[0]+1] if idx[1] != -1 else input_np[:, :, idx[0]:]
        these_labels_masked = np.multiply(these_labels, np.expand_dims(tokens_to_keep, -1))
        # check if we need to mask another dimension
        if idx[1] == -1:
          this_mask = np.where(these_labels_masked == constants.PAD_VALUE, 0, 1)
          these_labels_masked = np.multiply(these_labels_masked, this_mask)
        else:
          these_labels_masked = np.squeeze(these_labels_masked, -1)
        labels[l] = these_labels_masked

      # print("debug <combined prediction>: ", combined_predictions)
      for task, tran_params in transition_params.items():
        # print("debug <task & tran_params>:", task, tran_params, tran_params.shape)
        task_predictions = np.empty_like(combined_predictions['%s_predictions' % task])
        token_take_mask = util.get_token_take_mask(task, task_config, combined_predictions, labels)
        # print("task <{}>".format(task), token_take_mask)
        if token_take_mask is not None:
          toks_to_keep_tiled = np.reshape(np.tile(tokens_to_keep, [1, batch_seq_len]),
                                          [batch_size, batch_seq_len, batch_seq_len])
          toks_to_keep_task = toks_to_keep_tiled[np.where(token_take_mask == 1)]
        else:
          toks_to_keep_task = tokens_to_keep
        sent_lens_task = np.sum(toks_to_keep_task, axis=-1)
        if 'srl' in transition_params:
          # print("debug <doing stuff for srl>: ",task)
          # print("debug <doing stuff for srl>: ", combined_scores['%s_scores' % task], sent_lens_task)
          for idx, (sent, sent_len) in enumerate(zip(combined_scores['%s_scores' % task], sent_lens_task)):
            viterbi_sequence, score = tf.contrib.crf.viterbi_decode(sent[:sent_len], tran_params)
            task_predictions[idx, :sent_len] = viterbi_sequence
        combined_predictions['%s_predictions' % task] = task_predictions




      # for i in layer_task_config:
      for task, task_map in task_config.items():
        for eval_name, eval_map in task_map['eval_fns'].items():
          eval_fn_params = eval_fns.get_params(task, eval_map, combined_predictions, feats, labels,
                                               vocab.reverse_maps, tokens_to_keep)
          if eval_map["name"].startswith("conll09_srl_") and eval_map["name"].endswith("srl_only"):# == "conll09_srl_eval_all_srl_only" or eval_map["name"] == "conll09_srl_eval_srl_only":
            # conll09_srl_eval_all_srl_only
            eval_fn_params["input_source"] = input_source
          eval_fn_params['accumulator'] = eval_accumulators[eval_name]
          eval_result = eval_fns.dispatch(eval_map['name'])(**eval_fn_params)
          eval_results[eval_name] = eval_result
    except tf.errors.OutOfRangeError:
      break
  # print(eval_results)
  for k in eval_results.keys():
    if isinstance(eval_results[k], np.ndarray):
      eval_results[k] = eval_results[k].tolist()
  for k in combined_loss.keys():
    if isinstance(combined_loss[k], np.ndarray):
      combined_loss[k] = combined_loss[k].tolist()
  tf.logging.log(tf.logging.INFO, json.dumps(eval_results))



with tf.Session() as sess:

  dev_input_op = dev_input_fn()

  test_input_ops = {}

  for test_file in test_filenames:
      def test_input_fn():
        return train_utils.get_input_fn(vocab, data_config, [test_file], hparams.batch_size, num_epochs=1, shuffle=False,
                                        embedding_files=embedding_files, is_token_based_batching = hparams.is_token_based_batching)
      test_input_ops[test_file] = test_input_fn()

  sess.run(tf.tables_initializer())

  tf.logging.log(tf.logging.INFO, "Evaluating on dev files: %s" % str(dev_filenames))
  eval_fn(dev_input_op, sess, dev_filenames[0])
  for test_file, test_input_op in test_input_ops.items():
      tf.logging.log(tf.logging.INFO, "Evaluating on test file: %s" % str(test_file))
      eval_fn(test_input_op, sess, test_file)

