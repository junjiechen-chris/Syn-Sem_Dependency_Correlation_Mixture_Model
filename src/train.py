from collections import OrderedDict

import tensorflow as tf
import argparse
import os
from functools import partial
import train_utils
from vocab import Vocab
from model import LISAModel
import numpy as np
import sys
import util
from others import EvalResultsExporter
import neptune.new as neptune
# import neptune
from Exporter import BestCheckpointCopier

arg_parser = argparse.ArgumentParser(description='')
arg_parser.add_argument('--train_files', required=True,
                        help='Comma-separated list of training data files')
arg_parser.add_argument('--dev_files', required=True,
                        help='Comma-separated list of development data files')
arg_parser.add_argument('--save_dir', required=True,
                        help='Directory to save models, outputs, etc.')
# todo load this more generically, so that we can have diff stats per task
arg_parser.add_argument('--transition_stats',
                        help='Transition statistics between labels')
arg_parser.add_argument('--hparams', type=str,
                        help='Comma separated list of "name=value" hyperparameter settings.')
arg_parser.add_argument('--use_llisa_proj', action='store_true', default=False,
                        help='Whether to use llisa proj')
arg_parser.add_argument('--neptune_job_name', type=str, default=None,
                        help='custom neptune job name')
arg_parser.add_argument('--debug', dest='debug', action='store_true',
                        help='Whether to run in debug mode: a little faster and smaller')
arg_parser.add_argument('--data_config', required=True,
                        help='Path to data configuration json')
arg_parser.add_argument('--model_configs', required=True,
                        help='Comma-separated list of paths to model configuration json.')
arg_parser.add_argument('--task_configs', required=True,
                        help='Comma-separated list of paths to task configuration json.')
arg_parser.add_argument('--layer_configs', required=True,
                        help='Comma-separated list of paths to layer configuration json.')
arg_parser.add_argument('--attention_configs',
                        help='Comma-separated list of paths to attention configuration json.')
arg_parser.add_argument('--num_gpus', type=int,
                        help='Number of GPUs for distributed training.')
arg_parser.add_argument('--keep_k_best_models', type=int,
                        help='Number of best models to keep.')
arg_parser.add_argument('--best_eval_key', required=True, type=str,
                        help='Key corresponding to the evaluation to be used for determining early stopping.')
arg_parser.add_argument('--early_stopping', type=bool, default=True,
                        help='whether to use early stopping for training -> may lead to unmature stopping')
arg_parser.add_argument('--okazaki_discounting', dest='okazaki_discounting', action='store_true',
                        help='whether to use okazaki style of discounting method')
arg_parser.add_argument('--output_attention_weight', dest='output_attention_weight', action='store_true',
                        help='whether to print out attention weight')
arg_parser.add_argument('--parser_dropout', dest='parser_dropout', action='store_true',
                        help='whether to add a dropout layer for parser aggregation')
arg_parser.add_argument('--aggregator_mlp_bn', dest='aggregator_mlp_bn', action='store_true',
                        help='whether to use batch normalization on aggregator mlp')
arg_parser.add_argument('--attn_debug', dest='attn_debug', action='store_true',
                        help='stub for attn debuging')
arg_parser.set_defaults(debug=False, num_gpus=1, keep_k_best_models=1)

args, leftovers = arg_parser.parse_known_args()

util.init_logging(tf.logging.INFO)

# Load all the various configurations
# todo: validate json
data_config = train_utils.load_json_configs(args.data_config)
# print("debug <combined_config>:", data_config.items())
data_config = OrderedDict(sorted(data_config.items(), key=lambda x: x[1]['conll_idx'] if isinstance(x[1]['conll_idx'], int) else x[1]['conll_idx'][0]))
if not args.debug:
  neptune_handler = None
else:
  neptune_handler = None
# neptune_handler = None



model_config = train_utils.load_json_configs(args.model_configs)
task_config = train_utils.load_json_configs(args.task_configs, args)
# print("debug <task_config>: ", task_config)
layer_config = train_utils.load_json_configs(args.layer_configs)
attention_config = train_utils.load_json_configs(args.attention_configs)

# attention_config = {}
# if args.attention_configs and args.attention_configs != '':
#   attention_config = train_utils.load_json_configs(args.attention_configs)

# Combine layer, task and layer, attention maps
# todo save these maps in save_dir
layer_task_config, layer_attention_config = util.combine_attn_maps(layer_config, attention_config, task_config)

hparams = train_utils.load_hparams(args, model_config, neptune_handler)
## NEED TO REMOVE
if args.attn_debug:
  hparams.attn_debug = True

# Set the random seed. This defaults to int(time.time()) if not otherwise set.
np.random.seed(hparams.random_seed)
tf.set_random_seed(hparams.random_seed)

if not os.path.exists(args.save_dir):
  os.makedirs(args.save_dir)

train_filenames = args.train_files.split(',')
dev_filenames = args.dev_files.split(',')

vocab = Vocab(data_config, args.save_dir, train_filenames)
vocab.update(dev_filenames)

embedding_files = [embeddings_map['pretrained_embeddings'] for embeddings_map in model_config['embeddings'].values()
                   if 'pretrained_embeddings' in embeddings_map]
if "glove_300d" in model_config and hparams.glove_300d:
  embedding_files.append(model_config["glove_300d"]["glove_300d_embeddings"])


def train_input_fn():
  return train_utils.get_input_fn(vocab, data_config, train_filenames, hparams.batch_size,
                                  num_epochs=hparams.num_train_epochs, shuffle=True,
                                  is_token_based_batching = hparams.is_token_based_batching,
                                  embedding_files=embedding_files,
                                  shuffle_buffer_multiplier=hparams.shuffle_buffer_multiplier)


def dev_input_fn():
  return train_utils.get_input_fn(vocab, data_config, dev_filenames, hparams.batch_size,
                                  num_epochs=1, shuffle=False,
                                  embedding_files=embedding_files, is_token_based_batching = hparams.is_token_based_batching)


# Generate mappings from feature/label names to indices in the model_fn inputs
feature_idx_map, label_idx_map = util.load_feat_label_idx_maps(data_config)
# feature_idx_map = {}
# label_idx_map = {}
# for i, f in enumerate([d for d in data_config.keys() if
#                        ('feature' in data_config[d] and data_config[d]['feature']) or
#                        ('label' in data_config[d] and data_config[d]['label'])]):
#   if 'feature' in data_config[f] and data_config[f]['feature']:
#     feature_idx_map[f] = i
#   if 'label' in data_config[f] and data_config[f]['label']:
#     if 'type' in data_config[f] and data_config[f]['type'] == 'range':
#       idx = data_config[f]['conll_idx']
#       j = i + idx[1] if idx[1] != -1 else -1
#       label_idx_map[f] = (i, j)
#     else:
#       label_idx_map[f] = (i, i+1)


# Initialize the model
model = LISAModel(hparams, model_config, layer_task_config, layer_attention_config, feature_idx_map, label_idx_map,
                  vocab)

if args.debug:
  tf.logging.log(tf.logging.INFO, "Created trainable variables: %s" % str([v.name for v in tf.trainable_variables()]))

# Distributed training
distribution = tf.contrib.distribute.MirroredStrategy(num_gpus=args.num_gpus) if args.num_gpus > 1 else None

# Set up the Estimator
checkpointing_config = tf.estimator.RunConfig(save_checkpoints_steps=hparams.eval_every_steps-1, keep_checkpoint_max=3,
                                              train_distribute=distribution, tf_random_seed=hparams.random_seed)
estimator = tf.estimator.Estimator(model_fn=model.model_fn, model_dir=args.save_dir, config=checkpointing_config)

# Set up early stopping -- always keep the model with the best F1
export_assets = {"%s.txt" % vocab_name: "%s/assets.extra/%s.txt" % (args.save_dir, vocab_name)
                 for vocab_name in vocab.vocab_names_sizes.keys()}
srl_early_stop_hook = tf.estimator.experimental.stop_if_no_increase_hook(estimator, 'srl_f1', max_steps_without_increase=24000,  min_steps=hparams.training_min_steps if not args.debug else 40000)
tf.logging.log(tf.logging.INFO, "Exporting assets: %s" % str(export_assets))
save_best_exporter = tf.estimator.BestExporter(compare_fn=partial(train_utils.best_model_compare_fn,
                                                                  key=args.best_eval_key),
                                               serving_input_receiver_fn=train_utils.serving_input_receiver_fn,
                                               assets_extra=export_assets,
                                               exports_to_keep=args.keep_k_best_models)

best_copier = BestCheckpointCopier(
   name='best_checkpoint', # directory within model directory to copy checkpoints to
   checkpoints_to_keep=args.keep_k_best_models, # number of checkpoints to keep
   score_metric=args.best_eval_key, # metric to use to determine "best"
   compare_fn=lambda x,y: x.score>y.score, # comparison function used to determine "best" checkpoint (x is the current checkpoint; y is the previously copied checkpoint with the highest/worst score)
   sort_key_fn=lambda x: x.score,
   sort_reverse=True ,
    neptune_handler = neptune_handler) # sort#keep larger checkpoints

# Train forever until killed
train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, hooks=[srl_early_stop_hook] if args.early_stopping else None)
eval_spec = tf.estimator.EvalSpec(input_fn=dev_input_fn, throttle_secs=hparams.eval_throttle_secs,
                                  exporters=[save_best_exporter, best_copier])

# Run training

# print("debug <confirm vocab predicate content>: ", vocab.vocab_maps['predicate'])

tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
