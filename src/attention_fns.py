from functools import partial

import tensorflow as tf
import constants
import nn_utils
import transformation_fn


def copy_from_predicted(mode, train_attention_to_copy, eval_attention_to_copy):
  attention_to_copy = train_attention_to_copy if mode == tf.estimator.ModeKeys.TRAIN else eval_attention_to_copy

  # check whether this thing is actually scores or if it's predictions, and needs
  # to be expanded out to one-hot scores. If it's actually scores, dims should be
  # batch x batch_seq_len x batch_seq_len, and thus rank should be 3
  if len(attention_to_copy.get_shape()) < 3:
    # use non-standard on and off values because we're going to softmax this later, and want the result to be 0/1
    attention_to_copy = tf.one_hot(attention_to_copy, tf.shape(attention_to_copy)[-1], on_value=constants.VERY_LARGE,
                                   off_value=constants.VERY_SMALL)

  return tf.cast(tf.nn.softmax(attention_to_copy, dim=-1), tf.float32), None

def linear_aggregation(mode, train_attention_aggregation, eval_attention_aggregation, parser_dropout=0.9):
  #suppose attention_to_aggregated is in list
  attention_to_aggregated= train_attention_aggregation if mode == tf.estimator.ModeKeys.TRAIN else eval_attention_aggregation
  attention_to_aggregated, weight = nn_utils.graph_aggregation_softmax_done(attention_to_aggregated, parser_dropout)
  return tf.cast(attention_to_aggregated, tf.float32), weight
def mean_aggregation(mode, train_attention_aggregation, eval_attention_aggregation, parser_dropout=0.9):
  #suppose attention_to_aggregated is in list
  attention_to_aggregated= train_attention_aggregation if mode == tf.estimator.ModeKeys.TRAIN else eval_attention_aggregation
  attention_to_aggregated = nn_utils.graph_mean_aggregation(attention_to_aggregated, parser_dropout)
  return tf.cast(attention_to_aggregated, tf.float32), None

def chain_linear_aggregation(mode, train_attention_aggregation, eval_attention_aggregation, reduction_mode = "sum"):
  #suppose attention_to_aggregated is in list
  attention_to_aggregated= train_attention_aggregation if mode == tf.estimator.ModeKeys.TRAIN else eval_attention_aggregation
  attention_to_aggregated, weight = nn_utils.linear_graph_aggregation(attention_to_aggregated, reduction_mode)
  return tf.cast(attention_to_aggregated, tf.float32), weight



def mean_aggregation_prob(mode, train_attention_aggregation, eval_attention_aggregation, parser_dropout=0.9):
  #suppose attention_to_aggregated is in list
  attention_to_aggregated= train_attention_aggregation if mode == tf.estimator.ModeKeys.TRAIN else eval_attention_aggregation
  attention_to_aggregated = nn_utils.graph_mean_aggregation_prob(attention_to_aggregated, parser_dropout)
  print("attention to aggregate", attention_to_aggregated)
  return tf.cast(attention_to_aggregated, tf.float32), None

def pass_through(mode, train_attention_aggregation, eval_attention_aggregation, parser_dropout=0.9):
  attention_to_aggregated= train_attention_aggregation if mode == tf.estimator.ModeKeys.TRAIN else eval_attention_aggregation
  attention_to_aggregated = tf.squeeze(attention_to_aggregated, axis=[1])
  return tf.cast(attention_to_aggregated, tf.float32), None

def linear_aggregation_by_mlp(mode, train_attention_aggregation, eval_attention_aggregation, v, mlp_dropout, projection_dim, parser_dropout=0.9, batch_norm=False):
  attention_to_aggregated= train_attention_aggregation if mode == tf.estimator.ModeKeys.TRAIN else eval_attention_aggregation
  aggregated_attention, weight = nn_utils.graph_mlp_aggregation(attention_to_aggregated, v, mlp_dropout, projection_dim, parser_dropout, batch_norm)

  # raise NotImplementedError
  return tf.cast(aggregated_attention, tf.float32), weight


dispatcher = {
  'copy_from_predicted': copy_from_predicted,
  'linear_aggregation': linear_aggregation,
  'mean_aggregation': mean_aggregation,
  'mean_aggregation_prob': mean_aggregation_prob,
  'linear_aggregation_mlp': linear_aggregation_by_mlp,
  'chain_linear_aggregation': chain_linear_aggregation,
  'pass_through': pass_through
}


def dispatch(fn_name):
  try:
    return dispatcher[fn_name]
  except KeyError:
    print('Undefined attention function `%s' % fn_name)
    exit(1)

src_param_translator = {
              "parse_bert_benepar": "parse_gold",
              "label_bert_benepar": "parse_label",
              "parse_bert_biaffine": "parse_gold",
              "label_bert_biaffine": "parse_label",
              "parse_label": "parse_label",
              "parse_gold": "parse_gold"
            }
transformation_fn_list = [
  'get_labeled_adjacent_mtx',
  'get_root_up_mtx',
  'get_een_up_mtx',
  'get_void_up_mtx',
  'get_PreColMaskVoid_up_mtx',
  'get_een_kup1down_mtx'
]
def get_params(mode, attn_map, train_outputs, features, labels, hparams, model_config, tokens_to_keep):
  params = {'mode': mode}
  params_map = attn_map['params']
  # if attn_map['name']
  # print("debug <attention fn get parameter>: ", params, params_map, features, labels)
  for param_name, param_values in params_map.items():
    # if this is a map-type param, do map lookups and pass those through


    if 'labels' in param_values:
      input_labels = []
      for label_item in param_values['labels']:
        if isinstance(label_item, dict):
          # outputs = []
          transformation_param = {}
          for src in label_item['sources']:
            #dangerous hack
            if src in labels:
              label = labels[src]
            elif src in features:
              label = features[src]
            # print(src)
            # print(src_param_translator[src])
            transformation_param[src_param_translator[src]] = label
          transformation_param['tokens_to_keep'] = tokens_to_keep
          if "transformation_fn" in label_item:
            transformation_fn_name = label_item['transformation_fn']
            if hparams.use_hparams_transformation_fn:
              transformation_fn_name = hparams.transformation_fn
            print("transformation_fn_name:", transformation_fn_name)

            # transformation_param['transformation_name'] = transformation_fn
            if transformation_fn_name.startswith('get_hsdp_adjacent_mtx'):
              func_name = transformation_fn_name[:21]
              transformation_param['chain'] = transformation_fn_name.split('_')[4:]
            elif transformation_fn_name.startswith('get_k_adjacent_mtx_hard'):
              func_name = transformation_fn_name[:23]
            elif transformation_fn_name.startswith('get_k_adjacent_mtx'):
              func_name = transformation_fn_name[:18]
            elif transformation_fn_name.startswith('get_labeled_adjacent_mtx_add'):
              func_name = transformation_fn_name[:28]
              transformation_param['chain'] = transformation_fn_name.split('_')[5:]

              if hparams.is_ud and hparams.conll05:
                transformation_param['num_labels'] = 39
              elif not hparams.is_ud and hparams.conll05:
                transformation_param['num_labels'] = 44
              elif not hparams.is_ud and not hparams.conll05:
                transformation_param['num_labels'] = 45
              else:
                tf.logging.log("unknown dataset format")
                raise ValueError
              if hparams.relchain_softmax_smoothing:
                transformation_param['smoothing_softmax'] = True
            elif '_'.join(transformation_fn_name.split('_')[:4]) in transformation_fn_list:
              func_name = '_'.join(transformation_fn_name.split('_')[:4])
              transformation_param['chain'] = transformation_fn_name.split('_')[4:]
              if hparams.is_ud and hparams.conll05:
                transformation_param['num_labels'] = 39
              elif not hparams.is_ud and hparams.conll05:
                transformation_param['num_labels'] = 44
              elif not hparams.is_ud and hparams.conll12:
                transformation_param['num_labels'] = 45
              elif not hparams.is_ud and hparams.conll09:
                transformation_param['num_labels'] = 69
              else:
                tf.logging.log("unknown dataset format")
                raise ValueError
              head_gen_fn = nn_utils.generating_head_mtx_from_head_label_dist
              if "head_label_aggregation" in param_values or hparams.use_labeled_adjacency_mtx_hparams_option:
                if not "head_label_aggregation" in param_values:
                  head_label_aggregation = hparams.head_label_aggregation
                else:
                  head_label_aggregation = param_values["head_label_aggregation"]
                head_gen_fn = partial(head_gen_fn, head_label_aggregation=head_label_aggregation)
              if "label_score_multiplier" in param_values or hparams.use_labeled_adjacency_mtx_hparams_option:
                if not "label_score_multiplier" in param_values:
                  multiplier = hparams.label_score_multiplier
                else:
                  multiplier = param_values["label_score_multiplier"]
                head_gen_fn = partial(head_gen_fn, ls_multiplier=multiplier)
              if "label_score_aggregation" in param_values or hparams.use_labeled_adjacency_mtx_hparams_option:
                if not "label_score_aggregation" in param_values:
                  label_score_aggregation = hparams.label_score_aggregation
                else:
                  label_score_aggregation = param_values["label_score_aggregation"]
                head_gen_fn = partial(head_gen_fn, label_score_aggregation=label_score_aggregation)
              if "use_strength_bias" in param_values or hparams.use_labeled_adjacency_mtx_hparams_option:
                if not "use_strength_bias" in param_values:
                  use_strength_bias = hparams.use_strength_bias
                else:
                  use_strength_bias = param_values["use_strength_bias"]
                head_gen_fn = partial(head_gen_fn, use_strength_bias=use_strength_bias)
              if "head_dropout_rate" in param_values or hparams.use_hparams_head_dropout_option:
                transformation_param['head_dropout'] = True
                if not "head_dropout_rate" in param_values:
                  dropout = hparams.head_dropout_rate
                else:
                  dropout = param_values['head_dropout_rate']
                transformation_param['head_dropout_rate'] = dropout
              if "head_replacing" in param_values or hparams.use_hparams_head_replacing_option:
                transformation_param['head_replacing'] = True
                if not "head_replacing" in param_values:
                  rate = hparams.head_dropout
                else:
                  rate = param_values['head_replacing_rate']
                transformation_param['head_replacing_rate'] = rate
              if "label_replacing" in param_values or hparams.use_hparams_label_replacing_option:
                transformation_param['label_replacing'] = True
                if not "label_replacing" in param_values:
                  rate = hparams.label_dropout
                else:
                  rate = param_values['label_replacing_rate']
                transformation_param['label_replacing_rate'] = rate
              if "on_prob" in param_values or hparams.use_hparams_on_prob_option:
                if not "on_prob" in param_values:
                  rate = hparams.on_prob
                else:
                  rate = param_values['on_prob']
                transformation_param['on_prob'] = rate
              if "prod_mode" in param_values or hparams.use_hparams_prod_mode_option:
                if not "prod_mode" in param_values:
                  mode = hparams.prod_mode
                else:
                  mode = param_values["prod_mode"]
                transformation_param['prod_mode'] = mode
              transformation_param['new_masking'] = hparams.new_masking
              transformation_param['extreme_value'] = hparams.extreme_value
              transformation_param['head_gen_fn_train'] = head_gen_fn
              transformation_param['head_gen_fn_eval'] = head_gen_fn
            elif transformation_fn_name.startswith('get_adjacent_mtx') and hparams.transformation_fn_norm == "none" and hparams.using_log_prob == True:
              func_name = "get_adjacent_mtx_nonenorm_logprob"
              transformation_param['chain'] = transformation_fn_name.split('_')[3:]
            elif transformation_fn_name.startswith('get_adjacent_mtx'):
              func_name = transformation_fn_name[:16]
              transformation_param['chain'] = transformation_fn_name.split('_')[3:]
            else:
              func_name = transformation_fn
            if 'allow_intermediate_nodes' in param_values:
              transformation_param['allow_intermediate_nodes'] = param_values['allow_intermediate_nodes']

            output = transformation_fn.dispatch(func_name)(**transformation_param)
            input_labels += [output]
            # params[param_name] = tf.expand_dims(output, 1)
          else:
            tf.logging.log(tf.logging.ERROR, "no transformation function specified")
            raise NotImplementedError
        else:
          tf.logging.log(tf.logging.ERROR, "output entry must be a dict")
          raise NotImplementedError
      params[param_name] = tf.stack(input_labels, axis=1)
      if 'reduction_mode' in param_values:
        if hparams.use_hparams_aggregation_reduction_mode:
          params['reduction_mode'] = hparams.aggregation_reduction_mode
        else:
          params['reduction_mode'] = param_values['reduction_mode']
    elif 'label' in param_values:
      if isinstance(param_values['label'], list): # only for compatability reason
        params[param_name] = tf.stack([labels[src] for src in param_values['label']], axis=1)
      elif isinstance(param_values['label'], dict): # only for compatability reason
        params[param_name] = tf.stack([transformation_fn.dispatch(transformation_fn_name)(labels[src]) for src, transformation_fn_name in param_values['label'].items()], axis=1)
      elif isinstance(param_values['label'], str): # only for compatability reason
        params[param_name] = labels[param_values['label']]
      else:
        print('Undefined attention source format')
        raise NotImplementedError
        # todo sentence feature may be invoked by non-aggregation attentions
      params['parser_dropout'] = hparams.parser_dropout
      if hparams.aggregator_mlp_bn:
        params['batch_norm'] = True
      if 'sentence_feature' in param_values:
        params['mlp_dropout'] = hparams['mlp_dropout']
        params['projection_dim'] = model_config['linear_aggregation_scorer_mlp_size']
        params['v'] = features['sentence_feature']
    elif 'output' in param_values:
      if isinstance(param_values['output'], dict): # only for compatability reason
        # outputs_layer = train_outputs[param_values['layer']]
        # params[param_name] = outputs_layer[param_values['output']]
        params[param_name] = tf.stack([train_outputs[layer][output_name] for layer, output_name in param_values['output'].items()], axis=1)
      # elif isinstance(param_values['output'], str): # only for compatability reason
      #   params[param_name] = labels[param_values['label']]
      else:
        print('Undefined attention source format')
        raise NotImplementedError
        # todo sentence feature may be invoked by non-aggregation attentions
      params['parser_dropout'] = hparams.parser_dropout
      if hparams.aggregator_mlp_bn:
        params['batch_norm'] = True
      if 'sentence_feature' in param_values:
        params['mlp_dropout'] = hparams['mlp_dropout']
        params['projection_dim'] = model_config['linear_aggregation_scorer_mlp_size']
        params['v'] = features['sentence_feature']
    elif 'outputs' in param_values:
      outputs = []
      for output_item in param_values['outputs']:
        if isinstance(output_item, dict):
          # outputs = []
          transformation_param = {}
          for layer_name, output_name in output_item['sources'].items():
            outputs_layer = train_outputs[layer_name]
            output = outputs_layer[output_name]
            transformation_param['{}'.format(layer_name)] = output
          transformation_param['tokens_to_keep'] = tokens_to_keep
          if "transformation_fn" in output_item:
            transformation_fn_name = output_item['transformation_fn']
            if hparams.use_hparams_transformation_fn:
              transformation_fn_name = hparams.transformation_fn
            if transformation_fn_name.startswith('get_hsdp_adjacent_mtx'):
              func_name = transformation_fn_name[:21]
              transformation_param['chain'] = transformation_fn_name.split('_')[4:]
            elif transformation_fn_name.startswith('get_k_adjacent_mtx_hard'):
              func_name = transformation_fn_name[:23]
            elif transformation_fn_name.startswith('get_k_adjacent_mtx'):
              func_name = transformation_fn_name[:18]
            elif transformation_fn_name.startswith('get_labeled_adjacent_mtx_add'):
              func_name = transformation_fn_name[:28]
              transformation_param['chain'] = transformation_fn_name.split('_')[5:]
              if hparams.is_ud and hparams.conll05:
                transformation_param['num_labels'] = 39
              elif not hparams.is_ud and hparams.conll05:
                transformation_param['num_labels'] = 44
              elif not hparams.is_ud and hparams.conll12:
                transformation_param['num_labels'] = 45
              elif not hparams.is_ud and hparams.conll09:
                transformation_param['num_labels'] = 69
              else:
                tf.logging.log("unknown dataset format")
                raise ValueError
              if hparams.relchain_softmax_smoothing:
                transformation_param['smoothing_softmax'] = True
            elif '_'.join(transformation_fn_name.split('_')[:4]) in transformation_fn_list:
              func_name = '_'.join(transformation_fn_name.split('_')[:4])
              transformation_param['chain'] = transformation_fn_name.split('_')[4:]
              if hparams.is_ud and hparams.conll05:
                transformation_param['num_labels'] = 39
              elif not hparams.is_ud and hparams.conll05:
                transformation_param['num_labels'] = 44
              elif not hparams.is_ud and hparams.conll12:
                transformation_param['num_labels'] = 45
              elif not hparams.is_ud and hparams.conll09:
                transformation_param['num_labels'] = 69
              else:
                tf.logging.log("unknown dataset format")
                raise ValueError
              head_gen_fn = nn_utils.generating_head_mtx_from_head_label_dist
              if "head_label_aggregation" in param_values or hparams.use_labeled_adjacency_mtx_hparams_option:
                if not "head_label_aggregation" in param_values:
                  head_label_aggregation = hparams.head_label_aggregation
                else:
                  head_label_aggregation = param_values["head_label_aggregation"]
                head_gen_fn = partial(head_gen_fn, head_label_aggregation=head_label_aggregation)
              if "label_score_multiplier" in param_values or hparams.use_labeled_adjacency_mtx_hparams_option:
                if not "label_score_multiplier" in param_values:
                  multiplier = hparams.label_score_multiplier
                else:
                  multiplier = param_values["label_score_multiplier"]
                head_gen_fn = partial(head_gen_fn, ls_multiplier=multiplier)
              if "label_score_aggregation" in param_values or hparams.use_labeled_adjacency_mtx_hparams_option:
                if not "label_score_aggregation" in param_values:
                  label_score_aggregation = hparams.label_score_aggregation
                else:
                  label_score_aggregation = param_values["label_score_aggregation"]
                head_gen_fn = partial(head_gen_fn, label_score_aggregation=label_score_aggregation)
              if "use_strength_bias" in param_values or hparams.use_labeled_adjacency_mtx_hparams_option:
                if not "use_strength_bias" in param_values:
                  use_strength_bias = hparams.use_strength_bias
                else:
                  use_strength_bias = param_values["use_strength_bias"]
                head_gen_fn = partial(head_gen_fn, use_strength_bias=use_strength_bias)
              if "head_dropout_rate" in param_values or hparams.use_hparams_head_dropout_option:
                transformation_param['head_dropout'] = True
                if not "head_dropout_rate" in param_values:
                  dropout = hparams.head_dropout_rate
                else:
                  dropout = param_values['head_dropout_rate']
                transformation_param['head_dropout_rate'] = dropout
              if "head_replacing" in param_values or hparams.use_hparams_head_replacing_option:
                transformation_param['head_replacing'] = True
                if not "head_replacing" in param_values:
                  rate = hparams.head_dropout
                else:
                  rate = param_values['head_replacing_rate']
                transformation_param['head_replacing_rate'] = rate
              if "label_replacing" in param_values or hparams.use_hparams_label_replacing_option:
                transformation_param['label_replacing'] = True
                if not "label_replacing" in param_values:
                  rate = hparams.label_dropout
                else:
                  rate = param_values['label_replacing_rate']
                transformation_param['label_replacing_rate'] = rate
              if "on_prob" in param_values or hparams.use_hparams_on_prob_option:
                if not "on_prob" in param_values:
                  rate = hparams.on_prob
                else:
                  rate = param_values['on_prob']
                transformation_param['on_prob'] = rate
              if "prod_mode" in param_values or hparams.use_hparams_prod_mode_option:
                if not "prod_mode" in param_values:
                  mode = hparams.prod_mode
                else:
                  mode = param_values["prod_mode"]
                transformation_param['prod_mode'] = mode
              transformation_param['new_masking'] = hparams.new_masking
              transformation_param['extreme_value'] = hparams.extreme_value
              transformation_param['head_gen_fn_train'] = head_gen_fn
              transformation_param['head_gen_fn_eval'] = head_gen_fn
            elif transformation_fn_name.startswith('get_hsdp_labeled_adjacent_mtx'):
              func_name = transformation_fn_name[:29]
              transformation_param['chain'] = transformation_fn_name.split('_')[5:]
              if hparams.is_ud and hparams.conll05:
                transformation_param['num_labels'] = 39
              elif not hparams.is_ud and hparams.conll05:
                transformation_param['num_labels'] = 44
              elif not hparams.is_ud and not hparams.conll05:
                transformation_param['num_labels'] = 45
              else:
                tf.logging.log("unknown dataset format")
                raise ValueError
              if hparams.relchain_softmax_smoothing:
                transformation_param['smoothing_softmax'] = True
            elif transformation_fn_name.startswith('get_adjacent_mtx') and hparams.transformation_fn_norm == "none" and hparams.using_log_prob == True:
              func_name = "get_adjacent_mtx_nonenorm_logprob"
              transformation_param['chain'] = transformation_fn_name.split('_')[3:]
            elif transformation_fn_name.startswith('get_adjacent_mtx'):
              func_name = transformation_fn_name[:16]
              transformation_param['chain'] = transformation_fn_name.split('_')[3:]
            else:
              func_name = transformation_fn_name
            if 'allow_intermediate_nodes' in param_values:
              transformation_param['allow_intermediate_nodes'] = param_values['allow_intermediate_nodes']

            tf.logging.log(tf.logging.INFO, "Using {} transformation function".format(func_name))
            output = transformation_fn.dispatch(func_name)(**transformation_param)
            print("transformed_output:", output)
            outputs += [output]
            # params[param_name] = tf.expand_dims(output, 1)
          else:
            tf.logging.log(tf.logging.ERROR, "no transformation function specified")
            raise NotImplementedError
        else:
          tf.logging.log(tf.logging.ERROR, "output entry must be a dict")
          raise NotImplementedError
      params[param_name] = tf.stack(outputs, axis=1)

      # params['parser_dropout'] = hparams.parser_dropout
      if hparams.aggregator_mlp_bn:
        params['batch_norm'] = True
    elif 'feature' in param_values:
      if isinstance(param_values['feature'], dict):
        attn_constraints = []
        for src, transformation_name in param_values['feature'].items():
          attn_map = transformation_fn.dispatch(transformation_name)(
            **transformation_fn.get_params(features[src], transformation_name, src, tokens_to_keep))
          attn_constraints += [attn_map]
        params[param_name] = tf.stack(attn_constraints, axis=1)
      elif isinstance(param_values['feature'], list):  # only for compatability reason
        params[param_name] = tf.stack([features[src] for src in param_values['feature']], axis=1)
      elif isinstance(param_values['feature'], str):  # only for compatability reason
        params[param_name] = features[param_values['label']]
      else:
        print('Undefined attention source format')
        raise NotImplementedError
      # todo sentence feature may be invoked by non-aggregation attentions
      params['parser_dropout'] = hparams.parser_dropout
      if hparams.aggregator_mlp_bn:
        params['batch_norm'] = True
      if 'sentence_feature' in param_values:
        params['mlp_dropout'] = hparams.mlp_dropout
        params['projection_dim'] = model_config['linear_aggregation_scorer_mlp_size']
        params['v'] = features['sentence_feature']
    elif 'layer' in param_values:
      outputs_layer = train_outputs[param_values['layer']]
      params[param_name] = outputs_layer[param_values['output']]
    else:
      params[param_name] = param_values['value']
  # print("debug <attention fn parameters>: ", params)
  return params
