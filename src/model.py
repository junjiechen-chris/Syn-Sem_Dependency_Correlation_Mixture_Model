import numpy as np
import tensorflow as tf
from tensorflow.estimator import ModeKeys
import constants
import evaluation_fns
import attention_fns
# import lib
import value_fns
import output_fns
import transformer
import nn_utils
import train_utils
import tf_utils
import util
from lazy_adam_v2 import LazyAdamOptimizer

glove_300d_handler = []
class LISAModel:

  def __init__(self, hparams, model_config, task_config, attention_config, feature_idx_map, label_idx_map,
               vocab, not_load_transition=False):
    self.train_hparams = hparams
    self.test_hparams = train_utils.copy_without_dropout(hparams)

    self.model_config = model_config
    self.task_config = task_config
    self.attention_config = attention_config
    self.feature_idx_map = feature_idx_map
    self.label_idx_map = label_idx_map
    self.vocab = vocab
    self.not_load_transition = not_load_transition

    if hparams.cwr != 'None':
      cwr_embeddings = []
      for embedding_name, embedding_map in self.model_config['cached_cwr'].items():
        embedding_dim = embedding_map['embedding_dim']
        input_cached_embeddings = embedding_map['cached_embeddings']
        embedding_table = self.get_cached_embedding_table(embedding_name, embedding_dim,
                                                          pretrained_fname=input_cached_embeddings, cwr_type=hparams.cwr)
        cwr_embeddings.append(embedding_table)
        tf.logging.log(tf.logging.INFO, "Created cached embeddings for '%s'." % embedding_name)
      self.cwr_embedding = np.concatenate(cwr_embeddings, axis=0)
      if hparams.cwr_ood:
        # normal_init uses mu=0, sigma=0.05
        print("cwr_emb shape", self.cwr_embedding.shape)
        self.cwr_embedding = np.concatenate([self.cwr_embedding, 0.05 * np.random.randn(1, 3072)], axis=0)
    if hparams.glove_300d:
      embedding_map = self.model_config['glove_300d']
      # embedding_dim = embedding_map['embedding_dim']
      input_cached_embeddings = embedding_map['glove_300d_embeddings']
      self.glove_300d_embeddings = util.load_pretrained_embeddings(input_cached_embeddings)
      tf.logging.log(tf.logging.INFO, "Created cached embeddings for '%s'." % "glove 300 d")
      pass
    # print("debug <LISA task config>: ", task_config)

  def hparams(self, mode):
    if mode == ModeKeys.TRAIN:
      return self.train_hparams
    return self.test_hparams

  def get_embedding_table(self, name, embedding_dim, include_oov, pretrained_fname=None, num_embeddings=None, cwr_ood=False):
    if False:
      pass
    else:
      with tf.variable_scope("%s_embeddings" % name):
        initializer = tf.random_normal_initializer()
        if pretrained_fname:
          pretrained_embeddings = util.load_pretrained_embeddings(pretrained_fname)

          pretrained_num_embeddings, pretrained_embedding_dim = pretrained_embeddings.shape
          if pretrained_embedding_dim != embedding_dim:
            util.fatal_error("Pre-trained %s embedding dim does not match specified dim (%d vs %d)." %
                             (name, pretrained_embedding_dim, embedding_dim))
          if num_embeddings and num_embeddings != pretrained_num_embeddings:
            util.fatal_error("Number of pre-trained %s embeddings does not match specified "
                             "number of embeddings (%d vs %d)." % (name, pretrained_num_embeddings, num_embeddings))
          num_embeddings = pretrained_num_embeddings
          if not cwr_ood:
            num_embeddings-=1
            pretrained_embeddings = pretrained_embeddings[:-1, :]
          initializer = tf.constant_initializer(pretrained_embeddings)

        embedding_table = tf.get_variable(name="embeddings", shape=[num_embeddings, embedding_dim],
                                          initializer=initializer)

        if include_oov:
          oov_embedding = tf.get_variable(name="oov_embedding", shape=[1, embedding_dim],
                                          initializer=tf.random_normal_initializer())
          embedding_table = tf.concat([embedding_table, oov_embedding], axis=0,
                                      name="embeddings_table")

        return embedding_table

  def get_cached_embedding_table(self, name, embedding_dim, pretrained_fname=None, num_embeddings=None, cwr_type = None):
    with tf.variable_scope("%s_cached_embeddings" % name):
      pretrained_embeddings = util.load_cached_pretrained_embedding(pretrained_fname, cwr_type)
      return pretrained_embeddings

  def model_fn(self, features, mode):

    # todo can estimators handle dropout for us or do we need to do it on our own?
    hparams = self.hparams(mode)
    tf.logging.log(tf.logging.INFO, "Running in {} mode.".format(mode))


    with tf.variable_scope("LISA", reuse=tf.AUTO_REUSE):
      # features = tf.Print(features, [features, tf.shape(features)], 'input features')
      batch_shape = tf.shape(features)
      batch_size = batch_shape[0]
      batch_seq_len = batch_shape[1]
      layer_config = self.model_config['layers']
      sa_hidden_size = layer_config['head_dim'] * layer_config['num_heads']

      feats = {f: features[:, :, idx] for f, idx in self.feature_idx_map.items()}

      # todo this assumes that word_type is always passed in
      words = feats['word_type']
      # print("debug <input features>:", features)

      # for masking out padding tokens
      tokens_to_keep = tf.where(tf.equal(words, constants.PAD_VALUE), tf.zeros([batch_size, batch_seq_len]),
                                tf.ones([batch_size, batch_seq_len]))

      # Extract named features from monolithic "features" input
      feats = {f: tf.multiply(tf.cast(tokens_to_keep, tf.int32), v) for f, v in feats.items()}
      # feats = {f: tf.Print(feats[f], [feats[f]]) for f in feats.keys()}
      # print("<debug features>: ",feats)
      # print("debug <model_config>:", self.model_config)

      # Extract named labels from monolithic "features" input, and mask them
      # todo fix masking -- is it even necessary?
      labels = {}
      for l, idx in self.label_idx_map.items():
        # print("debug <label_idx_map idx>: ", idx)
        these_labels = features[:, :, idx[0]:idx[0]+1] if idx[1] != -1 else features[:, :, idx[0]:]
        these_labels_masked = tf.multiply(these_labels, tf.cast(tf.expand_dims(tokens_to_keep, -1), tf.int32))
        # check if we need to mask another dimension
        # these_labels_masked_print = tf.Print(these_labels_masked, [tf.shape(these_labels_masked), these_labels_masked, idx],
        #                                'thses labels masked')
        if idx[1] == -1:
          last_dim = tf.shape(these_labels)[2]
          this_mask = tf.where(tf.equal(these_labels_masked, constants.PAD_VALUE),
                               tf.zeros([batch_size, batch_seq_len, last_dim], dtype=tf.int32),
                               tf.ones([batch_size, batch_seq_len, last_dim], dtype=tf.int32))
          these_labels_masked = tf.multiply(these_labels_masked, this_mask)
        else:
          these_labels_masked = tf.squeeze(these_labels_masked, -1, name='these_labels_masked_squeezing')
          # these_labels_masked = tf.Print(these_labels_masked, [tf.shape(these_labels_masked), these_labels_masked], 'thses labels masked after squeezed')
        labels[l] = these_labels_masked
        # labels = [tf.Print(l, [l]) for l in labels]

      # labels = [tf.Print("label_l", l)for l in labels]
      # load transition parameters
      transition_stats = util.load_transition_params(self.task_config, self.vocab, hparams.train_with_crf)

      # Create embeddings tables, loading pre-trained if specified

      embeddings = {}
      for embedding_name, embedding_map in self.model_config['embeddings'].items():
        embedding_dim = embedding_map['embedding_dim']
        if 'pretrained_embeddings' in embedding_map:
          input_pretrained_embeddings = embedding_map['pretrained_embeddings']
          include_oov = True
          embedding_table = self.get_embedding_table(embedding_name, embedding_dim, include_oov,
                                                     pretrained_fname=input_pretrained_embeddings, cwr_ood=hparams.cwr_ood)
        else:
          num_embeddings = self.vocab.vocab_names_sizes[embedding_name]
          embedding_table = self.get_embedding_table(embedding_name, embedding_dim, include_oov,
                                                     num_embeddings=num_embeddings)
        # embedding_table = tf.Print(embedding_table, [tf.shape(embedding_table), embedding_table])
        embeddings[embedding_name] = embedding_table

        tf.logging.log(tf.logging.INFO, "Created embeddings for '%s'." % embedding_name)
        tf.logging.log(tf.logging.INFO, embeddings[embedding_name])

      inputs_list = []
      gp_embs = []
      with tf.device("CPU:0"):
        if hparams.cwr != "None" or hparams.glove_300d:
          if hparams.cwr != "None":
            cached_cwr_embeddings = tf.get_variable("cwr_embedding", shape=self.cwr_embedding.shape, trainable=False)
          def init_fn(scaffold, sess):
            if hparams.cwr != "None":
              sess.run(cached_cwr_embeddings.initializer, {cached_cwr_embeddings.initial_value: self.cwr_embedding})
          scaffold = tf.train.Scaffold(init_fn=init_fn)


        for input_name, input_transformation_name in self.model_config['inputs'].items():
          # print("debug <actual inputs>:", input_name, input_transformation_name)
          input_values = feats[input_name]
          # input_values = tf.Print(input_values, ["input value under {}".format(input_name), input_values, tf.shape(input_values)])
          if input_transformation_name == "cached_embeddings":
            # if hparams.cwr_ood:
            #   ROOT_emb = tf.get_variable("root_emb", shape=[1, 3072], initializer=tf.random_normal_initializer(), trainable=False)
            #   cached_cwr_embeddings_oov = tf.concat([cached_cwr_embeddings, ROOT_emb], axis=0)
            #   input_embedding_lookup = tf.nn.embedding_lookup(cached_cwr_embeddings_oov, input_values)
            # else:
            input_embedding_lookup = tf.nn.embedding_lookup(cached_cwr_embeddings, input_values)
            with tf.variable_scope("cwr_assembly"):
              num_layers = 3#input_embedding_lookup.get_shape()[2]
              weight = tf.get_variable("cwr_weight", shape=[num_layers])
              scale = tf.get_variable("cwr_scale", shape=[])
              input_embedding_lookup = scale * tf.math.reduce_sum(
              tf.split(input_embedding_lookup, axis=-1, num_or_size_splits=num_layers) * tf.reshape(tf.nn.softmax(weight), shape=[num_layers, 1, 1, 1]), axis=0)
          elif input_transformation_name == "bert_embeddings":
            input_embedding_lookup = tf.nn.embedding_lookup(cached_cwr_embeddings, input_values)
          elif input_transformation_name == "embeddings":
            print("embeddings", input_name, embeddings[input_name])
            input_embedding_lookup = tf.nn.embedding_lookup(embeddings[input_name], input_values)
          elif input_transformation_name == "predicate":
            print("embeddings", input_name, embeddings[input_name])
            input_embedding_lookup = tf.nn.embedding_lookup(embeddings[input_name], input_values)
            gp_embs.append(input_embedding_lookup)
            continue
          else:
            print("unknown input transformation {}".format(input_transformation_name))
            raise NotImplementedError
          # input_embedding_lookup = tf.Print(input_embedding_lookup, ["input embedding under {}".format(input_name), input_embedding_lookup])
          inputs_list.append(input_embedding_lookup)
          tf.logging.log(tf.logging.INFO, "Added %s to inputs list." % input_name)
      # TODO a mere workaround with one element concat
      current_input = tf.concat(inputs_list, axis=-1)

      ## <guard: condition to enter sentence features>
      ## suppose the dim is of (B, S, H)
      if hparams.input_project_layer_norm:
        current_input = tf.contrib.layers.layer_norm(current_input)

      sentence_feature = tf.reduce_sum(current_input * tf.expand_dims(tokens_to_keep, -1), axis=1)
      sentence_feature /= tf.expand_dims(tf.reduce_sum(tokens_to_keep, axis=1), -1) #To get the mean of all embeddings
      feats['sentence_feature'] = sentence_feature
      ## <guard: condition to enter sentence features>
      current_input = tf.nn.dropout(current_input, hparams.input_dropout)
      if len(gp_embs) >0:
        current_input = tf.concat([current_input, gp_embs[0]], axis=-1)

      with tf.variable_scope('project_input'):
        current_input = nn_utils.MLP(current_input, sa_hidden_size, n_splits=1)

      # current_input = tf.Print(current_input, [tf.shape(current_input)], "input shape")

      predictions = {}
      eval_metric_ops = {}
      export_outputs = {}
      loss = tf.constant(0.)
      items_to_log = {}


      num_layers = max(self.task_config.keys()) + 1
      tf.logging.log(tf.logging.INFO, "Creating transformer model with %d layers" % num_layers)
      with tf.variable_scope('transformer'):
        current_input = transformer.add_timing_signal_1d(current_input)
        for i in range(num_layers):
          # print("debug: <constructing {}-th layer>".format(i))
          with tf.variable_scope('layer%d' % i):

            special_attn = [[], []] #first bracket is for hard injection attns, the sencond is for discounting attns
            special_values = []
            if i in self.attention_config:

              this_layer_attn_config = self.attention_config[i]
              # print("debug: <layer_{} config>: ".format(i), this_layer_attn_config)

              print("debug <attention configuration>@{}: ".format(i), this_layer_attn_config)

              for attn_fn_item in this_layer_attn_config.keys():
                for attn_fn, attn_fn_map in this_layer_attn_config[attn_fn_item].items():
                  # print("debug <attn_fn, attn_fn_map>: ", attn_fn, ' ', attn_fn_map)
                  if 'length' in attn_fn_map.keys() or hparams.use_hparams_headcounts:
                    hc = hparams.__dict__['{}_headcount'.format(attn_fn)] if hparams.use_hparams_headcounts else  attn_fn_map['length']
                    tf.logging.log(tf.logging.INFO, "{} is using {} attention mode with {} heads".format(attn_fn_item,
                      hparams.__dict__['{}_injection'.format(attn_fn)], hc))
                    for idx in range(hc): # To make sure that the three special attentions are different
                      with tf.variable_scope('{}_{}'.format(attn_fn_item, idx)):
                        attention_fn_params = attention_fns.get_params(mode, attn_fn_map, predictions, feats, labels, hparams, self.model_config, tokens_to_keep)
                      this_special_attn, special_attn_weight = attention_fns.dispatch(attn_fn_map['name'])(**attention_fn_params)
                      # todo patches everywhere!
                      # this_special_attn = tf.Print(this_special_attn, [this_special_attn])
                      if special_attn_weight is not None and hparams.output_attention_weight:
                        for i in range(special_attn_weight.get_shape()[0]):
                          items_to_log["{}_{}_weight_{}".format(attn_fn, idx, i)] = special_attn_weight[i]
                      if hparams.__dict__['{}_injection'.format(attn_fn)] == 'injection':
                        special_attn[0].append(this_special_attn)
                      elif hparams.__dict__['{}_injection'.format(attn_fn)] == 'discounting':
                        special_attn[1].append(this_special_attn)
                      else:
                        tf.logging.log(tf.logging.ERROR, "The spcified injection method {} has not been implemented".format(attn_fn_map['injection_method']))
                        raise NotImplementedError
                      # print(special_attn)
                  else:
                    with tf.variable_scope('{}'.format(attn_fn)):
                      attention_fn_params = attention_fns.get_params(mode, attn_fn_map, predictions, feats, labels, hparams, self.model_config, tokens_to_keep)
                      this_special_attn, _ = attention_fns.dispatch(attn_fn_map['name'])(**attention_fn_params)
                    if hparams.__dict__['{}_injection'.format(attn_fn)] == 'injection':
                      special_attn[0].append(this_special_attn)
                    elif hparams.__dict__['{}_injection'.format(attn_fn)] == 'discounting':
                      special_attn[1].append(this_special_attn)
                    else:
                      tf.logging.log(tf.logging.ERROR,
                                     "The spcified injection method {} has not been implemented".format(
                                       attn_fn_map['injection_method']))
                      raise NotImplementedError
                # print("debug <layer_{} special attention>: ".format(i), special_attn )

              if 'value_fns' in this_layer_attn_config:
                tf.logging.log(tf.logging.ERROR, "special value section has been dropped temporarily")
                raise NotImplementedError
                for value_fn, value_fn_map in this_layer_attn_config['value_fns'].items():
                  value_fn_params = value_fns.get_params(mode, value_fn_map, predictions, feats, labels, embeddings)
                  this_special_values = value_fns.dispatch(value_fn_map['name'])(**value_fn_params)
                  special_values.append(this_special_values)
                # print("debug <layer_{} special values>: ".format(i), special_values)
              if hparams.attn_debug:
                print(special_attn)
                # special_attn[1][1] = tf.Print(special_attn[1][1], [special_attn[1][0], special_attn[1][1]], "debug_check equal attn")
                # assert_op = tf.assert_none_equal(special_attn[1][0], special_attn[1][1])
                  # tf.logging.log(tf.logging.INFO, "attention behavior is identical")
            current_input = transformer.transformer(current_input, tokens_to_keep, layer_config['head_dim'],
                                                    layer_config['num_heads'], hparams.attn_dropout,
                                                    hparams.ff_dropout, hparams.prepost_dropout,
                                                    layer_config['ff_hidden_size'], special_attn, special_values, special_attention_mode = hparams.special_attention_mode)
            # current_input = tf.Print(current_input, [tf.shape(current_input)], "LISA input after transformer")
            if i in self.task_config:

              # if normalization is done in layer_preprocess, then it should also be done
              # on the output, since the output can grow very large, being the sum of
              # a whole stack of unnormalized layer outputs.
              current_input = nn_utils.layer_norm(current_input)

              # todo test a list of tasks for each layer
              for task, task_map in self.task_config[i].items():
                # print("debug <task>: ", task)
                # print("debug <task map>:" , task_map)
                task_labels = labels[task]
                # task_labels = tf.Print(task_labels, [task_labels] , 'task_label'.format(task))
                task_vocab_size = self.vocab.vocab_names_sizes[task] if task in self.vocab.vocab_names_sizes else -1

                # Set up CRF / Viterbi transition params if specified
                with tf.variable_scope("crf"):  # to share parameters, change scope here
                  # transition_stats_file = task_map['transition_stats'] if 'transition_stats' in task_map else None
                  task_transition_stats = transition_stats[task] if task in transition_stats else None

                  # create transition parameters if training or decoding with crf/viterbi
                  task_crf = 'crf' in task_map and task_map['crf']
                  task_viterbi_decode = task_crf or 'viterbi' in task_map and task_map['viterbi']
                  transition_params = None
                  if task_viterbi_decode or task_crf:
                    # print("loading transition params", self.not_load_transition)
                    if hparams.train_with_crf:
                      transition_params = tf.get_variable("transitions", [task_vocab_size, task_vocab_size])
                    else:
                      tf.logging.log(tf.logging.INFO, "Use default transition param")
                      tf.logging.log(tf.logging.INFO, "transition parameters:{}".format(task_transition_stats))
                      transition_params = tf.constant(task_transition_stats, dtype=tf.float32)#tf.get_variable("transitions", [task_vocab_size, task_vocab_size],
                                                        #initializer=tf.constant_initializer(task_transition_stats) if not self.not_load_transition else tf.constant_initializer(0),
                                                        #trainable=task_crf)
                    # if mode != ModeKeys.TRAIN:
                    #   transition_params = tf.Print(transition_params, [tf.get_variable("transitions", [task_vocab_size, task_vocab_size])],
                    #                                                "optimized transition?")
                    # transition_params = tf.cond(tf.equal(mode, ModeKeys.TRAIN),
                    #                               lambda: transition_params,
                    #                               lambda: tf.Print(transition_params, [
                    #                                 tf.get_variable("transitions", [task_vocab_size, task_vocab_size])],
                    #                                                "optimized transition?"))
                      # transition_params =
                    train_or_decode_str = "training" if task_crf else "decoding"
                    tf.logging.log(tf.logging.INFO, "Created transition params for %s %s" % (train_or_decode_str, task))

                output_fn_params = output_fns.get_params(mode, self.model_config, task_map['output_fn'], predictions,
                                                         feats, labels, current_input, task_labels, task_vocab_size,
                                                         self.vocab.joint_label_lookup_maps, tokens_to_keep,
                                                         transition_params, hparams)
                # print("debug <dispatch into {}>".format(task_map['output_fn']['name']))
                task_outputs = output_fns.dispatch(task_map['output_fn']['name'])(**output_fn_params)
                # print("debug <task_outputs>: ", task_outputs)
                # want task_outputs to have:
                # - predictions
                # - loss
                # - scores
                # - probabilities
                predictions[task] = task_outputs

                # do the evaluation
                for eval_name, eval_map in task_map['eval_fns'].items():
                  eval_fn_params = evaluation_fns.get_params(task_outputs, eval_map, predictions, feats, labels,
                                                             task_labels, self.vocab.reverse_maps, tokens_to_keep)
                  if eval_name == 'parse_eval' and hparams.using_input_with_root:
                    eval_fn_params['has_root_token']=True
                  eval_result = evaluation_fns.dispatch(eval_map['name'])(**eval_fn_params)
                  eval_metric_ops[eval_name] = eval_result

                # get the individual task loss and apply penalty
                this_task_loss = task_outputs['loss'] * task_map['penalty']

                # log this task's loss
                items_to_log['%s_loss' % task] = this_task_loss

                #outputing sub loss as well
                for key in task_outputs.keys():
                  if key.startswith('loss'):
                    items_to_log['{}_{}'.format(task, key)] = task_outputs[key]

                # add this loss to the overall loss being minimized
                # this_task_loss = tf.Print(this_task_loss, [this_task_loss], '{}_{}'.format(task, key))
                loss += this_task_loss

                # print("debug <accumulated loss>: ", loss)
              # break # only take one loss

      # set up moving average variables
      assign_moving_averages_dep = tf.no_op()
      if hparams.moving_average_decay > 0.:
        moving_averager = tf.train.ExponentialMovingAverage(hparams.moving_average_decay, zero_debias=True,
                                                            num_updates=tf.train.get_global_step())
        moving_average_op = moving_averager.apply(train_utils.get_vars_for_moving_average(hparams.average_norms))

        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, moving_average_op)

        # use moving averages of variables if evaluating
        assign_moving_averages_dep = tf.cond(tf.equal(mode, ModeKeys.TRAIN),
                                             lambda: tf.no_op(),
                                             lambda: nn_utils.set_vars_to_moving_average(moving_averager))
      # print("debug <finishing setting up moving avg variables>")

      with tf.control_dependencies([assign_moving_averages_dep]):

        items_to_log['loss'] = loss
          # print("debug <final loss>: ", loss)
          # get learning rate w/ decay
        # todo dirty workaround
        if mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL:
          this_step_lr = train_utils.learning_rate(hparams, tf.train.get_global_step())
          items_to_log['lr'] = this_step_lr
            # print("debug <items to log>: ", items_to_log)
            # print("debug <eval_metric_content>: ", eval_metric_ops)

          if hparams.optimizer == "lazyadam":
            optimizer = LazyAdamOptimizer(learning_rate=this_step_lr, beta1=hparams.beta1,
                                          beta2=hparams.beta2, epsilon=hparams.epsilon,
                                          use_nesterov=hparams.use_nesterov)
          elif hparams.optimizer == "adam":
            optimizer = tf.train.AdamOptimizer(learning_rate=this_step_lr, beta1=hparams.beta1,
                                                         beta2=hparams.beta2, epsilon=hparams.epsilon)
          else:
            raise NotImplementedError("The specified optimizer is not implemented")
          # loss = tf.Print(loss, [loss], "loss")
          # # loss_no_nan = tf.cond(tf.reduce_any(tf.is_nan(loss)), lambda: tf.zeros_like(loss), lambda: loss)
          # # loss_no_nan = tf.where(tf.is_nan(loss), tf.zeros_like(loss), loss)
          # loss_no_nan = tf.where(tf.math.is_nan(loss), tf.zeros_like(loss), loss)
          # loss_no_nan_printed = tf.Print(loss_no_nan, [loss_no_nan], "no nan loss")
          # grad_and_var = optimizer.compute_gradients(loss_no_nan_printed)

          # loss = tf.where(tf.math.is_nan(loss), tf.zeros_like(loss), loss)
          # loss = tf.Print(loss, [loss], "loss")
          grad_and_var = optimizer.compute_gradients(loss)

          gradients, variables = zip(*grad_and_var)
          # gradients_without_nan = [tf.cond(tf.reduce_any(tf.is_nan(item)), lambda: tf.zeros_like(item), item)for item in gradients]
          # gradients_without_nan = gradients
          gradients, gn = tf.clip_by_global_norm(gradients, hparams.gradient_clip_norm)
          # print([g is None for g in gradients])
          # gn = gn[0]
          zero_clipped_gradients = [tf.clip_by_value(g, 0., 0.) if g is not None else g for g in gradients]
          gradients_prev_inf_norm = [tf.cond(tf.logical_or(tf.math.is_inf(gn), tf.math.is_nan(gn)), lambda: g_zeros, lambda: g) if g is not None else None
                                     for g_zeros, g in zip(zero_clipped_gradients, gradients)]

          # gn = tf.Print(gn, [gn], "global norm")
          with tf.control_dependencies([gn]):
            train_op = optimizer.apply_gradients(zip(gradients_prev_inf_norm, variables), global_step=tf.train.get_global_step())

          # if hparams.debug and mode == tf.estimator.ModeKeys.TRAIN:
          #   gradients_to_print = [gradients[variables.index(var)] for var in nn_utils.gradient_to_watch]
          #   print(gradients_to_print)
          #   gradients[0] = tf.Print(gradients[0], gradients_to_print, "gradient for dependency label strength")

          # train_op = optimizer.apply_gradients(zip(gradients, variables), global_step=tf.train.get_global_step())


          # export_outputs = {'predict_output': tf.estimator.export.PredictOutput({'scores': scores, 'preds': preds})}

          logging_hook = tf.train.LoggingTensorHook(items_to_log, every_n_iter=100)


          histogram_summary = [summary for name, summary in nn_utils.histogram_output.items()]
          summary_hook = tf.train.SummarySaverHook(
            save_steps=500,
            summary_op=[tf.summary.scalar(k, items_to_log[k]) for k in items_to_log.keys()] + histogram_summary)


        flat_predictions = {"%s_%s" % (k1, k2): v2 for k1, v1 in predictions.items() for k2, v2 in v1.items()}
        # print("debug <flat predictions>:", flat_predictions)
        export_outputs = {tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                          tf.estimator.export.PredictOutput(flat_predictions)}

        tf.logging.log(tf.logging.INFO,
                       "Created model with %d trainable parameters" % tf_utils.get_num_trainable_parameters())
        # if hparams.cwr!= 'None':
        #   with tf.Session() as sess:
        #     sess.run(tf.global_variables_initializer(), feed_dict={cached_cwr_embeddings_placeholder: self.cwr_embedding})


        if mode == tf.estimator.ModeKeys.TRAIN:
          return tf.estimator.EstimatorSpec(mode, flat_predictions, loss, train_op, eval_metric_ops,
                                          training_hooks=[logging_hook, summary_hook], export_outputs=export_outputs, scaffold=scaffold if hparams.cwr!='None' else None)
        elif mode == tf.estimator.ModeKeys.EVAL:
          return tf.estimator.EstimatorSpec(mode, flat_predictions, loss, train_op, eval_metric_ops,
                                            training_hooks=[logging_hook], export_outputs=export_outputs, scaffold=scaffold if hparams.cwr!='None' else None)
        elif mode == tf.estimator.ModeKeys.PREDICT:
          return tf.estimator.EstimatorSpec(mode, flat_predictions, loss, tf.no_op(), eval_metric_ops,
                                            export_outputs=export_outputs,
                                            scaffold=scaffold if hparams.cwr != 'None' else None)
