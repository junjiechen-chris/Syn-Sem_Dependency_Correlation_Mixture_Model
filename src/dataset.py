import tensorflow as tf
import constants
from data_generator import conll_data_generator
# from tensor2tensor import utils

from t2t_data_reader import input_fn, token_based_batching


def map_strings_to_ints(vocab_lookup_ops, data_config, feature_label_names, cached_embedding = None):
  def _mapper(d):
    intmapped = []
    # print("debug <feature_label_names>: ", feature_label_names)
    # print("debug <data_config>:", data_config)
    for i, datum_name in enumerate(feature_label_names):
      if 'vocab' in data_config[datum_name]:
        # todo this is a little clumsy -- is there a better way to pass this info through?
        # todo also we need the variable-length feat to come last, gross
        if 'type' in data_config[datum_name] and data_config[datum_name]['type'] == 'range':
          idx = data_config[datum_name]['conll_idx']
          # print("debug <map with vocab-variable length>@{}: ".format(i), d[:, i:])
          if idx[1] == -1:
            # print("debug <converting {}>: ".format(datum_name), d[:, i:-1])
            intmapped.append(vocab_lookup_ops[data_config[datum_name]['vocab']].lookup(d[:, i:]))
          else:
            last_idx = i + 1#idx[1]
            # print("debug <converting {}>: ".format(datum_name), d[:, i:last_idx])
            intmapped.append(vocab_lookup_ops[data_config[datum_name]['vocab']].lookup(d[:, i:last_idx]))
        else:
          # print("debug <map with vocab>@{} @{}: ".format(datum_name, i), d[:, i])
          stub = d[:, i]
          # if i==7:
          #   stub = tf.Print(stub, [stub], "parse_label_value", summarize=60)
          #   stub = tf.Print(stub, [vocab_lookup_ops[data_config[datum_name]['vocab']].lookup(stub)], "converted parsed_label", summarize=60)
          intmapped.append(tf.expand_dims(vocab_lookup_ops[data_config[datum_name]['vocab']].lookup(stub), -1))
      else:
        # print("debug <stoi>@{}: ".format(i), d[:, i])
        intmapped.append(tf.expand_dims(tf.string_to_number(d[:, i], out_type=tf.int64), -1))

    ret = tf.cast(tf.concat(intmapped, axis=-1), tf.int32)
    # ret = tf.Print(ret, [ret, tf.shape(ret)], 'str2int conversion')
    # this is where the order of features/labels in input gets defined
    # todo: can i have these come out of the lookup as int32?
    # return ret, tf.zeros([2,1])
    return ret #tf.cast(tf.concat(intmapped, axis=-1), tf.int32)

  return _mapper


def get_data_iterator(data_filenames, data_config, vocab_lookup_ops, batch_size, num_epochs, shuffle,
                      shuffle_buffer_multiplier, is_token_based_batching, cached_embedding=None):



  # todo do something smarter with multiple files + parallel?

  with tf.device('/cpu:0'):

    # get the names of data fields in data_config that correspond to features or labels,
    # and thus that we want to load into batches
    feature_label_names = [d for d in data_config.keys() if \
                           ('feature' in data_config[d] and data_config[d]['feature']) or
                           ('label' in data_config[d] and data_config[d]['label'])]

    # get the dataset
    dataset = tf.data.Dataset.from_generator(lambda: conll_data_generator(data_filenames, data_config),
                                             output_shapes=[None, None], output_types=tf.string)

    # intmap the dataset
    dataset = dataset.map(map_strings_to_ints(vocab_lookup_ops, data_config, feature_label_names, None), num_parallel_calls=8)
    # dataset = dataset.map(map_strings_to_ints(vocab_lookup_ops, data_config, feature_label_names))

    dataset = dataset.cache()
    if is_token_based_batching:
      dataset = token_based_batching(dataset=dataset,
             batch_size_means_tokens=True,
             batch_size_multiplier=2,
             max_length=80,
             # mode,
            shuffle = shuffle,
            num_epochs = num_epochs,
            batchsize=batch_size,
            min_length=0,
            batch_shuffle_size=batch_size*shuffle_buffer_multiplier,
            pad_value = constants.PAD_VALUE)
    else:
      bucket_boundaries = constants.DEFAULT_BUCKET_BOUNDARIES
      bucket_batch_sizes = [batch_size] * (len(bucket_boundaries) + 1)
      dataset = dataset.apply(tf.contrib.data.bucket_by_sequence_length(element_length_func=lambda d: tf.shape(d)[0],
                                                                        bucket_boundaries=bucket_boundaries,
                                                                        bucket_batch_sizes=bucket_batch_sizes,
                                                                        padded_shapes=dataset.output_shapes,
                                                                        padding_values=constants.PAD_VALUE))
      if shuffle:
        dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(buffer_size=batch_size*shuffle_buffer_multiplier,
                                                                   count=num_epochs))
    # todo should the buffer be bigger?
    dataset.prefetch(buffer_size=1)

    # create the iterator
    # it has to be initializable due to the lookup tables
    iterator = dataset.make_initializable_iterator()
    tf.add_to_collection(tf.GraphKeys.TABLE_INITIALIZERS, iterator.initializer)

    return iterator.get_next()
    # return dataset # in case of distributed training
