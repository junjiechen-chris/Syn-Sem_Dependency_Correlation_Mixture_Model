import tensorflow as tf
import numpy as np
import json
class EvalResultsExporter(tf.estimator.Exporter):
  """Passed into an EvalSpec for saving the result of the final evaluation
  step locally or in Google Cloud Storage.
  """

  def __init__(self, name):
    assert name, '"name" argument is required.'
    self._name = name

  @property
  def name(self):
    return self._name

  def export(self, estimator, export_path, checkpoint_path,
    eval_result, is_the_final_export):

    if not is_the_final_export:
      return None

    tf.logging.info(('EvalResultsExporter (name: %s) '
      'running after final evaluation.') % self._name)
    tf.logging.info('export_path: %s' % export_path)
    tf.logging.info('eval_result: %s' % eval_result)

    for key, value in eval_result.iteritems():
      if isinstance(value, np.float32):
        eval_result[key] = value.item()

    tf.gfile.MkDir(export_path)

    with tf.gfile.GFile('%s/eval_results.json' % export_path, 'w') as f:
      f.write(json.dumps(eval_result))
