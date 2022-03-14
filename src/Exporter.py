import glob
import json
import os
import shutil

import neptune.new as neptune
import numpy as np
import tensorflow as tf


class Checkpoint(object):
	dir = None
	file = None
	score = None
	path = None

	def __init__(self, path, score):
		self.dir = os.path.dirname(path)
		self.file = os.path.basename(path)
		self.score = score
		self.path = path
	def todict(self):
		return {
			'score': self.score,
			'path': self.path
		  }
	@staticmethod
	def load_from_dict(path, score):
		print("creating checkpoint with {},  {}".format(path, score))
		return Checkpoint(path=path, score=score)

skip_list = []
class BestCheckpointCopier(tf.estimator.Exporter):
	checkpoints = None
	checkpoints_to_keep = None
	compare_fn = None
	name = None
	score_metric = None
	sort_key_fn = None
	sort_reverse = None

	def __init__(self, name='best_checkpoints', checkpoints_to_keep=5, score_metric='Loss/total_loss',
	             compare_fn=lambda x, y: x.score < y.score, sort_key_fn=lambda x: x.score, sort_reverse=False, neptune_handler=None):
		# assert  neptune_handler is not None
		self.neptune_handler = neptune_handler
		self.checkpoints = []
		self.checkpoints_to_keep = checkpoints_to_keep
		self.compare_fn = compare_fn
		self.name = name
		self.score_metric = score_metric
		self.sort_key_fn = sort_key_fn
		self.sort_reverse = sort_reverse
		super(BestCheckpointCopier, self).__init__()

	def _copyCheckpoint(self, checkpoint):
		desination_dir = self._destinationDir(checkpoint)
		os.makedirs(desination_dir, exist_ok=True)

		for file in glob.glob(r'{}*'.format(checkpoint.path)):
			self._log('copying {} to {}'.format(file, desination_dir))
			shutil.copy(file, desination_dir)
		checkpint_file = os.path.join(checkpoint.dir, 'checkpoint')
		shutil.copy(checkpint_file, desination_dir)
		# shutil.copy(os.join())

	def _destinationDir(self, checkpoint):
		return os.path.join(checkpoint.dir, self.name)

	def _keepCheckpoint(self, checkpoint):
		self._log('keeping checkpoint {} with score {}'.format(checkpoint.file, checkpoint.score))
		self.checkpoints.append(checkpoint)
		self.checkpoints = sorted(self.checkpoints, key=self.sort_key_fn, reverse=self.sort_reverse)
		self._log('current ckpt list: {}'.format([i.todict() for i in self.checkpoints]))

		self._copyCheckpoint(checkpoint)
		self._copyAssetsIfNotDone(checkpoint)
	def _copyAssetsIfNotDone(self, checkpoint):
		desination_dir = self._destinationDir(checkpoint)
		desination_dir = os.path.join(desination_dir, 'assets.extra')
		if not os.path.exists(desination_dir):
			checkpoint_asset = os.path.join(checkpoint.dir, 'assets.extra')
			self._log('copying asset {} to {}'.format(checkpoint_asset, desination_dir))
			shutil.copytree(checkpoint_asset, desination_dir)


	def _log(self, statement):
		tf.logging.info('[{}] {}'.format(self.__class__.__name__, statement))

	def _pruneCheckpoints(self, checkpoint):
		destination_dir = self._destinationDir(checkpoint)

		for checkpoint in self.checkpoints[self.checkpoints_to_keep:]:
			self._log('removing old checkpoint {} with score {}'.format(checkpoint.file, checkpoint.score))

			old_checkpoint_path = os.path.join(destination_dir, checkpoint.file)
			for file in glob.glob(r'{}*'.format(old_checkpoint_path)):
				self._log('removing old checkpoint file {}'.format(file))
				os.remove(file)

		self.checkpoints = self.checkpoints[0:self.checkpoints_to_keep]

	def _score(self, eval_result):
		return float(eval_result[self.score_metric])

	def _shouldKeep(self, checkpoint):
		return len(self.checkpoints) < self.checkpoints_to_keep or self.compare_fn(checkpoint, self.checkpoints[-1])

	def _export_detail_result(self, eval_result, checkpoint_path):
		for key, value in eval_result.items():
			if isinstance(value, np.ndarray):
				eval_result[key] = value.tolist()
			elif isinstance(value, list):
				eval_result[key] = [item.item() for item in value]
			else:
				eval_result[key] = value.item()
		with tf.gfile.GFile('%s/eval_results.json' % checkpoint_path, 'a') as f:
			f.write("\n")
			f.write(json.dumps(eval_result))

	def _should_restore_state(self, state_file):
		return os.path.exists(state_file)
	def _export_to_neptune(self, eval_result, tag):
		for key, value in eval_result.items():
			if key in skip_list:
				continue
			if isinstance(value, np.ndarray):
				the_list = value.tolist()
				list_len = len(the_list)
				for item, idx in zip(the_list, range(list_len)):
					self.neptune_handler['eval_result_{}/{}_{}'.format(tag, key, idx)].log(item, step=eval_result["global_step"].item())
			elif isinstance(value, list):
				# list_len = len(the_list)
				for item, idx in zip(value, range(len(value))):
					self.neptune_handler['eval_result_{}/{}_{}'.format(tag, key, idx)].log(item,
					                                                                       step=eval_result["global_step"].item())
			else:
				self.neptune_handler['eval_result_{}/{}'.format(tag, key)].log(value.item(), step=eval_result["global_step"].item())
		pass
	def export(self, estimator, export_path, checkpoint_path, eval_result, is_the_final_export):

		self._log('export checkpoint {}'.format(checkpoint_path))
		if self.neptune_handler is not None:
			self._export_to_neptune(eval_result, "all")
		# print("debug <eval_result>:", eval_result)
		score = self._score(eval_result)
		checkpoint = Checkpoint(path=checkpoint_path, score=score)
		checkpoint_records = os.path.join(self._destinationDir(checkpoint), 'checkpoint_record')
		if os.path.exists(checkpoint_records) and len(self.checkpoints)==0:
			with open(checkpoint_records) as f:
				ckpt_list = json.loads(f.readlines()[-1])
			ckpt_list = [Checkpoint.load_from_dict(**item) for item in ckpt_list]
			self.checkpoints=ckpt_list
			self._log('current ckpt list: {}'.format(self.checkpoints))


		if self._shouldKeep(checkpoint):
			if self.neptune_handler is not None:
				self._export_to_neptune(eval_result, "best")
			self._keepCheckpoint(checkpoint)
			self._pruneCheckpoints(checkpoint)
			self._export_detail_result(eval_result, self._destinationDir(checkpoint))
			with open(checkpoint_records, "a") as f:
				f.write("\n")
				f.write(json.dumps([item.todict() for item in self.checkpoints]))
		else:
			self._log('skipping checkpoint {}'.format(checkpoint.path))

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

