import tensorflow as tf
import numpy as np
import output_fns
from src import attention_fns


class OutputFnTests(tf.test.TestCase):

	# correct solution:
	def softmax(self, x):
		"""Compute softmax values for each set of scores in x."""
		e_x = np.exp(x - np.max(x))
		return e_x / e_x.sum(axis=0)


	def test_attention_to_aggregated(self):
		# raise NotImplemented
		with self.test_session():
			dependency_list = tf.constant([[[1, 24, 4, 4, 1]], [[1, 24, 3, 4, 1]], [[1, 24, 4, 4, 1]]])
			weight = tf.constant([0.2, 0.3, 0.5])
			dependency_list_weight_pair = (dependency_list, weight)
			attention = attention_fns.attention_to_aggregated(mode=tf.estimator.ModeKeys.TRAIN, train_attention_to_aggregated=dependency_list_weight_pair, eval_attention_to_aggregated=None)
			print(attention.eval())


if __name__ == '__main__':

	tf.test.main()
