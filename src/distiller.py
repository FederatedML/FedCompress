import tensorflow as tf

class Distiller(tf.keras.Model):

	def __init__(self, student, teacher, has_labels=False):
		super(Distiller, self).__init__()
		self.teacher = teacher
		self.student = student
		self.is_source_dataset_has_labels = has_labels

	def save(self, *args, **kwargs):
		return self.student.save(include_optimizer=False, *args, **kwargs)

	def save_weights(self, *args, **kwargs):
		return self.student.save_weights(*args, **kwargs)

	def load_weights(self, *args, **kwargs):
		return self.student.load_weights(*args, **kwargs)

	def compile(self, optimizer, metrics, distillation_loss_fn, temperature=8., energy_confidence=0.):
		super(Distiller, self).compile(optimizer=optimizer, metrics=metrics)
		self.distillation_loss_fn = distillation_loss_fn
		self.temperature = temperature
		self.energy_confidence = energy_confidence

	def train_step(self, data):

		if self.is_source_dataset_has_labels: # If knowledge distillation from source data
			x, _ = data
		else: # If knowledge distillation from single image source
			x = data

		# Forward pass of teacher
		teacher_predictions = self.teacher(x, training=False)

		if self.energy_confidence > 0.:
			pseudo_mask = tf.cast(-1. * tf.math.reduce_logsumexp(teacher_predictions, axis=-1) >= self.energy_confidence, dtype=tf.float32)

		with tf.GradientTape() as tape:
			# Forward pass of student
			student_predictions = self.student(x, training=True)

			# Compute losses
			distillation_loss = self.distillation_loss_fn(
				tf.nn.softmax(teacher_predictions / self.temperature, axis=1),
				tf.nn.softmax(student_predictions / self.temperature, axis=1),
			)

			if self.energy_confidence > 0.:
				loss = tf.reduce_mean(tf.cast(distillation_loss, dtype=tf.float32) * pseudo_mask)
			else:
				loss = distillation_loss

		# Compute gradients
		trainable_vars = self.student.trainable_variables
		gradients = tape.gradient(loss, trainable_vars)
		# Update weights
		self.optimizer.apply_gradients(zip(gradients, trainable_vars))
		# Return a dict of performance
		results = {m.name: m.result() for m in self.metrics}
		results.update({"distillation_loss": distillation_loss})
		return results

	def test_step(self, data):
		# Unpack the data
		x, y = data
		# Compute predictions
		y_prediction = self.student(x, training=False)
		# Update the metrics.
		self.compiled_metrics.update_state(y, y_prediction)
		# Return a dict of performance
		results = {m.name: m.result() for m in self.metrics}
		return results