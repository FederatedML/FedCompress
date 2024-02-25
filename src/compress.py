import functools
import tensorflow as tf
import tensorflow_model_optimization as tfmot
import distiller

def _apply_weight_clustering(layer, clustering_params):
	if isinstance(layer, tf.keras.layers.Conv2D):
		return tfmot.clustering.keras.cluster_weights(layer, **clustering_params)
	return layer

def compress_image_model(model, nb_clusters=50, verbose=True):
	clustering_params = {"number_of_clusters": nb_clusters,"cluster_centroids_init": tfmot.clustering.keras.CentroidInitialization.KMEANS_PLUS_PLUS}
	if verbose:
		print(f"Weight clustering with {nb_clusters} clusters", flush=True)
	return tf.keras.models.clone_model(model, clone_function=functools.partial(_apply_weight_clustering, clustering_params=clustering_params))

def self_compress(model, num_classes, model_loader, data_loader, data_shape=(32,32,3), nb_clusters=30, epochs=5, batch_size=128, learning_rate=1e-4, temperature=2.0, seed=0, verbose=0):
	ood_data = data_loader(batch_size=batch_size, seed=seed, reshape_size=data_shape[1:-1])
	model.trainable = False
	student = model_loader(input_shape=data_shape[1:], num_classes=num_classes)
	student.set_weights(model.get_weights())
	student_model = compress_image_model(student, nb_clusters=nb_clusters, verbose=False)
	trainer = distiller.Distiller(student=student_model, teacher=model, has_labels=False)
	trainer.compile(
		optimizer=tf.keras.optimizers.Adam(learning_rate), distillation_loss_fn=tf.keras.losses.KLDivergence(),
		metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')], temperature=temperature)
	trainer.fit(ood_data, epochs=epochs, verbose=0)
	student_compressed = tfmot.clustering.keras.strip_clustering(trainer.student)
	student_compressed.compile(metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')])
	return student_compressed

def self_compress_with_data(model, data, nb_clusters=50, epochs=5, learning_rate=1e-4):
	student_model = compress_image_model(model, nb_clusters=nb_clusters, verbose=False)
	student_model.compile(
		optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
		loss=tf.keras.losses.SparseCategoricalCrossentropy(name='loss',from_logits=True),
		metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')]
	)
	h = student_model.fit(data, epochs=epochs, verbose=0)
	student_compressed = tfmot.clustering.keras.strip_clustering(student_model)
	student_compressed.compile(metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')])
	return student_compressed, h
