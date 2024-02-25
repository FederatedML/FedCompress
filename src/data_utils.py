import os
import contextlib
import numpy as np
import random
import math
import scipy
import functools
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_datasets as tfds
from typing import List
from sklearn.utils.class_weight import compute_class_weight

class SplitDataset:

	@staticmethod
	def get_class_stats(labels):
		classes, samples_per_class = np.unique(labels, return_counts=True)
		indexes_per_class = [np.flatnonzero(labels==c) for c in classes]
		class_weights = 1/compute_class_weight(class_weight="balanced",classes=classes,y=labels)
		return classes, samples_per_class, indexes_per_class, class_weights

	@staticmethod
	def get_class_distribution(num_classes, num_partitions=10, concentration=0.0, seed=0):

		def _scale(x, low_range=(5e-4,1e-1), high_range=(50,1e+5)):
			x=1.-x
			assert  0.0<=x<=1.0, 'Concentration must be a float in [0.0,1.0] range.'
			if x <= 0.05:
				# Scale values in the bottom 5% of the range exponentially to low_range
				return math.pow(x / 0.05, 2.0) * 0.05 * (low_range[1] - low_range[0]) + low_range[0]
			elif x >= 0.95:
				# Scale values in the top 5% of the range exponentially to high_range
				return (1 - math.pow((1 - x) / 0.05, 2.0)) * 0.95 * (high_range[1] - high_range[0]) + high_range[0]
			else:
				# Scale values in the middle 90% of the range linearly to (1e-4, 500)
				return (x - 0.05) / 0.9 * (high_range[0] - low_range[1]) + low_range[1]

		return np.random.default_rng(seed).dirichlet(alpha=np.repeat(_scale(concentration), num_classes), size=num_partitions)

	@staticmethod
	def get_label_distribution(num_samples, num_partitions=10, concentration=0.0, min_length=1, seed=0):

		def _scale(x, min_value=0.01, max_value=1000):
			# Reverse the value of x so that it increases from 1.0 to 0.0
			x = 1.0 - x
			# Scale the value of x to the range [0, pi]
			x = x * math.pi
			# Calculate the scaled value using the reversed cosine curve
			y = (max_value - min_value) * (1.0 - math.cos(x)) / 2.0 + min_value
			return y

		# Fix seed
		random.seed(seed)
		np.random.seed(seed)
		# Ensure that min_chunk_length is valid
		assert min_length >= 0 and min_length * num_partitions <= num_samples, "Invalid value for min_chunk_length"
		assert concentration >= 0 and concentration <= 1.0, "Invalid value for concentration"

		if concentration == 0:
			# If concentration is zero, return equal sized chunks
			chunk_size = num_samples // num_partitions
			remainder = num_samples % num_partitions
			chunks = [chunk_size] * num_partitions
			for i in range(remainder): chunks[i] += 1
			return np.asarray(chunks)
		else:
			# Otherwise, sample chunk sizes from a dirichlet distribution
			chunk_sizes = scipy.stats.dirichlet.rvs([_scale(concentration)] * num_partitions, size=1)[0]
			# Round chunk sizes to the nearest integer and ensure that each chunk
			# has at least min_chunk_length samples
			chunks = [max(int(round(size * num_samples)), min_length) for size in chunk_sizes]
			# Ensure that the total number of samples is equal to num_samples
			while sum(chunks) != num_samples:
				# Calculate the difference between the total number of samples and num_samples
				diff = num_samples - sum(chunks)
				# Calculate the amount to increase or decrease each chunk by
				sign = lambda a: (a>0) - (a<0)
				chunk_size_diff = sign(diff)*(abs(diff) // num_partitions)
				# Increase or decrease the size of each chunk by chunk_size_diff
				chunks = [max(chunk + chunk_size_diff, min_length) for chunk in chunks]
				# If the total number of samples is still not equal to num_samples,
				# increase or decrease a random chunk by the remaining difference
				if sum(chunks) != num_samples:
					while sum(chunks) != num_samples:
						remainder = num_samples - sum(chunks)
						chunk_id = random.choice(range(num_partitions))
						if chunks[chunk_id]+remainder >= min_length: chunks[chunk_id] += remainder
			assert sum(chunks) == num_samples, "Chunks do not sum up to the total number of samples."
			return np.asarray(chunks)

	@staticmethod
	def create_partitions(num_classes, num_partitions, class_distribution, samples_distribution, class_weights, class_samples=None, align=True, min_length=10, max_iter=10, seed=0):
		# Create empty partitions array
		parts = np.zeros((num_partitions,num_classes), dtype=np.int64)
		# Distribute class samples across partitions based on distributions.
		for i,part_class_distibution in enumerate(class_distribution):
			parts[i] = np.around(part_class_distibution * class_weights * samples_distribution[i])
		# (Try to) Fix rounding errors.
		if align and (class_samples is not None):
			i=0
			while not __class__.check_alignment(parts, class_samples, samples_distribution, verbose=0):
				if i==max_iter: break;
				parts = __class__.align_samples_distribution(parts, class_distribution, class_weights, samples_distribution, min_length, seed+i)
				parts = __class__.align_class_distribution(parts, class_samples, samples_distribution, min_length, seed+i+1)
				i+=1

		assert __class__.valid_partitions(parts, class_samples), 'A valid partitioning could not be constructed.'

		return parts

	@staticmethod
	def align_class_distribution(partitions, class_samples, samples_distribution, min_length=10, seed=0):
		# Fix random seed
		np.random.seed(seed)
		# Local functions
		compute_diff = lambda x: (class_samples - x.sum(axis=0))
		compute_replace_classes = lambda x: (True if abs(x)>partitions.shape[0] else False)
		# Available swappable partitions
		swappable_partitions = np.arange(partitions.shape[0])
		num_samples_to_replace = compute_diff(partitions)

		for i,n in enumerate(num_samples_to_replace):
			# Create random partitions to modify classes
			indexes = np.random.choice(swappable_partitions,
						size=abs(n),
						replace=compute_replace_classes(compute_diff(partitions)[i]),
						p=scipy.special.softmax((samples_distribution - partitions.sum(axis=1))))
			# Remove/Add sample for current partition
			for idx in indexes:
				if partitions[idx][i] + np.sign(n)>min_length:
					partitions[idx][i] = partitions[idx][i] + np.sign(n)
		return partitions

	@staticmethod
	def normalize_probabilities(prob):
		from sklearn.preprocessing import normalize
		if sum(prob) != 1.0:
			prob = normalize(prob[:,np.newaxis], axis=0, norm='l1',).ravel()
		return prob

	@staticmethod
	def align_samples_distribution(partitions, class_distribution, class_weights, samples_distribution, min_length=10, seed=0):
		# Fix random seed
		np.random.seed(seed)
		# Local functions
		compute_diff = lambda x: (samples_distribution - x.sum(axis=1))
		compute_replace_partitions = lambda x: (True if abs(x)>partitions.shape[0]-1 else False)
		compute_replace_classes = lambda x: (True if abs(x)>partitions.shape[1] else False)
		# Available swappable classes
		swappable_classes = np.arange(partitions.shape[1])

		for i in range(partitions.shape[0]):
			# Available swappable partitions
			swappable_partitions = np.setdiff1d(np.arange(partitions.shape[0]),[i])
			# Local variables
			num_samples_to_replace = compute_diff(partitions)

			partition_replace = compute_replace_partitions(num_samples_to_replace[i])
			class_replace = compute_replace_classes(num_samples_to_replace[i])
			# Create random partitions swaps
			part_idxs = np.random.choice(
							swappable_partitions,
							size=abs(num_samples_to_replace[i]),
							replace=partition_replace,
							p=__class__.normalize_probabilities(scipy.special.softmax(np.delete(num_samples_to_replace,[i], axis=0))+1e-9),
			)
			# Create random class swaps
			class_idxs = np.random.choice(swappable_classes,
							size=abs(num_samples_to_replace[i]),
							replace=class_replace,
							p=scipy.special.softmax(class_distribution[i]*class_weights),
			)
			for j,c in zip(part_idxs,class_idxs):
				if (partitions[i][c] + np.sign(num_samples_to_replace[i])>min_length) and (partitions[j][c] - np.sign(num_samples_to_replace[i])>min_length):
					# Remove sample for current partition
					partitions[i][c] = partitions[i][c] + np.sign(num_samples_to_replace[i])
					# Move it to another (random) partition
					partitions[j][c] = partitions[j][c] - np.sign(num_samples_to_replace[i])
		return partitions

	@staticmethod
	def check_alignment(partitions, class_samples, samples_distribution, verbose=0):

		if verbose:
			__class__.print_diff(partitions, class_samples, samples_distribution)

		if ((samples_distribution - partitions.sum(1)).var()==.0) and ((class_samples - partitions.sum(0)).var()==.0):
			if ((samples_distribution - partitions.sum(1)).sum()==.0) and ((class_samples - partitions.sum(0)).sum()==.0):
				return True
		else:
			return False

	@staticmethod
	def valid_partitions(partitions, class_samples):
		return (class_samples - partitions.sum(0)).sum()==0.0

	@staticmethod
	def print_diff(partitions, class_samples, samples_distribution):
		x = (samples_distribution - partitions.sum(1))
		y = (class_samples - partitions.sum(0))

		print(f"Sum is [{x.sum()}, {y.sum()}]")
		print(f"Var is [{x.var():0.2f}, {y.var():0.2f}]")
		print(f"Samples error: {x}")
		print(f"Class error: {y}\n")

	@staticmethod
	def create_mask(partitions, class_indexes, seed=0):
		# Fix random seed
		np.random.seed(seed)
		# Number of classes and partitions
		num_classes = len(class_indexes)
		num_partitions = partitions.shape[0]
		# Create mask array
		partitions_class_indexes = np.empty_like(partitions, dtype=object)
		# Iterate over classes
		for c_idx in range(len(class_indexes)):
			# Create (un)availabe indexes set
			available_class_indexes = set(class_indexes[c_idx])
			unavailable_class_indexes = set()
			# Iterate over partitions
			for i in range(num_partitions):
				# Create set of indexes
				available_class_indexes -= unavailable_class_indexes
				num_samples = partitions[i][c_idx]
				# Choose indexes for each partition
				indexes = np.random.choice(np.array(list(available_class_indexes)), size=num_samples, replace=False)
				partitions_class_indexes[i][c_idx] = indexes
				# Update unavailable indexes
				unavailable_class_indexes.update(set(indexes))
		# Create one index array per partition.
		indexes = np.empty(num_partitions, dtype=object)
		for i in range(num_partitions):
			indexes[i] = np.concatenate(partitions_class_indexes[i])
		return indexes, partitions_class_indexes

	@staticmethod
	def create_dataset_partition(data, cid=0, num_clients=10, data_skew=0.0, class_skew=0.0, min_samples=1, max_iter=1000, seed=0):
		# Read dataset statistics
		classes, class_samples, class_indexes, class_weights = __class__.get_class_stats(labels=data[1])
		# Create samples distribution (# samples per partition)
		samples_distribution = __class__.get_label_distribution(num_samples=data[1].shape[0], num_partitions=num_clients, concentration=data_skew, min_length=min_samples, seed=seed)
		# Create class distribution (% classes per partition)
		class_distribution = __class__.get_class_distribution(num_classes=classes.size, num_partitions=num_clients, concentration=class_skew, seed=seed)
		# Create partition arrays (# samples per class for each partition)
		partitions = __class__.create_partitions(num_classes=classes.size, num_partitions=num_clients,
					class_distribution=class_distribution, samples_distribution=samples_distribution,
					class_weights=class_weights, class_samples=class_samples,
					align=True, min_length=min_samples, max_iter=max_iter, seed=seed)
		# Create indexes masks based on partitions
		indexes,_ = __class__.create_mask(partitions, class_indexes=class_indexes, seed=seed)
		# Create mask
		mask = np.zeros(data[1].shape[0], dtype=bool)
		mask[indexes[int(cid)]] = True
		# Return tuple of (indexes,labels)
		return (data[0][mask], data[1][mask])

	@staticmethod
	def concatenate_client_data(ds, clients_ids):
		client_ds = ds.create_tf_dataset_for_client(clients_ids[0])
		if len(clients_ids)>1:
			for i in clients_ids[1:]:
				client_ds = client_ds.concatenate(ds.create_tf_dataset_for_client(i))
		return client_ds

	@staticmethod
	def concatenate_client_data_from_numpy(ds, clients_ids):
		images, labels = [],[]
		for id in clients_ids:
			(x,y) = ds[id][()].values()
			images.extend(x)
			labels.extend(y)
		return (np.array(images), np.array(labels))

	@staticmethod
	def split_clients_ids_to_partitions(cid, num_clients, available_ids, seed=0):
		# Fix seed for reproducability.
		np.random.seed(seed)
		# Ensure split is possible.
		assert num_clients <= len(available_ids), 'Number of clients exceeds avaialable clients ids.'
		partitions = np.random.choice(a=num_clients, size=len(available_ids), replace=True)
		return [available_ids[i] for i in np.argwhere(np.isin(partitions, [int(cid)])).ravel()]

class MaskDataset:

	@staticmethod
	def create_lookup_table(mask):
		keys = tf.constant(mask)
		values = tf.ones_like(keys)
		return tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer(keys, values), default_value=0)

	@staticmethod
	def hash_table_filter(index, _, table):
		return tf.cast(table.lookup(index), tf.bool)

	@staticmethod
	def filter_dataset(ds, indexes):
		# Ensure that dataset is a tf.dataset object
		if isinstance(ds,list): ds = ds[0]
		# Construct lookup table from indexes
		table = __class__.create_lookup_table(np.asarray(indexes))
		# Convert to enumerated dataset
		ds = ds.enumerate()
		# Filter dataset based on lookup table
		ds = ds.filter(functools.partial(__class__.hash_table_filter, table=table))
		# Convert back to original dataset
		ds = ds.map(lambda _,x: x)
		return ds

class AugmentDataset:

	IMAGENET_STD = tf.reshape((0.2023, 0.1994, 0.2010), shape=(1, 1, 3))
	IMAGENET_MEAN = tf.reshape((0.4914, 0.4822, 0.4465), shape=(1, 1, 3))

	@staticmethod
	def image_valid_prep(x,y):
		x = tf.cast(x, tf.float32) / 255.
		x = (x - __class__.IMAGENET_MEAN) / __class__.IMAGENET_STD
		return x, y

	@staticmethod
	def image_train_prep(x,y=None, size=(32,32)):
		x = tf.cast(x, tf.float32) / 255.
		x = tf.image.random_flip_left_right(x)
		x = tf.image.pad_to_bounding_box(x, 4, 4, size[0]+8, size[1]+8)
		x = tf.image.random_crop(x, (size[0], size[1], 3))
		x = (x - __class__.IMAGENET_MEAN) / __class__.IMAGENET_STD
		return x if y is None else (x,y)

	# emnist support
	@staticmethod
	def image_valid_prep_v2(x,y):
		x = tf.cast(x, tf.float32) / 255.
		return x, y

	# emnist support
	@staticmethod
	def image_train_prep_v2(x,y=None, size=(32,32)):
		x = tf.cast(x, tf.float32) / 255.
		x = tf.image.random_flip_left_right(x)
		return x if y is None else (x,y)

	@staticmethod
	def image_ood_prep(x, color_mode='rgb', size=(32,32)):
		if color_mode=='grayscale': return tf.map_fn(__class__.image_train_prep_v2, x)
		return tf.map_fn(functools.partial(__class__.image_train_prep, size=size), x)

	@staticmethod
	def audio_valid_prep(x,y, bins=64, sr=16000, to_float=True):
		# Convert to float
		if to_float: x = tf.cast(x, tf.float32) / float(tf.int16.max)
		# Compute mel spectrograms
		x = __class__.logmelspectogram(x, bins=bins, sr=sr)
		return x, y

	@staticmethod
	def audio_train_prep(x, y=None, seconds=1, bins=64, sr=16000, to_float=True):
		# Convert to float
		if to_float: x = tf.cast(x, tf.float32) / float(tf.int16.max)
		# Pad to seconds length
		x = __class__.pad(x, sequence_length=int(sr*seconds))
		# Random crop (if larger)
		x = tf.image.random_crop(x, [int(sr*seconds)])
		# Compute mel spectrograms
		x = __class__.logmelspectogram(x, bins=bins, sr=sr)
		return x if y is None else (x,y)

	@staticmethod
	def audio_ood_prep(x, seconds=1, bins=64, sr=16000, to_float=True):
		return __class__.audio_train_prep(tf.squeeze(x), seconds=seconds, bins=bins, sr=sr, to_float=to_float)

	######################
	# Image Augmentations
	######################

	" Cutout"
	def cutout(images, labels=None, mask_size=(16,16)):
		_images = tfa.image.cutout(images, mask_size=mask_size)
		if labels is None: return _images
		return _images, labels

	" Mixup"
	def mixup(x, y=None):
		alpha = tf.random.uniform([], 0, 1)
		mixedup_x = alpha * x + (1 - alpha) * tf.reverse(x, axis=[0])
		if y is None: return mixedup_x
		return mixedup_x, y

	######################
	# Audio Augmentations
	######################

	@staticmethod
	def read_audio(fp, label=None):
		waveform, _ = tf.audio.decode_wav(tf.io.read_file(fp))
		return waveform[Ellipsis, 0], label

	@staticmethod
	def pad(waveform, sequence_length=16000):
		padding = tf.maximum(sequence_length - tf.shape(waveform)[0], 0)
		left_pad = padding // 2
		right_pad = padding - left_pad
		return tf.pad(waveform, paddings=[[left_pad, right_pad]])

	@staticmethod
	def logmelspectogram(x, bins=64, sr=16000, fmin=60.0, fmax=7800.0, fft_length=1024):
		# Spectrogram extraction
		s = tf.signal.stft(x, frame_length=400, frame_step=160, fft_length=fft_length)
		x = tf.abs(s)
		w = tf.signal.linear_to_mel_weight_matrix(bins, s.shape[-1], sr, fmin, fmax)
		x = tf.tensordot(x, w, 1)
		x.set_shape(x.shape[:-1].concatenate(w.shape[-1:]))
		x = tf.math.log(x+1e-6)
		return x[Ellipsis, tf.newaxis]

class LoadDataset:

	@staticmethod
	def get_cifar10(split, with_info=True):
		if isinstance(split,List): split = split[0]
		(train_images, train_labels), (test_images,test_labels) = tf.keras.datasets.cifar10.load_data()
		info = {'num_classes': len(np.unique(train_labels)), 'num_examples': test_images.shape[0] if split=='test' else train_images.shape[0]}
		ds = (test_images,test_labels) if split=='test' else (train_images, train_labels)
		return (ds, info) if with_info else ds

	@staticmethod
	def get_stylegan(split=None, with_info=True, reshape_size=(32,32), color_mode='rgb', name='stylegan_oriented'):
		with contextlib.redirect_stdout(None): # Read data from dir
			ds = tf.keras.preprocessing.image_dataset_from_directory(
				directory=os.path.join(os.environ['TFDS_DATA_DIR'],f'raw/{name}/'),
				label_mode=None, shuffle=False, batch_size=None, image_size=reshape_size, color_mode=color_mode, seed=0)
		return (ds, None) if with_info else ds

	@staticmethod
	def get_speech_commands(split, with_info=True):
		ds, info = tfds.load('speech_commands', split=split, with_info=True, as_supervised=True, shuffle_files=False)
		info = {'num_classes': info.features['label'].num_classes, 'num_examples': info.splits[split].num_examples}
		return (ds, info) if with_info else ds

	@staticmethod
	def get_librispeech(split=None, with_info=True, reshape_size=None, name='librispeech'):
		with contextlib.redirect_stdout(None): # Read data from dir
			ds = tf.keras.utils.audio_dataset_from_directory(
				directory=os.path.join(os.environ['TFDS_DATA_DIR'],f'raw/{name}/'),
				label_mode=None, shuffle=False, batch_size=None, seed=0)
		return (ds, None) if with_info else ds
