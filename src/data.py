import os
import functools
import numpy as np
import tensorflow as tf
from data_utils import LoadDataset, SplitDataset, MaskDataset, AugmentDataset
from sklearn.model_selection import train_test_split

DATASETS = {
	'cifar10': {
		'get_data_fn': LoadDataset.get_cifar10,
		'train_fn':AugmentDataset.image_train_prep,
		'test_fn': AugmentDataset.image_valid_prep,
		'augment_fn': AugmentDataset.cutout,
	},
	'stylegan': {
		'get_data_fn': LoadDataset.get_stylegan,
		'train_fn':AugmentDataset.image_ood_prep,
		'test_fn': None,
		'augment_fn': None,
	},
	'spcm': {
		'get_data_fn': LoadDataset.get_speech_commands,
		'train_fn':AugmentDataset.audio_train_prep,
		'test_fn': AugmentDataset.audio_valid_prep,
		'augment_fn': None,
	},
	'librispeech': {
		'get_data_fn': LoadDataset.get_librispeech,
		'train_fn':AugmentDataset.audio_ood_prep,
		'test_fn': None,
		'augment_fn': None,
	},
}

def get_cifar10(id=0, num_clients=10, return_eval_ds=False, batch_size=128, valid_split=0.1, seed=0, name='cifar10'):

	# Fix seed for reproducability.
	np.random.seed(seed)

	# Prepare evaluation set.
	if return_eval_ds:
		ds, info = DATASETS[name]['get_data_fn'](split='test', with_info=True)
		ds = tf.data.Dataset.from_tensor_slices(ds)\
				.map(DATASETS[name]['test_fn']).batch(batch_size*4).prefetch(-1)
		ds = (ds,None)
		num_samples, num_classes = info['num_examples'], info['num_classes']

	else:
		# Load data
		ds, info = DATASETS[name]['get_data_fn'](split='train', with_info=True)
		num_samples, num_classes = info['num_examples'], info['num_classes']
		# Get client data
		samples_to_ids = np.random.choice(a=np.arange(0,num_clients),size=num_samples).astype(int)
		mask = np.in1d(samples_to_ids, [int(id)])
		train_images,train_labels = (ds[0][mask,::], ds[1][mask,::])

		val_ds = None
		num_samples = (train_images.shape[0], None)

		# Validation set.
		if valid_split>0.0:
			train_images, valid_images, train_labels, valid_labels = train_test_split(train_images, train_labels, test_size=valid_split, random_state=seed)
			val_ds = tf.data.Dataset.from_tensor_slices((valid_images,valid_labels))\
						.map(DATASETS[name]['test_fn']).batch(batch_size).prefetch(-1)
			num_samples = (train_images.shape[0], valid_images.shape[0])

		# Train set.
		ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels))\
				.map(DATASETS[name]['train_fn'], num_parallel_calls=tf.data.AUTOTUNE).shuffle(10000,seed,True).batch(batch_size)
		if DATASETS[name]['augment_fn'] is not None:
			ds = ds.map(DATASETS[name]['augment_fn'], num_parallel_calls=tf.data.AUTOTUNE)
		ds = ds.prefetch(-1)
		ds = (ds,val_ds)

	return ds, num_classes, num_samples

def get_stylegan(batch_size=128, size=50000, reshape_size=(32,32), seed=0, name='stylegan'):
	# Fix seed for reproducability.
	np.random.seed(seed)
	# Load data
	ds, info = DATASETS[name]['get_data_fn'](split=None, with_info=True, reshape_size=reshape_size, color_mode='rgb')
	ds = ds.take(size).shuffle(10000,seed,True).batch(batch_size, drop_remainder=True)
	# Take subset of dataset
	ds = ds.map(functools.partial(DATASETS[name]['train_fn'], color_mode='rgb', size=reshape_size), num_parallel_calls=tf.data.AUTOTUNE)
	# Add augmmentations
	if DATASETS[name]['augment_fn'] is not None: # and dataset_name != 'emnist':
		ds = ds.map(functools.partial(DATASETS[name]['augment_fn'], mask_size=tuple([x//2 for x in reshape_size])), num_parallel_calls=tf.data.AUTOTUNE)
	ds = ds.prefetch(-1)
	return ds

def get_spcm(id=0, num_clients=10, return_eval_ds=False, batch_size=128, valid_split=0.1, seed=0, num_mels=40, name='spcm',):

	# Fix seed for reproducability.
	np.random.seed(seed)

	# Prepare evaluation set.
	if return_eval_ds:
		ds, info = DATASETS[name]['get_data_fn'](split='test', with_info=True)
		ds = ds.map(functools.partial(DATASETS[name]['test_fn'], bins=num_mels), num_parallel_calls=tf.data.AUTOTUNE).\
			batch(1).prefetch(-1)
		ds = (ds,None)
		num_classes, num_samples = info['num_classes'], info['num_examples']

	else:
		# Load data
		ds, info = DATASETS[name]['get_data_fn'](split='train', with_info=True)
		num_classes = info['num_classes']
		# Load labels
		labels = np.load(os.path.join(os.environ['TFDS_DATA_DIR'],f'speech_commands/train_labels.npy'))
		indexes = np.arange(labels.shape[0])
		# Get client partition
		(indexes,labels) = SplitDataset.create_dataset_partition(data=(indexes,labels), cid=int(id), num_clients=num_clients,
																data_skew=0.0, class_skew=0.0, seed=seed)
		# Validation set
		val_ds = None
		num_samples = (indexes.shape[0], None)

		if valid_split>0.0:
			indexes, val_indexes, _, _ = train_test_split(indexes, labels, test_size=valid_split, random_state=seed)
			val_ds = MaskDataset.filter_dataset(ds, indexes=np.sort(val_indexes))
			val_ds = val_ds.map(functools.partial(DATASETS[name]['test_fn'], bins=num_mels),num_parallel_calls=tf.data.AUTOTUNE).\
				batch(1).prefetch(-1)
			num_samples = (indexes.shape[0], val_indexes.shape[0])
		# Train set.
		ds = MaskDataset.filter_dataset(ds, indexes=np.sort(indexes))
		ds = ds.map(functools.partial(DATASETS[name]['train_fn'], bins=num_mels), num_parallel_calls=tf.data.AUTOTUNE)\
				.shuffle(10000,seed,True).batch(batch_size).prefetch(-1)
		ds = (ds,val_ds)

	return ds, num_classes, num_samples

def get_librispeech(batch_size=128, size=-1, seed=0, num_mels=40, reshape_size=None, name='librispeech'):
	# Fix seed for reproducability.
	np.random.seed(seed)
	# Load data
	ds, info = DATASETS[name]['get_data_fn'](split=None, with_info=True)
	ds = ds.take(size).shuffle(10000,seed,True)
	# Take subset of dataset
	ds = ds.map(functools.partial(DATASETS[name]['train_fn'], bins=num_mels),num_parallel_calls=tf.data.AUTOTUNE)\
    		.batch(batch_size, drop_remainder=True).prefetch(-1)
	return ds
