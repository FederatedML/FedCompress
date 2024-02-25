import tensorflow as tf
from dataclasses import dataclass

@dataclass(frozen=True)
class Params:
	conv_padding: str = "same"
	batchnorm_center: bool = True
	batchnorm_scale: bool = False
	batchnorm_epsilon: float = 1e-4
	classifier_activation: str = "linear"

def _batch_norm(name, params):
	def _bn_layer(layer_input):
		return tf.keras.layers.BatchNormalization(name=name,
			center=params.batchnorm_center,
			scale=params.batchnorm_scale,
			epsilon=params.batchnorm_epsilon)(layer_input)
	return _bn_layer

def _conv(name, kernel, stride, filters, params):
	def _conv_layer(layer_input):
		output = tf.keras.layers.Conv2D(
					name="{}/conv".format(name),
					filters=filters, kernel_size=kernel, strides=stride,
					padding=params.conv_padding, use_bias=False, activation=None)(layer_input)
		output = _batch_norm("{}/conv/bn".format(name), params)(output)
		output = tf.keras.layers.ReLU(name="{}/relu".format(name))(output)
		return output
	return _conv_layer

def _separable_conv(name, kernel, stride, filters, params):
	def _separable_conv_layer(layer_input):
		output = tf.keras.layers.DepthwiseConv2D(
					name="{}/depthwise_conv".format(name),
					kernel_size=kernel, strides=stride, depth_multiplier=1,
					padding=params.conv_padding, use_bias=False, activation=None)(layer_input)
		output = _batch_norm("{}/depthwise_conv/bn".format(name), params)(output)
		output = tf.keras.layers.ReLU(name="{}/depthwise_conv/relu".format(name))(output)
		output = tf.keras.layers.Conv2D(
					name="{}/pointwise_conv".format(name),
					filters=filters, kernel_size=(1, 1), strides=1,
					padding=params.conv_padding, use_bias=False, activation=None)(output)
		output = _batch_norm("{}/pointwise_conv/bn".format(name), params)(output)
		output = tf.keras.layers.ReLU(name="{}/pointwise_conv/relu".format(name))(output)
		return output
	return _separable_conv_layer

_YAMNET_LAYER_DEFS = [
	# (layer_function, kernel, stride, num_filters)
	(_conv,          [3, 3], 2,   32),
	(_separable_conv, [3, 3], 1,   64),
	(_separable_conv, [3, 3], 2,  128),
	(_separable_conv, [3, 3], 1,  128),
	(_separable_conv, [3, 3], 2,  256),
	(_separable_conv, [3, 3], 1,  256),
	(_separable_conv, [3, 3], 2,  512),
	(_separable_conv, [3, 3], 1,  512),
	(_separable_conv, [3, 3], 1,  512),
	(_separable_conv, [3, 3], 1,  512),
	(_separable_conv, [3, 3], 1,  512),
	(_separable_conv, [3, 3], 1,  512),
	(_separable_conv, [3, 3], 2, 1024),
	(_separable_conv, [3, 3], 1, 1024)
]

"""Define the core YAMNet model in Keras."""
def YAMNet(inputs, num_classes, params):
	net = inputs
	for (i, (layer_fun, kernel, stride, filters)) in enumerate(_YAMNET_LAYER_DEFS):
		net = layer_fun("layer{}".format(i + 1), kernel, stride, filters, params)(net)
	x = tf.keras.layers.GlobalAveragePooling2D(name='GMP_layer')(net)
	logits = tf.keras.layers.Dense(units=num_classes, use_bias=True, activation=params.classifier_activation)(x)
	return logits

"""Defines the YAMNet."""
def create_yamnet_model(input_shape=(None,0,1), num_classes=12, weights_dir=None):
	params = Params()
	inputs = tf.keras.layers.Input(input_shape, dtype=tf.float32)
	predictions = YAMNet(inputs=inputs, num_classes=num_classes, params=params)
	model = tf.keras.Model(inputs=inputs, outputs=predictions, name="YAMNet")
	if weights_dir is not None: model.load_weights(weights_dir).expect_partial()
	return model