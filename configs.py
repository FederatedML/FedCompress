import os
import uuid
from dataclasses import dataclass, field
from typing import List

@dataclass
class Params:
	# Approach
	method: str = 'standard'
	# Datasets
	dataset: str = "cifar10"
	ood_dataset: str = "stylegan"
	# Model
	model: str = 'resnet20'
	init_model_fp: str = ''
	# Training params
	client_epochs: int = 5
	client_compress_epochs: int = 5
	client_batch_size: int = 128
	server_epochs: int = 10
	server_batch_size: int = 256
	client_normal_lr: float = 1e-3
	client_compress_lr: float = 5e-4
	server_compress_lr: float = 1e-4
	server_compress_temperature: float = 2.0
	# Cluster params
	cluster_space: List = field(default_factory=lambda: [8,30])
	cluster_init_patience = 3
	cluster_patience = 4
	cluster_window = 3
	cluster_limit = 1
	cluster_step = 1
	# Federated params
	num_rounds: int = 30
	num_clients: int = 20
	participation: float = 1.0
	cpu_usage: float = 1.0
	gpu_usage: float = 0.08
	seed: int = 0
	# Logging params
	random_id: str = str(uuid.uuid4())[-10:]
	results_dir: str = os.path.abspath('./assets/results')
	model_dir: str = os.path.abspath('./assets/saved_models')

	def __post_init__(self):
		# Adjust batch size
		if self.dataset in ['spcm']:
			self.client_batch_size = 128
			self.server_batch_size = 256
		# Adjust ood data
		if self.dataset in ['spcm']:
			self.ood_dataset = "librispeech"
			self.model = 'yamnet'
		# Adjust local training epochs
		if self.method == 'standard':
			self.client_epochs += self.client_compress_epochs
			self.client_compress_epochs = 0
		if self.method in ['standard','client']:
			self.server_epochs = 0
		# Keep this last
		self.init_model_fp = os.path.join(os.path.abspath('./assets/init_models/'),"{}_{}_random_1.h5".format(self.model,self.dataset))

