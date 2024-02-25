import logging
import flwr as fl
import tensorflow as tf
import compress
import strategy
import utils
import pandas as pd

class _Server(fl.server.Server):

	def __init__(self, model_loader, data_loader, num_rounds, run_id=0,
			num_clients=10, participation=1.0,
			init_model_fp=None, model_save_dir='./',
			server_compression=True, client_compression = False,
			server_compression_config={"compression_step":1, "learning_rate":1e-4, "epochs":10},
			cluster_search_config = {
				"init_num_clusters":8, "max_num_clusters":20, "cluster_update_step":1,
				"init_rounds":1, "window":3, "patience":3, "limit": 1,
			},
			clients_config={"learning_rate":1e-4, "epochs":1},
			log_level=logging.INFO,
		):
		self.run_id = run_id
		(self.data, self.valid_data), self.num_classes, self.num_samples = data_loader(return_eval_ds=True)
		self.model_loader = model_loader
		self.input_shape = self.data.element_spec[0].shape
		self.init_model_fp = init_model_fp
		self.clients_config = clients_config
		self.num_clients = num_clients
		self.participation = participation
		self.set_strategy(self)
		self._client_manager = fl.server.client_manager.SimpleClientManager()
		self.max_workers = None
		self.num_rounds = num_rounds
		self.model_save_dir = model_save_dir
		logging.getLogger("flower").setLevel(log_level)

		# Self-compress parameters
		self.server_compression = server_compression
		self.client_compression = client_compression
		self.server_compression_config = server_compression_config
		self.cluster_search_config = cluster_search_config
		self.compression_rnds = utils.set_compression_rnds(server_compression_config['compression_step'], num_rounds)
		# Local Variables
		self.num_clusters = cluster_search_config['init_num_clusters']
		self.compression = False
		self.compression_metric = 'val_accuracy'
		self.compression_metric_dict = {self.compression_metric:[], 'nb_clusters':[], 'val_score':[]}

	def _is_final_round(self,rnd):
		return self.num_rounds==rnd

	" Get number of epochs to train on ood data."
	def _num_compression_epochs(self, rnd):
		# NOTE: To recover accuracy from large compression, we might consider to increase it gradually.
		return self.server_compression_config['epochs']

	" Set number of clusters based on allowed performance drop."
	def set_num_clusters(self, metrics):

		if self.client_compression or self.server_compression:
			# Check if metric for cluster selection exists
			if self.compression_metric in metrics.keys():
				metric = metrics[self.compression_metric]
				# Cluster reduction check
				flag = utils.compression_flag(df=pd.DataFrame(self.compression_metric_dict),
						init_num_clusters=self.cluster_search_config["init_num_clusters"],
						max_num_clusters=self.cluster_search_config["max_num_clusters"],
						init_rounds=self.cluster_search_config["init_rounds"],
						window=self.cluster_search_config["window"],
						patience=self.cluster_search_config["patience"],
						limit=self.cluster_search_config["limit"]/100,)

				if flag: self.num_clusters+=self.cluster_search_config["cluster_update_step"]
				print(f"[Server] - Best Score: {max(self.compression_metric_dict[self.compression_metric]):0.4f}, Current Score: {metric:0.4f}, Flag: {flag}.")
				return flag
		return False

	" Set best-seen performance across training."
	def set_performance(self, metrics):
		if self.compression_metric in metrics.keys():
			self.compression_metric_dict[self.compression_metric].append(metrics[self.compression_metric])
			self.compression_metric_dict['val_score'].append(metrics['val_score'])
			self.compression_metric_dict['nb_clusters'].append(self.num_clusters)

	" Set the max_workers used by ThreadPoolExecutor. "
	def set_max_workers(self, *args, **kwargs):
		return super(_Server, self).set_max_workers(*args, **kwargs)

	" Set server-side model aggregation strategy. "
	def set_strategy(self, *_):
		self.strategy = strategy.FedCustom(
			min_available_clients=self.num_clients,
			fraction_fit=self.participation,
			min_fit_clients=int(self.participation*self.num_clients),
			fraction_evaluate=0.0,
			min_evaluate_clients=0,
			evaluate_fn=self.get_evaluation_fn(),
			on_fit_config_fn=self.get_client_config_fn(),
			initial_parameters=self.get_initial_parameters(),
			fit_metrics_aggregation_fn=utils.weighted_average_train_metrics,
		)

	" Return ClientManager. "
	def client_manager(self, *args, **kwargs):
		return super(_Server, self).client_manager(*args, **kwargs)

	" Get model parameters. "
	def get_parameters(self, config={}):
		return self.model.get_weights()

	" Set model parameters"
	def set_parameters(self, parameters, config={}):
		if not hasattr(self, 'model'):
			self.model = self.model_loader(input_shape=self.input_shape[1:],num_classes=self.num_classes)
		self.model.compile(metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')])
		if parameters is not None:
			self.model.set_weights(parameters)

	"  Get initial model parameters to be used for training."
	def get_initial_parameters(self, *_):
		if self.init_model_fp is not None:
			self.init_weights = tf.keras.models.load_model(self.init_model_fp, compile=False).get_weights()
		else:
			self.init_weights = self.model_loader(input_shape=self.input_shape[1:], num_classes=self.num_classes).get_weights()
		return fl.common.ndarrays_to_parameters(self.init_weights)

	" Get evaluation function to perform server-side evalation."
	def get_evaluation_fn(self):
		def evaluation_fn(rnd, parameters, config):

			metrics=None
			# Add train model information
			results = utils.set_train_metrics(config)
			results['num_clusters'] = self.num_clusters
			# Update model parameters
			self.set_parameters(parameters, config)

			# Centralized evaluation
			metrics = self.model.evaluate(self.data, verbose=0)
			results['model_size'] = utils.get_gzipped_model_size_from_model(self.model)
			results["accuracy"] = metrics[1]
			if self._is_final_round(rnd):
				self.model.save(self.model_save_dir(False), include_optimizer=False)
			print(f"[Server] - Round {rnd}: For {self.num_clusters} clusters, accuracy {metrics[1]:0.4f} " +
				f"(Val. Accuracy: {results['val_accuracy'] if 'val_accuracy' in results.keys() else 0.0 :0.4f}).")

			# Update best-seen performance (BEFORE setting number of clusters!)
			self.set_performance(results)
			# NOTE: Update number of clusters based on round performance
			results['compression'] = self.set_num_clusters(results)

			# Self compression with OOD + Re-evaluation
			if self.server_compression:
				if rnd in self.compression_rnds and self._num_compression_epochs(rnd)>0:
					self.model = compress.self_compress(
						self.model, self.num_classes,
						model_loader=self.model_loader,
						data_loader=self.server_compression_config['data_loader'],
						data_shape=self.data.element_spec[0].shape,
						nb_clusters=self.num_clusters,
						epochs=self._num_compression_epochs(rnd),
						batch_size=self.server_compression_config['batch_size'],
						learning_rate=self.server_compression_config['learning_rate'],
						temperature=self.server_compression_config['temperature'],
						seed=self.server_compression_config['seed'])
					metrics = self.model.evaluate(self.data, verbose=0)
					results['compressed_model_size'] = utils.get_gzipped_model_size_from_model(self.model)
					results['compressed_accuracy'] = metrics[1]

					if self._is_final_round(rnd):
						self.model.save(self.model_save_dir(True), include_optimizer=False)

			print(f"[Server] - Round {rnd}: Next training round will be executed with {self.num_clusters} clusters (Compression {results['compression']}).")
			return (metrics[0], results), self.get_parameters()
		return evaluation_fn

	" Get clients fit configuration function."
	def get_client_config_fn(self):
		def get_on_fit_config_fn(rnd):
			self.clients_config["round"] = rnd
			self.clients_config["num_clusters"] = self.num_clusters
			return self.clients_config
		return get_on_fit_config_fn
