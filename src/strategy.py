import flwr as fl
import logging

class FedCustom(fl.server.strategy.fedavg.FedAvg):

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.train_metrics_aggregated = {}
		self.parameters = None

	"""Configure the next round of training."""
	def configure_fit(self, server_round, parameters, client_manager):
		config = {}
		if self.on_fit_config_fn is not None: # Custom fit config function provided
			config = self.on_fit_config_fn(server_round)
		if self.parameters is not None:
			parameters = self.parameters
		fit_ins = fl.common.FitIns(parameters, config)
		# Sample clients
		sample_size, min_num_clients = self.num_fit_clients(client_manager.num_available())
		clients = client_manager.sample(num_clients=sample_size, min_num_clients=min_num_clients)
		# Return client/config pairs
		return [(client, fit_ins) for client in clients]

	"""Aggregate fit results using weighted average."""
	def aggregate_fit(self, server_round, results, failures):
		if not results:
			return None, {}
		# Do not aggregate if there are failures and failures are not accepted
		if not self.accept_failures and failures:
			return None, {}

		# Convert results
		weights_results = [(fl.common.parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples) for _, fit_res in results]
		parameters_aggregated = fl.common.ndarrays_to_parameters(fl.server.strategy.aggregate.aggregate(weights_results))

        # Aggregate custom metrics if aggregation fn was provided
		metrics_aggregated = {}
		if self.fit_metrics_aggregation_fn:
			fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
			metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
			self.train_metrics_aggregated = metrics_aggregated
		elif server_round == 1:  # Only log this warning once
			fl.common.logger.log(logging.WARNING, "No fit_metrics_aggregation_fn provided")
		return parameters_aggregated, metrics_aggregated

	"""Evaluate model parameters using an evaluation function."""
	def evaluate(self, server_round, parameters):
		if self.evaluate_fn is None:
			# No evaluation function provided
			return None
		parameters_ndarrays = fl.common.parameters_to_ndarrays(parameters)
		eval_res, new_parameters = self.evaluate_fn(server_round, parameters_ndarrays, config=self.train_metrics_aggregated)
		self.parameters = fl.common.ndarrays_to_parameters(new_parameters)
		if eval_res is None:
			return None
		loss, metrics = eval_res
		return loss, metrics