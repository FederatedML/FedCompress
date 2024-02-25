import os
import utils
import argparse
import shutil
import flwr as fl
from time import sleep
import network

parser = argparse.ArgumentParser(description="Federated Model Compression")
parser.add_argument("--random_id", default='0a6809e9cb', type=str, help="unique id of experiment")
parser.add_argument("--rounds", default=30, type=int, help="number of totat federated rounds to run")
parser.add_argument("--timeout", default=None, type=utils.none_or_int, nargs='?', help="maximum seconds of round until timeout")
parser.add_argument("--clients", default=10, type=int, help="number of clients")
parser.add_argument("--cpu_percentage", default=0.5, type=float, help="percentage of cpu resources (cores) to be used by each client")
parser.add_argument("--gpu_percentage", default=0.08, type=float, help="percentage of gpu resources to be used by each client")
parser.add_argument("--participation", default=1.0, type=float, help="participation percentage of clients in each round")
parser.add_argument("--epochs", default=1, type=int, help="number of local client epochs to run per federated round")
parser.add_argument("--batch_size", default=32, type=int, help="batch size for local client training")
parser.add_argument("--learning_rate", default=1e-1, type=float, help="learning rate")
parser.add_argument("--dataset", default="cifar10", type=str, help="dataset name to use for trainig")
parser.add_argument("--model_name", default="resnet20", type=str, help="dataset name to use for trainig")
parser.add_argument("--init_model_fp", default=None, type=utils.none_or_str, nargs='?', help="initialization model path")
parser.add_argument("--results_dir", default="../assets/results", type=str, help="store results directory")
parser.add_argument("--model_dir", default="../assets/saved_models", type=str, help="store model directory")
parser.add_argument("--seed", default=0, type=int, help="seed used to achieve reproducibility")
# Compression Parameters
parser.add_argument("--init_num_clusters", default=8, type=int, help="initial number of clusters for weight clustering")
parser.add_argument("--max_num_clusters", default=20, type=int, help="maximum number of clusters for weight clustering")
parser.add_argument("--cluster_update_step", default=1, type=int, help="search step for optimal number of clusters for weight clustering")
parser.add_argument("--cluster_search_init_rounds", default=1, type=int, help="initial rounds before starting cluster search")
parser.add_argument("--cluster_search_window", default=3, type=int, help="window of metric average for optimal number of clusters for weight clustering")
parser.add_argument("--cluster_search_patience", default=3, type=int, help="patience step for optimal number of clusters for weight clustering")
parser.add_argument("--cluster_search_metric_limit", default=1.0, type=float, help="metric minimum improvement limit for switch number of clusters")
parser.add_argument("--client_compression", dest="client_compression", action="store_true", help="use client-side weight clustering")
parser.add_argument("--client_compression_epochs", default=5, type=int, help="local train epochs with weight clustering")
parser.add_argument("--client_compression_lr", default=1e-4, type=float, help="learning rate for local train epochs with weight clustering")
parser.add_argument("--server_compression", dest="server_compression", action="store_true", help="use server-side ood weight clustering with KD")
parser.add_argument("--server_compression_step", default=1, type=int, help="frequency step for applying server-side ood weight clustering with KD")
parser.add_argument("--server_compression_epochs", default=5, type=int, help="train epochs with server-side ood weight clustering with KD")
parser.add_argument("--server_compression_lr", default=1e-4, type=float, help="learning rate for server-side ood weight clustering with KD")
parser.add_argument("--server_compression_batch", default=256, type=int, help="batch size for server-side ood weight clustering with KD")
parser.add_argument("--server_compression_temperature", default=8.0, type=float, help="temperature for server-side ood weight clustering with KD")
args = parser.parse_args()

model_save_dir_fn = lambda x: os.path.abspath(os.path.join(args.model_dir,f"{args.model_name}_{args.dataset}_{args.server_compression}_{args.client_compression}_{x}_{args.random_id}.h5"))
store_dir_fn = lambda x: os.path.abspath(os.path.join(args.results_dir, f"{args.model_name}_{args.dataset}_{args.server_compression}_{args.client_compression}_{x[0]}_{args.random_id}.{x[1]}"))

def get_clients_config():
	return {
		"lr": args.learning_rate,
		"epochs": args.epochs,
		"compression_lr": args.client_compression_lr,
		"compression_epochs": args.client_compression_epochs,
	}

def get_server_compression_config():
	return {
		'compression_step': args.server_compression_step,
		'epochs':args.server_compression_epochs,
		'learning_rate':args.server_compression_lr,
		'batch_size': args.server_compression_batch,
		'temperature': args.server_compression_temperature,
		'data_loader': utils.create_dataloader_fn(ood=True)[args.dataset],
		'dataset_name': args.dataset,
		'seed': args.seed,
	}

def get_cluster_search_config():
    return {
		"init_num_clusters":args.init_num_clusters,
		"max_num_clusters":args.max_num_clusters,
		"cluster_update_step":args.cluster_update_step,
		"init_rounds": args.cluster_search_init_rounds,
		"window":args.cluster_search_window,
		"patience":args.cluster_search_patience,
		"limit": args.cluster_search_metric_limit,
	}

def create_client(cid):
	import utils
    # Sleep few seconds to allow for GPU to setup for each client
	sleep(int(cid)*0.75)
	# Assign free GPU
	os.environ['CUDA_VSIBLE_DEVICES']="0"
	# Start client
	from client import _Client
	return _Client(int(cid),
		num_clients=args.clients,
		model_loader=network.get_model(args.model_name),
		data_loader=utils.create_dataloader_fn()[args.dataset],
		batch_size = args.batch_size,
		client_compression=args.client_compression,
		seed=args.seed,
	)

def create_server(run_id=0):
	# Assign free GPU
	os.environ['CUDA_VSIBLE_DEVICES']="0"
	# Start server
	from server import _Server
	return _Server(run_id=run_id,
				num_rounds=args.rounds,
				num_clients=args.clients,
				participation=args.participation,
				model_loader=network.get_model(args.model_name),
				data_loader=utils.create_dataloader_fn()[args.dataset],
				init_model_fp=args.init_model_fp,
				model_save_dir=model_save_dir_fn,
				clients_config=get_clients_config(),
				# Compression parameters
				server_compression=args.server_compression,
				client_compression=args.client_compression,
				server_compression_config=get_server_compression_config(),
				cluster_search_config=get_cluster_search_config(),
	)

def main(run_id=0):

	server = create_server(run_id=run_id)
	history = fl.simulation.start_simulation(
		client_fn = create_client,
		server = server,
		num_clients=args.clients,
		ray_init_args= {
			"ignore_reinit_error": True, "include_dashboard": False,
			"dashboard_host": "127.0.0.1", "dashboard_port": 8265,
			# By setting `num_cpus` to match the number of available cores, we ensure that clients are terminated after been executed
			# This way gpu resources are released in every round.
			"num_cpus": min(args.clients,7)
        },
		client_resources={
			"num_cpus":args.cpu_percentage,
			"num_gpus": args.gpu_percentage
		},
		config=fl.server.ServerConfig(num_rounds=args.rounds, round_timeout=args.timeout),
	)
	return history

if __name__ == "__main__":

	parsed_args = '\t' + '\t'.join(f'{k} = {v}\n' for k, v in vars(args).items())
	print('Parameters:')
	print(parsed_args)

	history = main()
	df = utils.store_history(history, args, store_dir_fn=store_dir_fn)
	print(df)
