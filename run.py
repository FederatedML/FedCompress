import os
import subprocess
import argparse
from configs import Params
os.environ['TF_CPP_MIN_LOG_LEVEL']="3"
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['CUDA_DEVICE_ORDER']="PCI_BUS_ID"

def create_parser():
    # Create the parser
    parser = argparse.ArgumentParser(description='FedCompress arguments.')
    parser.add_argument('--datasets', nargs='+', default=['cifar10'], choices=['cifar10'], help="List of datasets to use. Default is ['cifar10'].")
    parser.add_argument('--methods', nargs=1, default=['both'], choices=['fedcompress', 'client', 'fedavg'], help="Method to use. One of 'fedcompress', 'client', 'standard'. Default is 'both'.")
    return parser

def main(args):
	datasets = args.datasets
	methods = args.methods
	rnds = {'cifar10':100,'cifar100':500,'pathmnist':100,'spcm':50,'voxforge':40}

	for dataset in datasets:
		for method in methods:
			assert method in ['fedcompress','client','fedavg'], 'Parameter `method` must be one of [`fedcompress`, `client`, `fedavg`]. Provided {}.'.format(method)
			params = Params(method=method, dataset=dataset, num_rounds=rnds[dataset])

			call_cmd = ["python3", "./src/main.py",
				"--random_id", params.random_id,
				"--rounds", str(params.num_rounds),
				"--clients",str(params.num_clients),
				"--dataset", dataset,
				"--model_name", params.model,
				"--participation", str(params.participation),
				"--cpu_percentage", str(params.cpu_usage),
				"--gpu_percentage", str(params.gpu_usage),
				"--seed", str(params.seed),
				"--server_compression" if method in ['fedcompress'] else None,
				"--client_compression" if method in ['fedcompress', 'client'] else None,
				# Training params
 				"--epochs", str(params.client_epochs),
				"--learning_rate", str(params.client_normal_lr),
				"--client_compression_epochs", str(params.client_compress_epochs),
				"--client_compression_lr", str(params.client_compress_lr),
				"--batch_size", str(params.client_batch_size),
				"--server_compression_epochs", str(params.server_epochs),
				"--server_compression_lr", str(params.server_compress_lr),
				"--server_compression_temperature", str(params.server_compress_temperature),
				"--server_compression_batch", str(params.server_batch_size),
				# Logging
				"--results_dir", params.results_dir,
				"--model_dir", params.model_dir,
				"--init_model_fp", params.init_model_fp,
				# Compression parameters
				"--init_num_clusters", str(params.cluster_space[0]),
				"--max_num_clusters", str(params.cluster_space[1]),
				"--cluster_update_step", str(params.cluster_step),
				"--cluster_search_init_rounds", str(params.cluster_init_patience),
				"--cluster_search_window", str(params.cluster_window),
				"--cluster_search_patience", str(params.cluster_patience),
				"--cluster_search_metric_limit", str(params.cluster_limit),
			]
			call_cmd = [c for c in call_cmd if c is not None]
			print('\nCalling command:')
			print(' '.join(call_cmd),'\n')
			subprocess.call(call_cmd)

if __name__ == "__main__":
	args = create_parser().parse_args()
	main(args)