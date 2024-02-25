# Federated Weight Clustering with Adaptive clustering

Federated Learning (FL) is a promising technique for the collaborative training of deep neural networks across multiple devices while preserving data privacy. Despite its potential benefits, FL is hindered by excessive communication costs due to repeated server-client communication during training. To address this challenge, model compression techniques, such as sparsification and weight clustering are applied, which often require modifying the underlying model aggregation schemes or involve cumbersome hyperparameter tuning, with the latter not only adjusts the model's compression rate but also limits model's potential for continuous improvement over growing data. In this paper, we propose FedCompress, a novel approach that combines dynamic weight clustering and server-side knowledge distillation to reduce communication costs while learning highly generalizable models. Through a comprehensive evaluation on diverse public datasets, we demonstrate the efficacy of our approach compared to baselines in terms of communication costs and inference speed.

A complete description of our work can be found in our [paper](https://arxiv.org/pdf/2401.14211.pdf).

## Dependencies

Create a new Python enviroment (virtualenvs, anacoda, etc.) and install all required packages via:

```console
foo@bar:~$ pip install -r requirements.txt
```

## Executing experiments

From the `root` directory of this repo, run:

```console
# Standard FedAvg
foo@bar:~$ ./run.py --datasets cifar10 --method fedavg
# FedAvg + Client-side compression via weight-clustering
foo@bar:~$ ./run.py --datasets cifar10 --method fedavg
# FedCompress (Ours)
foo@bar:~$ ./run.py --datasets cifar10 --method fedavg
```

> **_NOTE:_**  You can configure all federated parameters (i.e. number of federated rounds, etc.,) by adjusting them in the `configs.py` file.

## Reference

If you use this repository, please consider citing:

<pre>@misc{tsouvalas2024communicationefficient,
      title={Communication-Efficient Federated Learning through Adaptive Weight Clustering and Server-Side Distillation},
      author={Vasileios Tsouvalas and Aaqib Saeed and Tanir Ozcelebi and Nirvana Meratnia},
      year={2024},
      eprint={2401.14211},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
</pre>
