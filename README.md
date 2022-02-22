# FLIS

The code of paper [FLIS: Clustered Federated Learning via Inference Similarity for Non-IID Data Distribution].

In this repository, we implemented the FLIS algorithms (FLIS-DC, FLIS-HC). The algorithms are evaluated on 4 datasets (Cifar-100/10, Fashion-MNIST, SVHN) with label distribution skew (noniid-#label2, noniid-#label3, noniid-labeldir).

Example scripts to run the code are provided under `scripts/`. Please follow the paper to modify the scripts for more experiments. You may change the parameters listed in the following table.


| Parameter                      | Description                                 |
| ----------------------------- | ---------------------------------------- |
| `dir` | The path to store the logs. |
| `rounds` | The number of rounds to train the model |
| `dataset`      | Dataset to use. Options: `cifar10`, `fmnist`, `svhn`. |
| `partition`    | The partition way. Options: `noniid-labeldir`, `noniid-#label1` (or 2, 3, ..., which means the fixed number of labels each party owns). |
| `beta` | The concentration parameter of the Dirichlet distribution for heterogeneous partition. |
| `datadir` | The path of the dataset. |
| `savedir` | The path to store the final results and models. |
| `cluster_alpha` | The clustering threshold, default = `0.5`. |
| `gpu` | The ID of GPU to run the program, default = `0`. |


## Results

{% comment %} 
### Partition: `non-iid-#label2`
| Algorithm      | FMNIST | CIFAR-10 | SVHN   |
| ---            | ---    | ---      | ---    |
| `FedIS-HT`     | 97.96% | 91.47%   | 95.63% |
| `FedIS-AHC`    | 97.41% | 84.06%   | 93.57% |


### Partition: `non-iid-#label3`
| Algorithm      | FMNIST | CIFAR-10 | SVHN   |
| ---            | ---    | ---      | ---    |
| `FedIS-HT`     | 96.03% | 84.36%   | 93.30% |
| `FedIS-AHC`    | 94.48% | 75.60%   | 90.36% |


### Partition: `non-iid-labeldir(beta=0.1)`
| Algorithm      | FMNIST | CIFAR-10 | SVHN   |
| ---            | ---    | ---      | ---    |
| `FedIS-HT`     | 86.89% | 62.96%   | 82.02% |
| `FedIS-AHC`    | 79.63% | 54.11%   | 66.89% |
{% endcomment %}
