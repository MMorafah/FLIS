# FLIS: Clustered Federated Learning via Inference Similarity for Non-IID Data Distribution

The official code of paper ["FLIS: Clustered Federated Learning via Inference Similarity for Non-IID Data Distribution"](https://arxiv.org/abs/2208.09754). </br>
**"Accepted to FL NeurIPS workshop 2022"**.

In this repository, we release the official implementation for FLIS algorithms (FLIS-DC, FLIS-HC). The algorithms are evaluated on 4 datasets (Cifar-100/10, Fashion-MNIST, SVHN) with non-iid label distribution skew (noniid-#label2, noniid-#label3, noniid-labeldir).


## Usage

We provide scripts to run the algorithms, which are put under `scripts/`. Here is an example to run the script:
```
cd scripts
bash flis_dc.sh
bash flis_hc.sh
```
Please follow the paper to modify the scripts for more experiments. You may change the parameters listed in the following table.

The descriptions of parameters are as follows:
| Parameter | Description |
| --------- | ----------- |
| ntrials      | The number of total runs. |
| rounds       | The number of communication rounds per run. |
| num_users    | The number of clients. |
| frac         | The sampling rate of clients for each round. |
| local_ep     | The number of local training epochs. |
| local_bs     | Local batch size. |
| lr           | The learning rate for local models. |
| momentum     | The momentum for the optimizer. |
| model        | Network architecture. Options: `TODO` |
| dataset      | The dataset for training and testing. Options are discussed above. |
| partition    | How datasets are partitioned. Options: `homo`, `noniid-labeldir`, `noniid-#label1` (or 2, 3, ..., which means the fixed number of labels each party owns). |
| datadir      | The path of datasets. |
| logdir       | The path to store logs. |
| log_filename | The folder name for multiple runs. E.g., with `ntrials=3` and `log_filename=$trial`, the logs of 3 runs will be located in 3 folders named `1`, `2`, and `3`. |
| alg          | Federated learning algorithm. Options are discussed above. |
| beta         | The concentration parameter of the Dirichlet distribution for heterogeneous partition. |
| local_view   | If true puts local test set for each client |
| gpu          | The IDs of GPU to use. E.g., `TODO` |
| print_freq   | The frequency to print training logs. E.g., with `print_freq=10`, training logs are displayed every 10 communication rounds. |

<!---
## Results
{% comment %} 
### Partition: `non-iid-#label2`
| Algorithm      | FMNIST | CIFAR-10 | SVHN   |
| ---            | ---    | ---      | ---    |
| `FedIS-DC`     | 97.96% | 91.47%   | 95.63% |
| `FedIS-HC`    | 97.41% | 84.06%   | 93.57% |


### Partition: `non-iid-#label3`
| Algorithm      | FMNIST | CIFAR-10 | SVHN   |
| ---            | ---    | ---      | ---    |
| `FedIS-DC`     | 96.03% | 84.36%   | 93.30% |
| `FedIS-HC`    | 94.48% | 75.60%   | 90.36% |


### Partition: `non-iid-labeldir(beta=0.1)`
| Algorithm      | FMNIST | CIFAR-10 | SVHN   |
| ---            | ---    | ---      | ---    |
| `FedIS-DC`     | 86.89% | 62.96%   | 82.02% |
| `FedIS-HC`    | 79.63% | 54.11%   | 66.89% |
{% endcomment %}
-->

## Citation 
Please cite our work if you find it relavent to your research and used our implementations. 
```
@article{morafah2022flis,
  title={FLIS: Clustered Federated Learning via Inference Similarity for Non-IID Data Distribution},
  author={Morafah, Mahdi and Vahidian, Saeed and Wang, Weijia and Lin, Bill},
  journal={arXiv preprint arXiv:2208.09754},
  year={2022}
}
```

## Acknowledgements

Some parts of our code and implementation has been adapted from [NIID-Bench](https://github.com/Xtra-Computing/NIID-Bench) repository.

## Contact 
If you had any questions, please feel free to contact me at mmorafah@eng.ucsd.edu
