# Configuration

## Configuration Form
A configuration file is a yaml file with the following fields. We also provide
the configuration files used in our evaluation.
* Configuration fields for RICC
    + `case`: Identifier for each experiment. You must use a different name for each experiment.
    + `graph`: Name of the graph dataset. (Enron, Facebook, Twitter_small, Twitter_large).
    + `edge_manipulation_cost`: Type of the edge manipulation cost. (equal, uni, cat).
    + `target_node_type`: Type of the target node selection method. (close, rand, lcc).
    + `attack`: Type of the attack. (ENM, NNI).
    + `epoch`: Maximum number of the epoch. We use the term **_iteration_** instead of **_epoch_** in our paper.
    + `lr`: The learning rate. We use the term  **$\epsilon$** in our paper.
    + `buffer`: The buffer ratio. We use the term **$\rho$** in our paper.
    + `sampling_size`: The number of sampled nodes. We use the term **$N$** in our paper.
    + `interval`: The epoch interval for printing and recording metrics during the experiment.
    + `num_modified_edges`: The number of modified edges.
    + `num_added_nodes`: The number of added edges.
    + `num_target_nodes`: The number of target nodes.

* Configuration fields for collective classification
    + `theta`: The absolute value used for prior scores. We use the term  **$\theta$** in our paper.
    + `weight`: The weight value used for collective classification.
    + `iteration`: The maximum number of iterations used for collective classification. We use the term **$T_{max}$** in our paper.
    + `threshold`: The threshold value used to classify Sybil nodes.


## Examples

To reproduce the classification result on the Enron dataset using our default
setting in Table 1, please run:
```
$ python RICC.py --config Enron_equal_close_ENM.yaml
```
```
$ python RICC.py --config Enron_equal_close_NNI.yaml
```


To reproduce the classification result on the Enron graph manipulated using
different attack budgets, as shown in Figures 3 & 4, please run:
```
$ python RICC.py --config Enron_equal_close_ENM_K_80.yaml
```
```
$ python RICC.py --config Enron_equal_close_NNI_N_110.yaml
```


To reproduce the classification result on the Enron graph manipulated using
a different number of target nodes, as shown in Figure 6, please run:
```
$ python RICC.py --config Enron_equal_close_ENM_T_600.yaml
```


To reproduce the classification result on the Enron graph manipulated using
different attack strategies, as shown in Tables 2 & 3, please run:
```
$ python RICC.py --config Enron_equal_lcc_ENM.yaml
```
```
$ python RICC.py --config Enron_equal_rand_ENM.yaml
```
```
$ python RICC.py --config Enron_cat_close_ENM.yaml
```
```
$ python RICC.py --config Enron_uni_close_ENM.yaml
```


If you want to run RICC with the Facebook dataset, please run:
```
$ python RICC.py --config Facebook_equal_close_ENM.yaml
```
```
$ python RICC.py --config Facebook_equal_close_NNI.yaml
```

