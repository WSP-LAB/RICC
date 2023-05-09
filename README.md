# RICC
RICC is a robust collective classification framework designed to identify Sybil
accounts on online social networks. We observed that the classification results
for adversarial Sybil accounts often significantly change when deploying a new
training set different from the original training set. Leveraging this
observation, RICC achieves robustness against state-of-the-art adversarial
attacks by stabilizing classification results across different training sets
randomly sampled in each round. For more details, please refer to our
[paper](https://leeswimming.com/papers/shin-www23.pdf), "RICC: Robust Collective
Classification of Sybil Accounts", which appeared in The Web Conference (WWW)
2023.

## Requirements
We implemented RICC in Python and tested on a machine running Ubuntu 20.04.5
LTS and Python 3.8. To get ready for running RICC, please install the
dependencies by running the following commands.

```
$ git clone https://github.com/WSP-LAB/RICC.git
$ cd RICC
$ pip install -r requirements.txt
```

## Dataset
We provide the Enron and Facebook datasets in the `dataset` directory. These
datasets are from the Stanford Large Network Dataset Collection (SNAP). You can
access the original datasets from [here](https://snap.stanford.edu/data/).

For the **Twitter_small** and **Twitter_large** datasets, we refer the users to
the following links.

> Twitter_small dataset :
> [https://success.cse.tamu.edu/releases/](https://success.cse.tamu.edu/releases/)
> <br>
> Twitter_large dataset :
> [http://wangbinghui.net/dataset.html](http://wangbinghui.net/dataset.html)

## Usage

### Configuration
Please refer to this
[link](https://github.com/WSP-LAB/RICC/blob/master/configs/README.md)
for writing a configuration file.

### Execution
To run RICC, execute the script `RICC.py` by passing the configuration file as
an argument.
```
$ cd RICC/script
$ python RICC.py --config [Name of configuration file]
```

For example, to reproduce the classification results on the Enron dataset using
our default settings in Table 1, please execute the following command:
```
$ python RICC.py --config Enron_equal_close_ENM.yaml
```
```
$ python RICC.py --config Enron_equal_close_NNI.yaml
```
For more detailed examples, please refer to this
[link](https://github.com/WSP-LAB/RICC/blob/master/configs/README.md).

## Authors
This research project has been conducted by [WSP Lab](https://wsp-lab.github.io/)
at KAIST.

* [Dongwon Shin](https://godeastone.github.io/)
* [Suyoung Lee](https://leeswimming.com/)
* [Sooel Son](https://sites.google.com/site/ssonkaist/home)

## Citation
To cite our paper:
```bibtex
@INPROCEEDINGS{shin:www:2023,
  author = {Dongwon Shin and Suyoung Lee and Sooel Son},
  title = {{RICC}: Robust Collective Classification of Sybil Accounts},
  booktitle = {Proceedings of the ACM Web Conference},
  pages = {2329--2339},
  year = 2023
}
```
