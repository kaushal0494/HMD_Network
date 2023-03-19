# HMD_Network
Welcome to the HMD_Net repository!

This repository contains the implementation of the HMD_Network paper titled [Learning to Distract: A Hierarchical Multi-Decoder Network for Automated Generation of Long Distractors for Multiple-Choice Questions for Reading Comprehension](https://dl.acm.org/doi/pdf/10.1145/3340531.3411997), which was published in the Conference on Information and Knowledge Management (CIKM) 2020. The implementation is based on the [OpenNMT](https://github.com/OpenNMT/OpenNMT-py) repository and adapted for the distractor generation task.

# Dataset
RACE_DG and RACE_DG+ datasets can be downloaded from [here](https://iith-my.sharepoint.com/:f:/g/personal/cs18resch11003_iith_ac_in/Ep53QDxqDfFKqekYccbOPAkBjmEyZxirbGu52x-3aNSPUA?e=15HY9m)

# Dependencies

```
- python 3.6
- pytorch=1.2.0
```
All other dependencies list can be found in `environment.yml`

# Quick Start
To get started with the HMD_Net model, please follow the steps below:

- Install all the dependencies by running the command `conda env create -f environment.yml`. Some of the packages will be installed through pip, so you may need to install them separately.

- The run instructions for the HMD_Net model are the same as the OpenNMT-py run instructions. Please follow the steps below for model training  and generate:
```
# This script is used to load and preprocess the data, as well as load GloVe embeddings.
* ./script/preprocess.sh

# This script is used to train the model
* ./script/train.sh 

# This script is used to generate the distractors
* ./script/translate.sh 
```

# License
This project is licensed under the MIT License. Please see the LICENSE file for more details.

# Citation
```
@inproceedings{maurya2020learning,
  title={Learning to Distract: A Hierarchical Multi-Decoder Network for Automated Generation of Long Distractors for Multiple-Choice Questions for Reading Comprehension},
  author={Maurya, Kaushal Kumar and Desarkar, Maunendra Sankar},
  booktitle={Proceedings of the 29th ACM International Conference on Information \& Knowledge Management},
  pages={1115--1124},
  year={2020}
}
```
