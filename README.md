# HMD_Network
Welcome to the HMD_Net repository!

This repository contains the implementation of the HMD_Network paper titled [Learning to Distract: A Hierarchical Multi-Decoder Network for Automated Generation of Long Distractors for Multiple-Choice Questions for Reading Comprehension](https://dl.acm.org/doi/pdf/10.1145/3340531.3411997), which was published in the Conference on Information and Knowledge Management (CIKM) 2020. The implementation is based on the [OpenNMT](https://github.com/OpenNMT/OpenNMT-py) repository and adapted for the distractor generation task.

# Dataset
RACE_DG and RACE_DG+ datasets can be downloaded from [here](https://iith-my.sharepoint.com/:f:/g/personal/cs18resch11003_iith_ac_in/Ep53QDxqDfFKqekYccbOPAkBjmEyZxirbGu52x-3aNSPUA?e=15HY9m)

# Dependencies
```
- python 3.6
- pytorch=1.2.0
- List of all dependencies can be found in environment.yml
```
# Quick Start
- Install all the dependencies by *conda env create -f environment.yml* (some of the packages installed through *pip* so, need to install them separately)
- Run instructions are same as OpenNMT-py run instructions 
  ```
  - *./script/preprocess.sh* - load and preprocess the data and load glove embeddings as well
  - *./script/train.sh* - train the model
  - *./script/translate.sh* - generated the distractors
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
