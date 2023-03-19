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
# This script is used to load and preprocess the data, as well as load GloVe embeddings
bash script/preprocess.sh

# This script is used to train the model
bash script/train.sh 

# This script is used to generate the distractors
bash script/translate.sh 
```

# License
This project is licensed under the MIT License. Please see the LICENSE file for more details.

# Citation
```
@inproceedings{10.1145/3340531.3411997,
author = {Maurya, Kaushal Kumar and Desarkar, Maunendra Sankar},
title = {Learning to Distract: A Hierarchical Multi-Decoder Network for Automated Generation of Long Distractors for Multiple-Choice Questions for Reading Comprehension},
year = {2020},
isbn = {9781450368599},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3340531.3411997},
doi = {10.1145/3340531.3411997},
abstract = {The task of generating incorrect options for multiple-choice questions is termed as distractor generation problem. The task requires high cognitive skills and is extremely challenging to automate. Existing neural approaches for the task leverage encoder-decoder architecture to generate long distractors. However, in this process two critical points are ignored - firstly, many methods use Jaccard similarity over a pool of candidate distractors to sample the distractors. This often makes the generated distractors too obvious or not relevant to the question context. Secondly, some approaches did not consider the answer in the model, which caused the generated distractors to be either answer-revealing or semantically equivalent to the answer.In this paper, we propose a novel Hierarchical Multi-Decoder Network (HMD-Net) consisting of one encoder and three decoders, where each decoder generates a single distractor. To overcome the first problem mentioned above, we include multiple decoders with a dis-similarity loss in the loss function. To address the second problem, we exploit richer interaction between the article, question, and answer with a SoftSel operation and a Gated Mechanism. This enables the generation of distractors that are in context with questions but semantically not equivalent to the answers. The proposed model outperformed all the previous approaches significantly in both automatic and manual evaluations. In addition, we also consider linguistic features and BERT contextual embedding with our base model which further push the model performance.},
booktitle = {Proceedings of the 29th ACM International Conference on Information &amp; Knowledge Management},
pages = {1115–1124},
numpages = {10},
keywords = {natural language generation, distractor generation, question-answering},
location = {Virtual Event, Ireland},
series = {CIKM '20}
}
```
