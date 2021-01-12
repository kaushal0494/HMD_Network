# HMD_Net
This repository consists of the implementation of the model HMD_Network paper ([link](https://dl.acm.org/doi/pdf/10.1145/3340531.3411997)) which is accepted in CIKM'20. We modified the implementation of [OpenNMT-py](https://github.com/OpenNMT/OpenNMT-py) to adopt for distractor generation task. 

# Dependencies
- python 3.6
- pytorch=1.2.0
- List of all dependencies can be found in environment.yml

# Quick Start
- Install all the dependencies by *conda env create -f environment.yml* (some of the packages installed through *pip* so, need to install them separately)
- Run instructions are same as OpenNMT-py run instructions 
  - *./script/preprocess.sh* - load and preprocess the data also, load glove embeddings
  - *./script/train.sh* - to train the model
  - *./script/translate.sh* - to generated the distractors
  

