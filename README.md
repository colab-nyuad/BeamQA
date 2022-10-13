# BeamQA

Implementation of "Question Answering over Incomplete Knowledge Graphs using Path Prediction and Graph Embeddings"


# Instructions 

## Data 

We use two datasets MetaQA and WebQuestionSP
All data folder can be downloaded from https://drive.google.com/file/d/1FK0tIZauEYtkoZ0ET3qYIg2De_DHaQz9/view?usp=sharing.  

#### Graph emeddings 
We generate graph embeddings using Libkge, the graphs and config files are provided. Further instructions on how to train embeddings can be found in LibkGE repository https://github.com/uma-pi1/kge

#### Train-eval MetaQA : 
cd BeamQA/MetaQA 
python main.py --gpu [number] --kg_type half --mode train-BeamQA


#### Train-eval WQSP : 
cd BeamQA/WQSP
python main.py --gpu [number] --hops [num_hops] --kg_type [kg_type] --mode train-BeamQA


### Path generation 
Code for generating paths and Synthetic questions can be found in BeamQA/Path_generation


## Results

## How to cite
