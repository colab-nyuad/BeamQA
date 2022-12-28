# BeamQA

Implementation of "Question Answering over Incomplete Knowledge Graphs using Path Prediction and Graph Embeddings"


# Instructions 
### Quick start
```sh
# retrieve and install project 
git clone https://github.com/colab-nyuad/BeamQA

# install requirements
pip install -r requirements.txt
git clone https://github.com/uma-pi1/kge.git
cd kge
pip install -e .
```

## Data 

We use two datasets MetaQA and WebQuestionSP
The Data folder can be downloaded from [this link](https://drive.google.com/file/d/1oEDSK2e1R67L1fee4YhxGnrwkArSe162/view?usp=sharing).*  

#### Graph emeddings 
We generate graph embeddings using Libkge, the graphs and config files are provided. Further instructions on how to train embeddings can be found in LibkGE repository https://github.com/uma-pi1/kge


#### Train-eval MetaQA : 
```sh
cd BeamQA/MetaQA 
python main.py --gpu [number] --kg_type half --mode train-BeamQA
```
Or 
```sh
bash train.sh
```


#### Train-eval WQSP : 
```sh
cd BeamQA/WQSP
python main.py --gpu [number] --hops [num_hops] --kg_type [kg_type] --mode train-BeamQA
```
Or uncomment lines for WQSP
```sh
bash train.sh
```

### Path generation 
Code for generating paths and Synthetic questions can be found in BeamQA/Path_generation


## Results

## How to cite
