# IDGC

## Abstract
To extract knowledge from a dataset of unlabelled graphs, we aim to conduct the task of interpretable graph-level clustering, which aims to find good clusters of graphs and also gain useful insights into the clustering result by interpreting why each graph is allocated to its corresponding cluster. In this pa- per, we successfully tackle this task by developing an interpretable deep graph-level clustering (IDGC) framework, which not only achieves good clustering performance, but also provides insightful interpretations on the clustering result. Extensive experiments on six benchmark datasets demonstrate the outstanding performance of our method.



## Requirements
```
pytorch                   1.8.0             
torch-geometric           2.0.2
conda install -c pytorch faiss-cpu=1.7.4 mkl=2021 blas=1.0=mkl
pip install matplotlib
pip install pandas
pip install soyclustering
pip install rdkit
```
## Usage

You can run IDGC by
```
python -m models.train
```

You can modify configures in utils/Configures.py
 
