# LitKGE
LitKGE: Improving numeric LITerals based Knowledge Graph Embedding models

## File descriptions:
LitWD48K:

1. Datasets/LitWD48K/features.txt: contains the features generated for the entities in LitWD48K
2. Datasets/LitWD48K/numeric_literals_feature.txt: contains the combination of literals and the features generated for the entities, in LitWD48K


FB15K-237:

1. Datasets/FB15K-237/features.txt: contains the features generated for the entities in FB15K-237
2. Datasets/FB15K-237/numeric_literals_feature.txt: contains the combination of literals and the features generated for the entities, in FB15K-237


YAGO3-10:
1. Datasets/YAGO3-10/features.txt: contains the features generated for the entities in YAGO3-10
2. Datasets/YAGO3-10/numeric_literals_feature.txt: contains the combination of literals and the features generated for the entities, in YAGO3-10


## Set-up
conda create --name LitKGE python=3.8
conda install pyg -c pyg -c conda-forge
Install PyTorch
Install other requirements: pip install -r requirements.txt
Download node2vec code from https://github.com/aditya-grover/node2vec
Download LiteralE code from https://github.com/SmartDataAnalytics/LiteralE

## Instructions
- In order to generate features look at Running/run_generate_feature.sh
- Use preprocess_num_lit.py to preprocess the numerical literals and features.
- Then, follow the steps in https://github.com/SmartDataAnalytics/LiteralE to run the models using the hyperparameters given in the paper 
