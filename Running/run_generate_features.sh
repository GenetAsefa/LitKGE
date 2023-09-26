
python ../generate_numerical_features.py generate_numerical_features --structured_triples=../Datasets/LiWD48K/train2id.txt --attributive_triples=../Datasets/LiWD48K/numerical_literals_short2.txt --ent2id=../Datasets/LiWD48K/entity2id.txt --rel2id=../Datasets/LiWD48K/relation2id.txt --attr2id=../Datasets/LiWD48K/attr2id2.txt --num-walks=10 --walk-length=3 --weighted --directed



python ../generate_numerical_features.py generate_numerical_features --structured_triples=../Datasets/FB15K-237/train2id.txt --attributive_triples=../Datasets/FB15K-237/numerical_literals_short2.txt --ent2id=../Datasets/FB15K-237/entity2id.txt --rel2id=../Datasets/FB15K-237/relation2id.txt --attr2id=../Datasets/FB15K-237/attr2id2.txt --num-walks=80 --walk-length=3 --weighted --directed



python ../generate_numerical_features.py generate_numerical_features --structured_triples=../Datasets/YAGO3-10/train2id.txt --attributive_triples=../Datasets/YAGO3-10/numerical_literals_short2.txt --ent2id=../Datasets/YAGO3-10/entity2id.txt --rel2id=../Datasets/YAGO3-10/relation2id.txt --attr2id=../Datasets/YAGO3-10/attr2id2.txt --num-walks=80 --walk-length=3 --weighted --directed
