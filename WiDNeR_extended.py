from argparse import ArgumentParser
import numpy as np
import pandas as pd
import random, math
import networkx as nx
import logging, os
from dateutil.parser import parse
from WiDNeR_Light import WiDNeR_Light
pd.options.mode.chained_assignment = None

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[
                    logging.FileHandler("generate_property_paths.log", mode='w'),
                    logging.StreamHandler()
                    ]
)

def parse_args():
    parser = ArgumentParser(description="Run WiDNeR_extended.")
    parser.add_argument('command', choices=['WiDNeR_extended'])
    parser.add_argument('--structured_triples', help='Input the file with structured triples')
    parser.add_argument('--attributive_triples', help='Input the file with attributive triples')

    return parser.parse_args()

def WiDNeR_extended(structured_triples, attributive_triples, input_file_name):
    ## generate relation-relation network
    rel_rel_net = WiDNeR_Light(structured_triples)
    #print(rel_rel_net)

    ## generate relation-attribute network
    structured_triples_ = structured_triples.drop('head', axis=1)
    attributive_triples_ = attributive_triples.drop('value', axis=1)
    rel_attr_net = WiDNeR_attribute(structured_triples_, attributive_triples_)
    #print(rel_attr_net)

    ## put together relation-relation network and  relation-attribute network
    rel_attr_net.rename(columns={'relation': 'relation_x', 'attribute': 'relation_y'}, inplace=True)
    net = pd.concat([rel_rel_net, rel_attr_net], ignore_index=True)
    #print(net)


    filename1 = input_file_name + '_WiDNeR_extended_attr.txt'
    filename2 = input_file_name + '_WiDNeR_extended_net.txt'
    rel_attr_net.to_csv(filename1, sep=' ', index=False, header=False)
    net.to_csv(filename2, sep=' ', index=False, header=False)

    return rel_attr_net, net, filename1, filename2

def WiDNeR_attribute(structured_triples, attributive_triples):

    ##merge important part of structured and attributive triples
    struc_attrib = structured_triples.merge(attributive_triples, left_on='tail', right_on='head', how='inner')
    struc_attrib = struc_attrib.drop('head', axis=1)
    #print(len(struc_attrib), struc_attrib)

    struc_attrib.drop_duplicates(inplace=True)
    #print(len(struc_attrib), struc_attrib)

    rel_attr_net = struc_attrib.groupby(['relation','attribute']).size().to_frame('weight').reset_index()

    return rel_attr_net

def main(structured_triples, attributive_triples):
    structured_triples_ = pd.read_csv(structured_triples, header=None, usecols=[1,2], names=['relation', 'tail'], delimiter='\t| ')
    attributive_triples_ = pd.read_csv(attributive_triples, header=None, usecols=[0,1], names=['head','attribute'], delimiter='\t| ')
    input_file_name = os.path.splitext(structured_triples)[0]
    logging.info(f'reading {len(structured_triples_)} triples has been completed!')
    logging.info(f'reading {len(attributive_triples_)} triples has been completed!')

    WiDNeR_extended(structured_triples_, attributive_triples_, input_file_name)

if __name__ == '__main__':

    args = parse_args()

    if args.command == 'WiDNeR_extended':
        main(structured_triples=args.structured_triples, attributive_triples=args.attributive_triples)
