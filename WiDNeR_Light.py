##use this code to generate relation-relation network
from argparse import ArgumentParser
import numpy as np
import pandas as pd
import random, math
import networkx as nx
import logging, os
from dateutil.parser import parse

pd.options.mode.chained_assignment = None

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[
                    logging.FileHandler("generate_property_paths.log", mode='w'),
                    logging.StreamHandler()
                    ]
)

def WiDNeR_Light(structured_triples_):

    ## different relations - direct
    result_direct = structured_triples_.merge(structured_triples_, left_on='tail', right_on='head', how='inner')
    #print(result_direct)
    diff_rels_direct = result_direct.loc[result_direct['relation_x'] != result_direct['relation_y']]
    #print(len(diff_rels_direct))
    diff_rels_direct.drop_duplicates(inplace=True)
    #print(len(diff_rels_direct))
    #print(diff_rels_direct)
    diff_rels_direct = diff_rels_direct.groupby(['relation_x', 'relation_y']).size().to_frame('weight').reset_index()
    #print(diff_rels_direct)

    ## same relations - direct
    same_rels_direct = result_direct.loc[(result_direct['relation_x'] == result_direct['relation_y']) & ((result_direct['head_x'] != result_direct['tail_x']) | (result_direct['head_x'] != result_direct['tail_y'])) ]
    #print(len(same_rels_direct))
    same_rels_direct.drop_duplicates(inplace=True)
    #print(len(same_rels_direct))
    #print(same_rels_direct)
    same_rels_direct = same_rels_direct.groupby(['relation_x', 'relation_y']).size().to_frame('weight').reset_index()
    #print(same_rels_direct)

    ## combine different and same relations
    rel_rel_net = pd.concat([diff_rels_direct, same_rels_direct], ignore_index=True)
    rel_rel_net = rel_rel_net.sort_values(by=['weight'], ascending=False, ignore_index=True)
    #print(rel_rel_net)

    return rel_rel_net

def read_triples(structured_triples):
    structured_triples_ = pd.read_csv(structured_triples, header=None, names=['head','relation', 'tail'], delimiter='\t| ')
    logging.info(f'reading {len(structured_triples_)} triples has been completed!')

    return structured_triples_

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('command', choices=['WiDNeR_Light'])
    parser.add_argument('--structured_triples', help='Input the file with structured triples')

    args = parser.parse_args()

    if args.command == 'WiDNeR_Light':
        structured_triples_ = read_triples(structured_triples=args.structured_triples)
        rel_rel_net = WiDNeR_Light(structured_triples_)
        filename = os.path.splitext(args.structured_triples)[0]
        rel_rel_net.to_csv(f'{filename}_WiDNeR_Light.txt', sep='\t')
