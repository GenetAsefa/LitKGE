##use this code to generate numerical features
from argparse import ArgumentParser
import numpy as np
import pandas as pd
import random, math
import networkx as nx
import logging, os, time, csv
from dateutil.parser import parse
from datetime import datetime, timezone
import calendar, time;
pd.options.mode.chained_assignment = None
from WiDNeR_extended import WiDNeR_extended
from node2vec_master.src import node2vec as nv
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[
                    logging.FileHandler("generate_property_paths.log", mode='w'),
                    logging.StreamHandler()
                    ]
)

def parse_args():
    '''
	Parses the arguments.
	'''

    parser = ArgumentParser(description="Run generate_numerical_features.")

    parser.add_argument('command', choices=['generate_numerical_features'])

    parser.add_argument('--structured_triples', help='Input the file with structured triples')

    parser.add_argument('--attributive_triples', help='Input the file with attributive triples')

    parser.add_argument('--ent2id', help='Input the file with ent2id')

    parser.add_argument('--rel2id', help='Input the file with rel2id')

    parser.add_argument('--attr2id', help='Input the file with attr2id')

    parser.add_argument('--walk-length', type=int, default=3, help='Length of walk per source. Default is 3.')

    parser.add_argument('--num-walks', type=int, default=10, help='Number of walks per source. Default is 10.')

    parser.add_argument('--p', type=float, default=1, help='Return hyperparameter. Default is 1.')

    parser.add_argument('--q', type=float, default=1, help='Inout hyperparameter. Default is 1.')

    parser.add_argument('--weighted', dest='weighted', action='store_true', help='Boolean specifying (un)weighted. Default is unweighted.')

    parser.add_argument('--unweighted', dest='unweighted', action='store_false')

    parser.set_defaults(weighted=False)

    parser.add_argument('--directed', dest='directed', action='store_true', help='Graph is (un)directed. Default is undirected.')

    parser.add_argument('--undirected', dest='undirected', action='store_false')

    parser.set_defaults(directed=False)

    parser.add_argument('--random-state', type=int, default=0000, help='Random seed for the random walk algorithm. Default is 0000.')

    parser.add_argument('--normalize_date', dest='normalize_date', action='store_true', help='Boolean specifying date should be normalized or not. Default is not to normalize.')

    #parser.add_argument('--not_to_normalize_date', dest='not_to_normalize_date', action='store_false')

    parser.set_defaults(normalize_date=False)

    return parser.parse_args()

def generate_numerical_features(structured_triples, attributive_triples, ent2id, rel2id, attr2id, normalize_date):
    ## read Triples
    structured_triples_ = pd.read_csv(structured_triples, header=None, names=['head','relation', 'tail'], delimiter='\t| ')
    attributive_triples_ = pd.read_csv(attributive_triples, header=None, names=['head','attribute', 'value'], delimiter='\t| ')
    #print(structured_triples_)
    #print(attributive_triples_)

    ## normaliz date values
    if normalize_date:
        attributive_triples_ = normalize_date_values(attributive_triples_)
        print('normalized')

    ## generate a network using WiDNeR_extended
    input_file_name = os.path.splitext(structured_triples)[0]
    _,_,_, filename = WiDNeR_extended(structured_triples_, attributive_triples_, input_file_name)

    print(filename)

    ## generate random walk using node2vec's algorithm
    nx_G = read_graph(filename)
    G = nv.Graph(nx_G, args.directed, args.p, args.q)
    G.preprocess_transition_probs()
    print(args.num_walks, args.walk_length, args.random_state)
    cutpt = len(pd.read_csv(rel2id, header=None)) ## the number of relations to give as the smallest id of attributes for LW48k - 257, Fb15k-237 ..
    print(cutpt)
    walks = G.simulate_walks(args.num_walks, args.walk_length, args.random_state, cutpt)
    print(len(walks), ' walks in total')
    walks_2hop = []
    walks_3hop = []
    for w in walks:
        if len(w) == 2:
            walks_2hop.append(w)
        elif len(w) == 3:
            walks_3hop.append(w)
    print(len(walks_2hop), '2 hop walks in total')
    print(len(walks_3hop), '3 hop walks in total')

    ## usig the walks as templates get all the entries for each feature/walk
    features_h2 = get_feature_values(2, walks_2hop, structured_triples_, attributive_triples_)
    features_h3 = get_feature_values(3, walks_3hop, structured_triples_, attributive_triples_)

    ## combine features from different hopes, features_h1, features_h2, and features_h2
    features_h1 = attributive_triples_.rename(columns={'attribute': 'property_path'}, inplace=False)
    features = pd.concat([features_h1, features_h2, features_h3])

    features['value'] = pd.to_numeric(features['value'], downcast="float", errors='coerce')
    #print(features[features.isna().any(axis=1)])
    features=features.dropna().reset_index(drop=True)
    print(features[features.isna().any(axis=1)])
    print('features', features)
    print(features.nunique())


    ## take the average of the values of multivalued attributes
    features_averaged = features.groupby(['head','property_path']).agg(value_mean = ('value', 'mean')).reset_index()
    print(features_averaged)
    print(features_averaged.nunique())

    ##convert entity ids to names
    ent2id_= pd.read_csv(ent2id, header=None, names=['head','id'], delimiter='\t| ')
    features_averaged = ent_id_to_name(features_averaged, ent2id_)

    ##convert rel and attr ids with names
    rel2id_ = pd.read_csv(rel2id, header=None, names=['property_path','id'], delimiter='\t| ')
    attr2id_ = pd.read_csv(attr2id, header=None, names=['property_path','id'], delimiter='\t| ')
    features_averaged = prop_id_to_name(features_averaged, rel2id_, attr2id_)
    return features_averaged

def get_feature_values(num_hops, walks, structured_triples_, attributive_triples_):
    if num_hops ==2:
        rels = [item[0] for item in walks]
        #print(rels)
        props = [item[1] for item in walks]
        str_trip = structured_triples_.loc[structured_triples_['relation'].isin(rels)]
        attr_trip = attributive_triples_.loc[attributive_triples_['attribute'].isin(props)]
        result_h2 = str_trip.merge(attr_trip, left_on='tail', right_on='head', how='inner')
        result_h2['property_path'] = result_h2[['relation', 'attribute']].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)
        result_h2.drop(columns=['relation', 'attribute'], inplace=True)
        result_h2 = result_h2.reindex(columns=['head_x', 'property_path', 'value'])
        result_h2.rename(columns={'head_x': 'head'}, inplace=True)
        return result_h2

    if num_hops ==3:
        rels1 = [item[0] for item in walks]
        rels2 = [item[1] for item in walks]
        props = [item[2] for item in walks]
        str_trip1 = structured_triples_.loc[structured_triples_['relation'].isin(rels1)]
        str_trip2 = structured_triples_.loc[structured_triples_['relation'].isin(rels2)]
        attr_trip = attributive_triples_.loc[attributive_triples_['attribute'].isin(props)]
        result = str_trip1.merge(str_trip2, left_on='tail', right_on='head', how='inner')
        result_h3 = result.merge(attr_trip, left_on='tail_y', right_on='head', how='inner')
        result_h3['property_path'] = result_h3[['relation_x', 'relation_y', 'attribute']].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)
        result_h3.drop(columns=['relation_x', 'relation_y', 'attribute'], inplace=True)
        result_h3 = result_h3.reindex(columns=['head_x', 'property_path', 'value'])
        result_h3.rename(columns={'head_x': 'head'}, inplace=True)

        return result_h3

def read_graph(filename):
	'''
	Reads the input network in networkx.
	'''
	if args.weighted:
	    G = nx.read_edgelist(filename, nodetype=int, data=(('weight',float),), create_using=nx.DiGraph())

	else:
		G = nx.read_edgelist(filename, nodetype=int, create_using=nx.DiGraph())
		for edge in G.edges():
			G[edge[0]][edge[1]]['weight'] = 1

	if not args.directed:
		G = G.to_undirected()

	return G

def normalize_date_values(attributive_triples_):
    ## normaliz date values
    attributive_triples_date= attributive_triples_[attributive_triples_['value'].astype(str).str.contains("XMLSchema#dateTime")]

    attributive_triples_date['value']= attributive_triples_date['value'].astype(str).str.replace(r'\^\^.*', '', regex=True)
    attributive_triples_date['value']=attributive_triples_date['value'].astype(str)

    attributive_triples_date['value'] = attributive_triples_date['value'].map(lambda x: x.replace('"', ''))
    print(attributive_triples_date)
    attributive_triples_date["value"] = attributive_triples_date["value"].map(lambda x: (np.datetime64(x) - np.datetime64('1970-01-01T00:00:00Z'))/np.timedelta64(1, 's'))
    print(attributive_triples_date)
    ## other values different from date (i.e., time and geo)
    attributive_triples_= attributive_triples_[~attributive_triples_['value'].astype(str).str.contains("XMLSchema#dateTime")]
    attributive_triples_['value'] = attributive_triples_['value'].astype(str).str.replace(r'\^\^.*', '', regex=True)
    attributive_triples_['value'] = attributive_triples_['value'].astype(str).str.replace('+', '')
    attributive_triples_['value'] = attributive_triples_['value'].astype(str).str.replace('"', '')

    ## combine date and other values
    attributive_triples_ = pd.concat([attributive_triples_, attributive_triples_date])
    attributive_triples_ = attributive_triples_.astype({"value": float})

    return attributive_triples_

def ent_id_to_name(features_averaged, ent2id):
    features_averaged['head'] = features_averaged['head'].map(ent2id.set_index('id')['head']).fillna(features_averaged['head'])

    return features_averaged

def prop_id_to_name(features_averaged, relids, attrids):
    ids = pd.concat([relids, attrids], ignore_index=True)

    ids_pps = ids
    all_pps = features_averaged[features_averaged['property_path'].astype(str).str.contains('_')]['property_path'].tolist()
    all_pps = list(set(all_pps))
    #print(len(all_pps), len(set(all_pps)))
    for path in all_pps:
        path_ids = path.split('_')
        #print(path_ids)
        path_name = ''
        for id in path_ids:
            #print('id', ids['id'])
            name = ids.loc[ids['id']==int(id)]['property_path'].values[0]
            #print('name', name)
            path_name = path_name + '_' + name

        path_name = path_name.lstrip('_')
        #print('path_name', path_name)

        new_row = {'property_path':path_name, 'id':path}
        ids_pps = ids_pps.append(new_row, ignore_index=True)

    features_averaged['property_path'] = features_averaged['property_path'].map(ids_pps.set_index('id')['property_path']).fillna(features_averaged['property_path'])


    return features_averaged


if __name__ == '__main__':
    args = parse_args()

    if args.command == 'generate_numerical_features':
        numerical_fatures = generate_numerical_features(structured_triples=args.structured_triples, attributive_triples=args.attributive_triples, \
        ent2id=args.ent2id, rel2id=args.rel2id, attr2id=args.attr2id, normalize_date=args.normalize_date)
        filename = os.path.splitext(args.structured_triples)[0]
        numerical_fatures.to_csv(f'{filename}_numerical_features.txt', sep='\t', index=False, header=False)
