from pymilvus import MilvusClient, DataType, Function, FunctionType

import sys
import stopwordsiso
import langdetect
import string
import json_stream

import time

DUMPLIMIT = 100000;
RESETDB = 0; #set to >0 if you want to throw away the old DB
COLLECTIONNAME = 'demo_collection';

print('spinning up milvus');
analyzer_params = {
    'tokenizer': 'standard',
    'filter': ['lowercase'],
    };

#generate schema for DB
schema = MilvusClient.create_schema();
schema.add_field(
    field_name = 'id',
    datatype = DataType.INT64,
    is_primary = True,
    auto_id = True,
);
schema.add_field(
    field_name = 'text',
    datatype = DataType.VARCHAR,
    max_length = 1024,
    enable_analyzer = True,
    enable_match = True,
    analyzer_params = analyzer_params,
    );
schema.add_field(
    field_name = 'URI',
    datatype = DataType.VARCHAR,
    max_length = 256,
    );
schema.add_field(
    field_name = 'embeddedfrom',
    datatype = DataType.VARCHAR,
    max_length = 128,
    );
schema.add_field(
    field_name = 'sparse_text',
    datatype = DataType.SPARSE_FLOAT_VECTOR,
    );

bm25_function = Function(
    name = 'text_bm25_emb',
    input_field_names = ['text'],
    output_field_names = ['sparse_text'],
    function_type = FunctionType.BM25,
    );

schema.add_function(bm25_function);

#connect to a database
dbClient = MilvusClient('dbpediaAbstractsNoStopwords.db');

if RESETDB:
    if dbClient.has_collection(collection_name = COLLECTIONNAME):
        dbClient.drop_collection(collection_name = COLLECTIONNAME);
    index_params = dbClient.prepare_index_params();
    index_params.add_index(
        field_name = 'sparse_text',
        index_type = 'SPARSE_INVERTED_INDEX',
        metric_type = 'BM25',
        params = {
            'inverted_index_algo': 'DAAT_MAXSCORE',
            'bm25_k1': 1.2,
            'bm25_b': 0.75,
            }
        );
    dbClient.create_collection(
        collection_name = COLLECTIONNAME,
        schema = schema,
        index_params = index_params,
        );

#cache some stopwords
cachedStopWords = {
    'en': stopwordsiso.stopwords('en'),
    'de': stopwordsiso.stopwords('de'),
    'fr': stopwordsiso.stopwords('fr'),
    };

#input data from external file
testfile = './jsondata/00_EMBEDDED_catsAndAbstracts_from_0_to_1000.json';

testfile = open(testfile, 'rb');
print('loading data from json');
#dataObject = json.load(testfile);
dataObject = [];
#objects = ijson.items(testfile, 'vector');
objects = json_stream.load(testfile);
# print(dir(objects));
print('loop through objects');
stored = 0;
proctotal = 0;
for o in objects:
    vec = list(o['vector']);
    abstract = o['text'];
    uri = o['URI']
    tag = o['embeddedfrom'];

    if tag == 'shortabstract':
        #clean text
        try:
            lang = langdetect.detect(abstract);
        except:
            print('language detection FAIL for');
            print(abstract);
            continue;
        #TODO: this is going to crash if lang is not supported by stopwordsiso
        #solution: stopwordiso returns an empty set for a lnaguage it doesn't recognize
        if lang not in cachedStopWords:
            cachedStopWords[lang] = stopwordsiso.stopwords(lang);
        sw = cachedStopWords[lang];
        if len(sw):
            #get rid of all punctuation
            abstractNoPunctuation = abstract.translate(str.maketrans('', '', string.punctuation));
            #seems to be the faster way to do things?...
            noStopWordsText = '';
            for word in abstractNoPunctuation.split():
                if word not in sw:
                    noStopWordsText += word + ' ';

            dataObject.append({'text': noStopWordsText,
                               'URI': uri,
                               'embeddedfrom': 'shortabstractnostopwords',
                              });
        else:
            print('NO STOPWORDS AVAILABLE FOR LANG = ' + lang + '. SKIPPING');

        stored += 1;
        if stored%1000 == 0:
            print('parsed ' + str(stored) + ' entities');
        if stored == DUMPLIMIT:
            print('intermediate data push');
            res = dbClient.insert(collection_name = COLLECTIONNAME, data = dataObject);
            #reset
            dataObject = [];
            stored = 0;
            # print(res);
    proctotal += 1;

    if proctotal%1000 == 0:
        print('procd total of ' + str(proctotal) + ' records');


print('loading data into milvus');
res = dbClient.insert(collection_name = COLLECTIONNAME, data = dataObject);
print(res);
testfile.close();

#build index
#NOTE: this is not particularly clever, as this will rebuild the index very often...
#TODO: put this somewhere else
index_params = dbClient.prepare_index_params();
index_params.add_index(
    field_name = 'sparse_text',
    index_type = 'SPARSE_INVERTED_INDEX',
    metric_type = 'BM25',
    params = {
        'inverted_index_algo': 'DAAT_MAXSCORE',
        'bm25_k1': 1.2,
        'bm25_b': 0.75,
        }
    );

dbClient.create_index(
    collection_name = COLLECTIONNAME,
    index_params = index_params,
);

#testing
search_params = {
    'params': {'drop_ratio_search': 0.2},
    };

res = dbClient.search(
    collection_name = COLLECTIONNAME,
    data = ['what is the .app TLD?'],
    anns_field = 'sparse_text',
    output_fields = ['text'],
    limit = 3,
    search_params = search_params,
    );

print(res);
