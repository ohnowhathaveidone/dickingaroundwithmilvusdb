from pymilvus import MilvusClient, DataType
from pymilvus import model
from pymilvus.model.dense import SentenceTransformerEmbeddingFunction

import sys

import json_stream

import time

DUMPLIMIT = 10000;
RESETDB = 0;
COLLECTIONNAME = 'demo_collection';

print('spinning up milvus');
#generate schema for DB
schema = MilvusClient.create_schema();
schema.add_field(
    field_name = 'id',
    datatype = DataType.INT64,
    is_primary = True,
    auto_id = True,
);
schema.add_field(
    field_name = 'vector',
    datatype = DataType.FLOAT_VECTOR,
    dim = 768,
);
schema.add_field(
    field_name = 'text',
    datatype = DataType.VARCHAR,
    max_length = 1024,
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

dbClient = MilvusClient('mivlusDemo.db');

if RESETDB:
    if dbClient.has_collection(collection_name = COLLECTIONNAME):
        dbClient.drop_collection(collection_name = COLLECTIONNAME);
    dbClient.create_collection(
        collection_name = COLLECTIONNAME,
        dimension = 768,
        schema = schema,
        );

embedding_fn = SentenceTransformerEmbeddingFunction(
    model_name = 'LaBSE',
    device = 'cpu',
    );

#input data from external file
testfile = './jsondata/EMBEDDED_catsAndAbstracts_from_0_to_1000000.json';
# testfile = './jsondata/EMBEDDED_catsAndAbstracts_from_1000000_to_2000000.json';
# testfile = './jsondata/00_EMBEDDED_catsAndAbstracts_from_0_to_1000.json';
testfile = './jsondata/EMBEDDED_catsAndAbstracts_from_2000000_to_3000000.json';
testfile = './jsondata/EMBEDDED_catsAndAbstracts_from_3000000_to_4000000.json';
testfile = './jsondata/EMBEDDED_catsAndAbstracts_from_4000000_to_5000000.json';
testfile = './jsondata/EMBEDDED_catsAndAbstracts_from_5000000_to_6000000.json';
testfile = './jsondata/EMBEDDED_catsAndAbstracts_from_6000000_to_7000000.json';
testfile = './jsondata/EMBEDDED_catsAndAbstracts_from_7000000_to_8000000.json';
testfile = './jsondata/EMBEDDED_catsAndAbstracts_from_8000000_to_9000000.json';

testfile = open(testfile, 'rb');
print('loading data from json');
#dataObject = json.load(testfile);
dataObject = [];
#objects = ijson.items(testfile, 'vector');
objects = json_stream.load(testfile);
print(dir(objects));
print('loop through objects');
stored = 0;
proctotal = 0;
for o in objects:
    vec = list(o['vector']);
    text = o['text'];
    uri = o['URI']
    tag = o['embeddedfrom'];

    if tag == 'shortabstract':
        dataObject.append({'vector': vec,
                           'URI': uri,
                           'embeddedfrom': tag,
                           'text': text,
                            });

        stored += 1;
        if stored%1000 == 0:
            print('parsed ' + str(stored) + ' entities');
        if stored == DUMPLIMIT:
            print('intermediate data push');
            res = dbClient.insert(collection_name = COLLECTIONNAME, data = dataObject);
            #reset
            dataObject = [];
            stored = 0;
            print(res);
    proctotal += 1;

    if proctotal%1000 == 0:
        print('procd total of ' + str(proctotal) + ' records');

print(dataObject);
#sys.exit();
testfile.close();

print('loading data into milvus');
res = dbClient.insert(collection_name = COLLECTIONNAME, data = dataObject);
print(res);
#free memory
del(dataObject);

print('indexing');
#upon loading data from files, we need to index vector fields.
indexParams = MilvusClient.prepare_index_params();
indexParams.add_index(
    field_name = 'vector',
    index_type = 'IVF_FLAT',
    index_name = 'vector_index',
    metric = 'COSINE',
    params = {'nlist': 128},
);

dbClient.create_index(
    collection_name = COLLECTIONNAME,
    index_params = indexParams,
);

print('performing a search');
#try a asimple search?..
strtQuery = time.time();
res = dbClient.search(
    collection_name = COLLECTIONNAME,
    data = embedding_fn.encode_queries(['Mitteldichte Faserplatte',
                                        'a movie',
                                        'Canada',
                                        'A programming language for high performance applications',
                                        '.app is a gTLD (generic top-level domain) in ICANN’s New gTLD Program. Google purchased the gTLD in an ICANN Auction of Last Resort in February 2015. The TLD is of interest due to its utility in regards to branding mobile, web, and other applications.',
                                        '.app ist eine gTLD (generic top level domain) in ICANNs neuem gTLD Programm. Google kaufte diese gTLD im Rahmen einer ICANN Auction of last resort im Februar 2015. Diese TLD ist interessant, da sie für das Branding von mobilen, Web- und andere Applikationen genutzt werden kann.']),
    limit = 3,
    # output_fields = ['text', 'embeddedfrom', 'URI'],
    output_fields = ['URI'],
    );
stpQuery = time.time();

print(res);

print('query took ' + str(stpQuery - strtQuery) + ' s');
