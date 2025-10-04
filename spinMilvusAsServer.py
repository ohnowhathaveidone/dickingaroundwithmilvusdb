from fastapi import FastAPI
import uvicorn
from pymilvus import MilvusClient
import logging
import datetime
import sys
from pydantic import BaseModel

#TEST
from pymilvus import model
from pymilvus.model.dense import SentenceTransformerEmbeddingFunction

if len(sys.argv) < 2:
    sys.exit('usage: python ' + sys.argv[0] + ' milvusDataBaseDump.db');

#request classes. fastAPI wants this type of format
class QueryRequest(BaseModel):
    collection_name:    str
    filter_string:      str
    output_fields:      list

class SearchRequestVector(BaseModel):
    collection_name:    str
    data:               list
    limit:              int
    output_fields:      list

class SearchRequestFullText(BaseModel):
    collection_name:    str
    data:               list
    anns_field:         str
    limit:              int
    output_fields:      list
    search_params:      dict

#TODO:  extract this into a custom module
#       loggers really seem to be better than print :)
#       BUT: #DOLATER
#set up logger
logger = logging.getLogger(sys.argv[0] + __name__);
#we will do a more complex setup for the logger so that it writes to a file _and_ to stdout usig handlers
logger.setLevel(logging.DEBUG);
file_handler = logging.FileHandler('./logs/log_' + datetime.datetime.now().isoformat() + '.log', mode = 'w');
file_handler.setLevel(logging.DEBUG);
file_formatter = logging.Formatter(
    '%(asctime)s - %(levelname)s - %(message)s',
    );
file_handler.setFormatter(file_formatter);

stream_handler = logging.StreamHandler(sys.stdout);
stream_handler.setLevel(logging.DEBUG);
stream_formatter = logging.Formatter(
    '%(asctime)s - %(levelname)s: %(message)s',
    );
stream_handler.setFormatter(stream_formatter);

logger.addHandler(file_handler);
logger.addHandler(stream_handler);

#helper to get timestamps
def ts():
    return datetime.datetime.now().isoformat() + '\t';

#this set of params prevents loading the entire db into RAM. (probably) set to False to load into RAM
#NOTE: NO WORKY WORKY W/ MILVUS LITE!
extraCollectionProperties = {
    'mmap.enabled': True,
    };

#TODO: MAKE THE EMBEDDING MODEL MORE FLEXIBLE
#TEST
embedding_fn = SentenceTransformerEmbeddingFunction(
    model_name = 'LaBSE',
    device = 'cpu',
    );

#VVRRROOOOOMMMMM
logger.info('starting db server');
app = FastAPI();

logger.info('connecting to ' + sys.argv[1]);
dbClient = MilvusClient(sys.argv[1]);

#NOTE: DOESNT FUCKING WORK -> UNIMPLEMENTED
# print(dbClient.describe_database(sys.argv[1]));
logger.info('db loaded');

#print stats about db to stdout
logger.info('collecting db stats');
collections = dbClient.list_collections();
logger.info('available collections: ');
#logger.info('(also overwriting collection properties here :P)');
for c in collections:
    # print(dbClient.get_load_state(c)); #this info you can have on lite.
    #NOTE: THIS DOESNT FUCKING WORK ON LITE -> UNIMPLEMENTED
    # dbClient.alter_collection_properties(
    #     collection_name = c,
    #     properties = extraCollectionProperties,
    #     );
    logger.info(c);

logger.info('collection schemas: ');
for c in collections:
    schema = dbClient.describe_collection(collection_name = c);
    logger.info(f'schema for {c}: {schema}');
    stats = dbClient.get_collection_stats(collection_name = c);
    logger.info(f'stats for {c}: {stats}');

logger.info('performing dummy get to force loading data');
#force DB load
res = dbClient.get(
    collection_name = collections[0],
    ids = [42, 1337],
    );
print(res); #should not contain anything relevant

logger.info('setting up fastAPI endpoints');

#fastAPI setup
@app.get('/collections')
def list_collections():
    return dbClient.list_collections();

@app.post('/query')
def query_collection(req: QueryRequest):
    res = dbClient.query(
        collections_name = req.collection_name,
        filter = req.filter_string,
        output_fields = req.output_fields,
        );
    return res;

@app.post('/searchVector')
def search_vector(req: SearchRequestVector):
    res = dbClient.search(
        collection_name = req.collection_name,
        data = req.data,
        limit = req.limit,
        output_fields = req.output_fields,
        );
    return res;

@app.post('/searchFulltext')
def search_fulltext(req: SearchRequestFullText):
    res = dbClient.search(
        collection_name = req.collection_name,
        data = req.data,
        anns_field = req.anns_field,
        limit = req.limit,
        output_fields = req.output_fields,
        search_params = req.search_params,
        );
    return res;



#RUN THIS WITH:
#TODO: what does --reload do?...
#$ uvicorn spinMilvusAsServer.py:app --host=0.0.0.0 --port=1337
# alternatively, rather run directly through here

if __name__ == '__main__':
    logger.info('starting fastAPI instance');
    uvicorn.run(
        'spinMilvusAsServer:app',
        host = '127.0.0.1', #set to 0.0.0.0 to expose on network
        port = 1337,
        log_level = 'info',
        reload = False,
        );
