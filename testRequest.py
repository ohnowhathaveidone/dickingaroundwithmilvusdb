import requests
from pymilvus.model.dense import SentenceTransformerEmbeddingFunction

import sys

#define target instance -> parametrize accordingly in spinMilvusAsServer.py
ADDR = 'http://127.0.0.1';
PORT = 1337;

#some idiot checks
validArgs = ['vector', 'fulltext'];

def checkCliArgs():
    if len(sys.argv) < 2:
        return 1;
    elif sys.argv[1] not in validArgs:
        return 1;

    return 0;

if checkCliArgs():
    sys.exit('usage: python ' + sys.argv[0] + ' vector | fulltext');

#sample search phrases -> have fun with those.
searchPhrases = [
    'a tool commonly used to drive nails into wood',
    'a tool used to drive screws into wood',
    'a tool used in forging metal',
    'a high performance programming language for schientific and mathematical problems',
    ];

searchPhrases = [
    'Holzbauschrauben SPAX - für das Verschrauben ohne vorzubohren hohe Tragfähigkeit auch bei hohen Zugkräften durch Tellerkopf',
    ];

searchPhrases = [
    'A circular saw is a power-saw using a toothed or abrasive disc or blade to cut different materials using a rotary motion spinning around an arbor. A hole saw and ring saw also use a rotary motion but are different from a circular saw. Circular saws may also be loosely used for the blade itself. Circular saws were invented in the late 18th century and were in common use in sawmills in the United States by the middle of the 19th century.',
    'Circular saw',
    ];

#check for what we actually want to do.
#keep in mind that a lot of the parameters are hardcoded
#and that this can be done much better.
if sys.argv[1] == 'vector':
    #TODO: MAKE BETTER LINK TO WHAT IS ACTUALLY USED IN THE DB!
    embedding_fn = SentenceTransformerEmbeddingFunction(
        model_name = 'LaBSE',
        device = 'cpu',
        );

    searchEmbeddings = embedding_fn.encode_queries(searchPhrases);
    searchEmbeddings = [a.tolist() for a in searchEmbeddings];

    ENDPOINT = '/searchVector';
    headers = {
        'content-type': 'application/json',
        };

    payload = {
        'collection_name': 'demo_collection',
        'data': searchEmbeddings,
        'limit': 10,
        'output_fields': ['URI'],
        };
elif sys.argv[1] == 'fulltext':
    ENDPOINT = '/searchFulltext';
    headers = {
        'content-type': 'application/json',
        };

    payload = {
        'collection_name': 'demo_collection',
        'data': searchPhrases,
        'limit': 10,
        'anns_field': 'sparse_text',
        'output_fields': ['URI'],
        'search_params': {'drop_ratio_search': 0.2},
        };
else:
    sys.exit('sth got fucked');

print('results for search type ' + sys.argv[1] + ':');

res = requests.post(ADDR + ':' + str(PORT) + ENDPOINT, headers = headers, json = payload);

res = res.json();
for i in range(len(searchPhrases)):
    print(searchPhrases[i]);
    for r in res[i]:
        print(r);
    print('');
