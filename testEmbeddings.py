from sentence_transformers import SentenceTransformer, util
import sys

# Load a multilingual model
# uncomment if you want to try different models
# the results don't vary a lot at first glance.
# model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2');
model = SentenceTransformer('LaBSE');
#model = SentenceTransformer('distiluse-base-multilingual-cased-v1');

#laternative to load: LaBSE

phrases = [
    "Cadenberge", #test
    "Ein bequemer Stuhl mit hoher Rückenlehne aus massivem Eichenholz.", # Your product description in German
    "A comfortable chair with a high backrest, often used in offices and dining rooms. It is typically made of wood.", # The DBpedia description in English
    "Une chaise confortable en chêne massif avec un dossier haut.", #description in french (self-translated)
    "Une chaise confortable avec un dossier haut, fabriquée en chêne massif.", #... avec une structure plus similaire
    "Heute war das Wetter immer noch recht heiss. Ab Mittwoch soll es dann kühler werden.", #and an unrelated sentence in german
    "Trotz des heissen Wetters war es heute angenehm, auf einem bequemen Stuhl mit hoher Rückenlehne, aus massiver Eiche zu sitzen.", #a somewhat related sentence
    ];


embeddings = [];
for p in phrases:
    embeddings.append(model.encode(p, convert_to_tensor = False));
    print(str(len(embeddings[-1])));

print(embeddings[0]);
sys.exit();

for i in range(1, len(phrases)):
    cos_score = util.cos_sim(embeddings[0], embeddings[i]);
    print('similarity between');
    print(phrases[0]);
    print('and');
    print(phrases[i]);
    print('is ' + str(cos_score.item()));

'''
[1]preliminary result from short sentences:
the threshold for _very_ similar sentences that describe similar descriptions can be put around 0.8 - 0.85
0.7 - 0.75 will give you some 50% content match
~0.2 cosine distance is for rather unrelated sentences
given how the cosine distance works, most of this seems rather reasonable.

timing for running on this pc
real	0m7.860s
user	0m6.278s
sys	    0m0.626s

so about 7/5 sec per sentence.

this was driven with paraphrase-multilingual-mpnet-base-v2

[2] tests with LaBSE
seems to be more sensitive to additional info, high correlation for very similar sentences
timing:
real	0m8.441s
user	0m6.516s
sys	    0m0.560s


summary?
seem to perform similarly. latter model seems to be more sensitive to detecting diffs between different length sentences
timeing should be tested more. at first glance, seems to be the same.
'''

'''
# Encode both texts into embeddings
embedding_product = model.encode(product_description_de, convert_to_tensor=True);
embedding_dbpedia = model.encode(dbpedia_description_en, convert_to_tensor=True);

# Compute the cosine similarity between the embeddings
cosine_score = util.cos_sim(embedding_product, embedding_dbpedia);

print(f"Similarity score: {cosine_score.item()}")
'''
