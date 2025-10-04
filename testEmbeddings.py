from sentence_transformers import SentenceTransformer, util
import sys

# Load a multilingual model
# uncomment if you want to try different models
# the results don't vary a lot at first glance.
model = SentenceTransformer('LaBSE');

phrases = [
    "Ein bequemer Stuhl mit hoher Rückenlehne aus massivem Eichenholz.", #product description in German
    "A comfortable chair with a high backrest, often used in offices and dining rooms. It is typically made of wood.", # a description in English
    "Une chaise confortable en chêne massif avec un dossier haut.", #description in french
    "Une chaise confortable avec un dossier haut, fabriquée en chêne massif.", #... avec une structure plus similaire
    "Heute war das Wetter immer noch recht heiss. Ab Mittwoch soll es dann kühler werden.", #and an unrelated sentence in german
    "Trotz des heissen Wetters war es heute angenehm, auf einem bequemen Stuhl mit hoher Rückenlehne, aus massiver Eiche zu sitzen.", #a somewhat related sentence
    ];


embeddings = [];
for p in phrases:
    embeddings.append(model.encode(p, convert_to_tensor = False));
    print(str(len(embeddings[-1])));

print(embeddings[0]);

for i in range(1, len(phrases)):
    cos_score = util.cos_sim(embeddings[0], embeddings[i]);
    print('similarity between');
    print(phrases[0]);
    print('and');
    print(phrases[i]);
    print('is ' + str(cos_score.item()));
