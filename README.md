# dickingaroundwithmilvusdb
having fun with milvusDB and DBpedia data :)
this repo holds scripts and information prepared for the 2025 hackathon organized at [q30](https://q30.space/en/) in biel/bienne within the context of siwss AI weeks.  

## NB  
you _will_ most likely need linux or mac to run these things. the database used here is [milvusDB](https://milvus.io/), which currently only supports windows (locally) if you run it through docker, which i believe implies using WSL.  

## prerequsites  
i used python venv for this:  
´´´
cd /wherever/you/cloned/the/GH/repo  
python -m venv yourVenvName  
source yourVenvName/bin/activate  
pip install -r requirements.txt  
´´´

## scripts  
testEmbeddings.py  
- run some tests to see what milvus will throw at you  
spinMilvusAsServer.py  
- load a database into milvus  
- start a FastAPI server to accept seach queries as http requests  

## data  
you can get more serious databases from  
[zenodo](https://doi.org/10.5281/zenodo.17266309)  

