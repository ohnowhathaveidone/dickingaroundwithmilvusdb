from pymilvus import MilvusClient, DataType

# Connect to Milvus (adjust URI as needed)
client = MilvusClient('tmp.db');

# Create schema with text match enabled on 'text' field
schema = client.create_schema(enable_dynamic_field=False)
schema.add_field(
    field_name="id",
    datatype=DataType.INT64,
    is_primary=True,
    auto_id=True
)
schema.add_field(
    field_name='text',
    datatype=DataType.VARCHAR,
    max_length=1000,
    enable_analyzer=True,
    enable_match=True
)
schema.add_field(
    field_name = 'vector',
    datatype = DataType.FLOAT_VECTOR,
    dim = 3,
);

# Create the collection
collection_name = "bla";
if client.has_collection(collection_name=collection_name):
    client.drop_collection(collection_name=collection_name)
client.create_collection(collection_name=collection_name, schema=schema)

# Insert sample data
data = [
    {"text": "machine learning is powerful", 'vector': [1, 2, 3]},
    {"text": "deep learning for robotics", 'vector': [1, 2, 3]},
    {"text": "robotics and automation", 'vector': [1, 2, 3]},
]
client.insert(collection_name=collection_name, data=data)

# Query using TEXT_MATCH to find documents containing 'machine'
filter = "TEXT_MATCH(text, 'machine')";
filter = "TEXT_MATCH (text, \'machine\')";
# filter = "not TEXT_MATCH(text, 'deep') and TEXT_MATCH(text, 'machine') and TEXT_MATCH(text, 'learning')"
# filter = 'id><0';
results = client.query(
    collection_name=collection_name,
    filter=filter,
    output_fields=["id", "text"]
)
print(results)
