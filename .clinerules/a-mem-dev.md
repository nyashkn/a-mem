# FalkorDB-py Guidelines

## Introduction

FalkorDB is a graph database built for AI applications. The Python client library (`falkordb-py`) provides a way to interact with FalkorDB from Python applications. This document provides guidelines for working with FalkorDB-py based on lessons learned during implementation.

## Key Concepts

- **Graph**: The main entry point for interacting with FalkorDB. Represents a graph database.
- **Node**: Represents a node in the graph with properties and labels.
- **Edge**: Represents a relationship between nodes with a type and properties.
- **Query**: Cypher queries are used to interact with the graph.
- **Node and Edge Properties**: Attributes stored on nodes and edges.

## Working with FalkorDB Nodes

### Accessing Node Properties

One of the most important patterns to understand is how to access node properties:

```python
# INCORRECT: This will fail with "'Node' object has no attribute 'get'"
node_id = node.get('id', '')

# CORRECT: Access properties through the .properties attribute
node_id = node.properties.get('id', '')
```

Nodes in FalkorDB are Python objects with a `.properties` attribute that contains a dictionary of all properties. Always access properties through this attribute.

### Safety Checks

Always check if a Node object has the required attributes before accessing them:

```python
if hasattr(node, 'properties'):
    # Now safe to access node.properties
    value = node.properties.get('some_property', default_value)
else:
    # Handle the case where node doesn't have properties
    logger.warning(f"Node doesn't have properties attribute: {type(node)}")
```

## Vector Operations

### Creating Vector Indices

FalkorDB supports vector indices for similarity searches:

```python
# Create a vector index on a node property
graph.create_node_vector_index("Label", "property_name", dim=vector_size, similarity_function="cosine")
```

The `similarity_function` can be "cosine", "euclidean", or other supported functions.

### Working with Vector Embeddings

When storing vector embeddings:

1. For large vectors, consider storing them as JSON strings:
   ```python
   import json
   embedding_json = json.dumps(embedding)
   query = f"MATCH (n) SET n.embedding = '{embedding_json}'"
   ```

2. When retrieving vectors, parse them back:
   ```python
   embedding_str = node.properties.get('embedding', '[]')
   embedding = json.loads(embedding_str) if isinstance(embedding_str, str) else []
   ```

## Query Results

### Working with Query Results

Query results in FalkorDB-py contain:

- `result_set`: A list of records (rows) returned by the query
- `nodes_created`, `properties_set`, etc.: Metadata about the query execution

```python
result = graph.query("MATCH (n:Label) RETURN n LIMIT 10")
for record in result.result_set:
    node = record[0]  # First column contains the node
    # Work with node.properties
```

### Processing Result Sets

When working with result sets:

1. Always check if the result_set exists and has items:
   ```python
   if result.result_set and len(result.result_set) > 0:
       # Process results
   ```

2. For each record, access columns by index:
   ```python
   for record in result.result_set:
       # First column
       first_col = record[0]
       # Second column (if it exists)
       if len(record) > 1:
           second_col = record[1]
   ```

## Handling Complex Data Types

### Serialization

When storing complex data types (lists, dicts):

1. Convert to JSON strings before storage:
   ```python
   metadata["keywords"] = json.dumps(metadata["keywords"])
   ```

2. Parse back when retrieving:
   ```python
   if isinstance(value, str) and (value.startswith('[') or value.startswith('{')):
       try:
           metadata[key] = json.loads(value)
       except:
           metadata[key] = value
   ```

## Error Handling

Always implement robust error handling when working with FalkorDB:

```python
try:
    # FalkorDB operations
    result = graph.query("CREATE (:Label {prop: $value})", params={"value": 123})
except Exception as e:
    logger.error(f"Error executing FalkorDB operation: {e}")
    # Handle error appropriately
```

## Performance Considerations

1. **Indexing**: Create appropriate indices for properties you frequently query:
   ```python
   graph.create_node_range_index("Label", "property_name")
   ```

2. **Batch Operations**: For bulk operations, consider batching:
   ```python
   # Batch create 100 nodes
   params = {"props": [{"id": i} for i in range(100)]}
   query = "UNWIND $props AS prop CREATE (:Node {id: prop.id})"
   graph.query(query, params=params)
   ```

3. **Check for Existing Indices**: Before creating indices, check if they already exist:
   ```python
   indices = graph.list_indices().result_set
   has_index = any('property_name' in str(idx) for idx in indices)
   ```

## Testing

When testing FalkorDB integrations:

1. Use a dedicated test graph that can be created and deleted during tests
2. Clean up after tests by deleting test graphs
3. Mock FalkorDB interactions when appropriate for unit tests

## Common Issues and Solutions

1. **Issue**: "'Node' object has no attribute 'get'"
   **Solution**: Use `node.properties.get('key')` instead of `node.get('key')`

2. **Issue**: "Vector similarity search not supported"
   **Solution**: Check if your FalkorDB version supports vector operations; fall back to manual similarity calculations if needed

3. **Issue**: "Encountered unhandled type in inlined properties"
   **Solution**: Pre-process complex data types (like arrays, dicts) by converting them to JSON strings
