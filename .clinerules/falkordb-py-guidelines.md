## Brief overview
These guidelines focus on working with the FalkorDB-py library in Python applications. They address key patterns, common pitfalls, and best practices for interacting with the FalkorDB graph database.

## Data access patterns
- Always access Node properties through the `.properties` attribute, not directly: `node.properties.get('id', '')` instead of `node.get('id', '')`
- Include safety checks before accessing properties: `if hasattr(node, 'properties')`
- Access record columns by index in query results: `record[0]` for the first column

## Data serialization
- Store complex data types (lists, dictionaries) as JSON strings
- Use `json.dumps()` before storage and `json.loads()` when retrieving
- Handle serialization errors with appropriate try/except blocks
- When retrieving potential JSON strings, check if the value starts with '[' or '{' before parsing

## Vector operations
- Create appropriate vector indices for properties used in similarity searches
- Use proper dimension sizes when creating vector indices
- Fall back to manual similarity calculations if vector search functions are not supported
- Store large vector embeddings as JSON strings to avoid size limitations

## Error handling
- Implement robust error handling for all FalkorDB operations
- Check for existence of indices before creating new ones
- Handle the case where a Node doesn't have a properties attribute
- Provide useful error messages that include context about the operation

## Performance considerations
- Create indices for frequently queried properties
- Use batch operations for bulk data insertion or updates
- Check for existing indices before creating new ones
- Use dedicated test graphs that can be created and deleted during testing
