# RAG MCP Server Usage Examples

This document provides examples of how to use the RAG MCP Server tools and resources.

## Tool: rag_search

### Basic Search
Search for documents containing specific terms:

```python
# Find documents about "machine learning"
result = rag_search("machine learning")
```

### Limiting Results
Control the number of results returned:

```python
# Get top 5 most relevant documents
result = rag_search("artificial intelligence", number_docs=5)
```

### File Pattern Filtering
Filter results by file path patterns using glob syntax:

```python
# Search only in PDF files
result = rag_search("quarterly report", glob_pattern="*.pdf")

# Search only in email directories
result = rag_search("meeting notes", glob_pattern="*/emails/*")

# Search for files with "report" in the name
result = rag_search("financial data", glob_pattern="*report*")

# Search in specific subdirectories
result = rag_search("project update", glob_pattern="*/2024/*")
```

### Combined Examples
Use multiple parameters together:

```python
# Find top 3 PDF documents about "budget analysis"
result = rag_search(
    query="budget analysis",
    number_docs=3,
    glob_pattern="*.pdf"
)

# Search for recent documents with specific content
result = rag_search(
    query="team performance",
    number_docs=10,
    glob_pattern="*/2024/*"
)
```

## Understanding Results

The `rag_search` tool returns a structured response with the following information:

- **results**: List of matching document chunks
  - **file_path**: Full path to the source file
  - **document**: Content of the document chunk
  - **score**: Similarity score (0-1, higher = more relevant)
  - **chunk_index**: Position of this chunk in the original document
  - **is_deleted**: Whether the source file has been deleted
  - **deletion_timestamp**: When the file was deleted (if applicable)
- **query**: The original search query
- **total_results**: Number of results returned
- **filtered_by_pattern**: Glob pattern used for filtering (if any)

## Resources

### rag-config://server
Get current server configuration:
```
Shows Qdrant settings, embedding model, monitored directories, etc.
```

### rag-stats://database
Get database statistics:
```
Shows collection info, total vectors, indexed files count, etc.
```

## Glob Pattern Examples

| Pattern | Matches |
|---------|---------|
| `*.pdf` | All PDF files |
| `*.{pdf,docx}` | PDF and DOCX files |
| `*/emails/*` | Files in any "emails" directory |
| `*report*` | Files with "report" in the name |
| `*/2024/*` | Files in any "2024" directory |
| `Downloads/*.pdf` | PDF files in Downloads directory |
| `**/*meeting*` | Files with "meeting" in name (any depth) |

## Tips for Better Search Results

1. **Use specific terms**: More specific queries return more relevant results
2. **Combine filters**: Use glob patterns to narrow down to relevant file types
3. **Adjust result count**: Start with default (10), increase if needed
4. **Check scores**: Higher scores (closer to 1.0) indicate better matches
5. **Use semantic search**: The system understands meaning, not just keywords

## Common Use Cases

### Research and Analysis
```python
# Find research papers about a topic
rag_search("deep learning algorithms", glob_pattern="*/research/*.pdf")

# Search meeting notes for project discussions
rag_search("project timeline", glob_pattern="*/meetings/*")
```

### Document Management
```python
# Find all documents related to a client
rag_search("Acme Corporation", number_docs=20)

# Search for financial documents
rag_search("budget forecast", glob_pattern="*financial*")
```

### Email and Communication
```python
# Find emails about specific topics
rag_search("quarterly review", glob_pattern="*/emails/*")

# Search for correspondence with specific people
rag_search("John Smith discussion", glob_pattern="*/emails/*")
```
