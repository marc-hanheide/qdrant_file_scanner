# RAG Search CLI Tool

A comprehensive command-line interface for searching and managing documents in the RAG (Retrieval-Augmented Generation) database.

## Installation

The `rag-search` command is automatically installed when you install the `rag-file-monitor` package:

```bash
pip install -e .
```

## Features

- **Semantic Search**: Find documents using natural language queries
- **Similarity Search**: Find documents similar to an example file
- **Glob Pattern Search**: Search for documents by file path patterns
- **Advanced Filtering**: Filter by date range, minimum similarity score, and file patterns
- **Flexible Output**: Choose between brief, detailed, or JSON output formats
- **Bulk Operations**: Delete multiple documents from the database with confirmation
- **Safe Operations**: Dry-run mode and interactive confirmations

## Usage

### Basic Commands

```bash
# Search for documents about machine learning
rag-search --query "machine learning algorithms"

# Find files similar to an example document
rag-search --example ~/Documents/research_paper.pdf

# Search for all PDF files
rag-search --glob "*.pdf"

# Search with minimum similarity score
rag-search --query "budget report" --min-score 0.7

# Get detailed output
rag-search --query "project status" --verbose

# Get only file paths (useful for scripting)
rag-search --query "meeting notes" --files-only
```

### Advanced Filtering

```bash
# Search with date range
rag-search --query "quarterly report" --start-date "2024-01-01" --end-date "2024-03-31"

# Search with file pattern filter
rag-search --query "presentation" --file-pattern "*.pptx"

# Include deleted documents in search
rag-search --query "old files" --include-deleted

# Limit number of results
rag-search --query "documentation" --limit 5
```

### Date Filtering

The date filtering supports various formats:

```bash
# Absolute dates
rag-search --query "report" --start-date "2024-01-01"
rag-search --query "report" --start-date "2024/01/01"

# Relative dates
rag-search --query "recent" --start-date "today"
rag-search --query "recent" --start-date "yesterday"
rag-search --query "recent" --start-date "7 days ago"
rag-search --query "recent" --start-date "2 weeks ago"
rag-search --query "recent" --start-date "1 month ago"
```

### Bulk Delete Operations

```bash
# Preview what would be deleted (dry run)
rag-search --query "temporary files" --delete --dry-run

# Delete with interactive confirmation
rag-search --glob "*/temp/*" --delete

# Delete without confirmation (use with caution!)
rag-search --glob "*/cache/*" --delete --force
```

### Output Formats

```bash
# Default output (brief with scores)
rag-search --query "test"

# Detailed output with document previews
rag-search --query "test" --verbose

# Only file paths (one per line)
rag-search --query "test" --files-only

# JSON output for programmatic use
rag-search --query "test" --json
```

## Command Line Options

### Search Types (Required - choose one)

- `--query, -q`: Search query for semantic search
- `--example, -e`: Path to example file for similarity search
- `--glob, -g`: Glob pattern for file path matching

### Search Options

- `--limit, -l`: Maximum number of results (default: 10, ignored for glob)
- `--min-score, -s`: Minimum similarity score (0.0-1.0, default: 0.0)
- `--include-deleted`: Include deleted documents in results

### Filtering Options

- `--start-date`: Start date filter (YYYY-MM-DD, "today", "1 week ago", etc.)
- `--end-date`: End date filter (YYYY-MM-DD, "today", "1 week ago", etc.)
- `--file-pattern`: Additional glob pattern to filter results by file path

### Output Options

- `--verbose, -v`: Show detailed information for each result
- `--files-only, -f`: Output only file paths (one per line)
- `--json`: Output results as JSON

### Deletion Options

- `--delete, -d`: Delete found documents from the database
- `--force`: Skip confirmation prompts for deletion
- `--dry-run`: Show what would be deleted without actually deleting

### Configuration Options

- `--config, -c`: Path to configuration file (default: config.yaml)
- `--debug`: Enable debug logging

## Examples

### 1. Basic Semantic Search

```bash
# Find documents about machine learning
rag-search --query "machine learning algorithms"

# Find budget-related documents with high confidence
rag-search --query "budget analysis" --min-score 0.8 --limit 5
```

### 2. Similarity Search

```bash
# Find documents similar to a research paper
rag-search --example ~/Documents/research_paper.pdf --limit 10

# Find similar documents with minimum similarity
rag-search --example ~/Documents/template.docx --min-score 0.6
```

### 3. File Pattern Search

```bash
# Find all PDF files
rag-search --glob "*.pdf"

# Find files in a specific directory
rag-search --glob "/Users/*/Documents/*.docx"

# Find files with specific naming pattern
rag-search --glob "*report*2024*"
```

### 4. Date-based Filtering

```bash
# Find recent documents
rag-search --query "meeting notes" --start-date "1 week ago"

# Find documents from a specific period
rag-search --query "quarterly report" \
  --start-date "2024-01-01" \
  --end-date "2024-03-31"
```

### 5. Combined Filters

```bash
# Complex search with multiple filters
rag-search --query "project status" \
  --min-score 0.7 \
  --file-pattern "*.pdf" \
  --start-date "2024-01-01" \
  --limit 10 \
  --verbose
```

### 6. Bulk Operations

```bash
# Preview deletion of temporary files
rag-search --glob "*/tmp/*" --delete --dry-run

# Delete old cache files with confirmation
rag-search --glob "*/cache/*" --delete

# Force delete without confirmation (careful!)
rag-search --query "temporary files" --delete --force
```

### 7. Output for Scripting

```bash
# Get file paths for further processing
files=$(rag-search --query "reports" --files-only)

# JSON output for programmatic use
rag-search --query "data" --json | jq '.[] | .file_path'
```

## Configuration

The tool uses the same `config.yaml` file as the other RAG tools. It will automatically look for the configuration file in:

1. Current working directory (`./config.yaml`)
2. Package directory

You can also specify a custom config file:

```bash
rag-search --config /path/to/custom/config.yaml --query "test"
```

## Integration with Other Tools

The `rag-search` tool works seamlessly with other RAG tools in the package:

- **rag-monitor**: Indexes files for search
- **rag-manage**: Manages deleted documents
- **rag-mcp-server**: Provides MCP server interface

## Tips and Best Practices

1. **Start with broad queries**: Begin with general terms and refine with filters
2. **Use minimum scores**: Set appropriate `--min-score` values to filter out low-quality matches
3. **Test with dry-run**: Always use `--dry-run` before bulk deletions
4. **Combine filters**: Use multiple filters to narrow down results effectively
5. **Use files-only for scripting**: The `--files-only` option is perfect for shell scripts
6. **Check with verbose first**: Use `--verbose` to understand what documents are being found

## Troubleshooting

### Common Issues

1. **No results found**: 
   - Check if the database is populated by running `rag-monitor`
   - Try broader search terms
   - Lower the `--min-score` threshold

2. **Configuration not found**:
   - Ensure `config.yaml` exists in the current directory
   - Use `--config` to specify the path explicitly

3. **Qdrant connection errors**:
   - Ensure Qdrant is running on the configured host/port
   - Check the Qdrant configuration in `config.yaml`

### Debug Mode

Use `--debug` to see detailed information about what the tool is doing:

```bash
rag-search --query "test" --debug
```

This will show database connections, search operations, and other internal details.
