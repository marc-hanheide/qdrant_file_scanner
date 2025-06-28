# RAG File Monitor

A Python tool that monitors directories for file changes and automatically indexes documents in a Qdrant vector database for RAG (Retrieval-Augmented Generation) systems.

## Features

- **File Monitoring**: Watches multiple directories for file changes in real-time
- **Multi-format Support**: Processes HTML, PDF, TXT, DOCX, and other text files
- **Vector Storage**: Generates embeddings and stores them in Qdrant vector database
- **Change Detection**: Only reprocesses files when they actually change
- **Chunking**: Intelligently splits large documents into smaller chunks
- **Configurable**: YAML-based configuration for all settings

## Installation

### Option 1: Quick Setup (Recommended)
1. Make sure you have Python 3.8+ installed
2. Clone or download this project
3. Run the setup script:

```bash
chmod +x setup.sh
./setup.sh
```

### Option 2: Manual Setup
1. Create a virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install PyTorch (CPU version):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

3. Install dependencies:
```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt  # For development
```

4. Install the project:
```bash
pip install -e .
```

## Configuration

1. Copy and edit the `config.yaml` file:
   - Update `directories` to point to folders you want to monitor
   - Adjust `file_extensions` for the file types you need
   - Modify Qdrant settings if your server runs on different host/port
   - Set `delete_embeddings_on_file_deletion` to control deletion behavior:
     - `true`: Delete embeddings when files are deleted (traditional behavior)
     - `false`: Keep embeddings but mark as deleted (allows restoration)

2. Make sure your Qdrant server is running at `http://localhost:6633`

## Usage

### Initial Setup and Monitoring
```bash
# Activate virtual environment (if not already active)
source .venv/bin/activate

# Scan existing files and start monitoring
rag-monitor

# Or use the script directly
python file_monitor.py
```

### Options
```bash
# Only scan existing files (no monitoring)
rag-monitor --scan-only

# Only monitor for changes (skip initial scan)
rag-monitor --monitor-only

# Use custom config file
rag-monitor --config my_config.yaml
```

### Managing Deleted Documents

When `delete_embeddings_on_file_deletion` is set to `false`, you can manage deleted documents:

```bash
# List all documents marked as deleted
rag-manage list-deleted

# Restore a deleted document (if file exists again)
rag-manage restore-deleted /path/to/file.txt

# Permanently delete a specific document
rag-manage purge-deleted /path/to/file.txt

# Clean up old deleted documents (older than 30 days)
rag-manage cleanup-deleted --older-than-days 30
```

## How It Works

1. **Initial Scan**: Processes all existing files in monitored directories
2. **Real-time Monitoring**: Watches for file creation, modification, and deletion
3. **Text Extraction**: Extracts text content based on file type
4. **Chunking**: Splits large documents into overlapping chunks
5. **Embedding**: Generates vector embeddings using sentence-transformers
6. **Storage**: Stores text and embeddings in Qdrant with metadata
7. **Change Detection**: Uses file hashes to avoid reprocessing unchanged files
8. **Deletion Handling**: Configurable behavior for when files are deleted:
   - **Delete embeddings**: Removes all traces from Qdrant (default: false)
   - **Mark as deleted**: Keeps embeddings but flags them as deleted, allowing restoration if file reappears

## Supported File Types

- **Text Files**: .txt, .md, .rtf
- **HTML**: .html, .htm
- **PDF**: .pdf (using PyPDF2)
- **Word Documents**: .docx (using python-docx)

## Qdrant Collection Schema

Each document chunk is stored with the following metadata:
- `file_path`: Original file path
- `file_hash`: MD5 hash for change detection
- `chunk_index`: Position of chunk in document
- `document`: The actual text content
- `timestamp`: When the document was indexed
- `file_size`: Size of original document
- `is_deleted`: Boolean flag indicating if the source file was deleted
- `deletion_timestamp`: When the document was marked as deleted (if applicable)

## Logging

The tool logs all operations to both console and a log file (`rag_monitor.log` by default). Check the logs for processing status and any errors.

## Development

```bash
# Activate virtual environment
source .venv/bin/activate

# Install development dependencies (if not already installed)
pip install -r requirements-dev.txt

# Run tests
pytest

# Format code
black .
```

## Troubleshooting

1. **Qdrant Connection Issues**: Verify your Qdrant server is running on the correct host/port
2. **File Processing Errors**: Check the log file for detailed error messages
3. **Permission Issues**: Ensure the tool has read access to monitored directories
4. **Large Files**: Adjust `max_file_size_mb` in config if needed

## MCP Server

The project includes a Model Context Protocol (MCP) server that provides access to the RAG database through standardized tools and resources.

### Installation

The MCP server requires additional dependencies:

```bash
pip install "mcp[cli]>=1.10.0"
```

### Usage

#### Starting the MCP Server

```bash
# Start the MCP server using FastMCP
rag-mcp-server

# Or run directly
python mcp_server.py
```

#### Development and Testing

```bash
# Test with MCP Inspector
mcp dev mcp_server.py

# Install in Claude Desktop
mcp install mcp_server.py --name "RAG Document Search"
```

### Available Tools

#### `rag_search`
Searches for relevant documents in the RAG database using semantic similarity.

**Parameters:**
- `query` (required): The search string used to find relevant documents
- `number_docs` (optional, default: 10): Number of documents to return
- `glob_pattern` (optional): Glob pattern to filter results by file path (case insensitive)
  - Examples: `"*.pdf"`, `"*/emails/*"`, `"*report*"`

**Returns:**
- Structured response with matching document chunks, including:
  - File path and content
  - Similarity score (0-1, higher is better)
  - Chunk index within the document
  - Whether the source file has been deleted

**Example Usage:**
```python
# Search for documents about "machine learning"
result = rag_search("machine learning")

# Search for PDF documents about "quarterly reports"
result = rag_search("quarterly reports", number_docs=5, glob_pattern="*.pdf")

# Search in email folder
result = rag_search("project update", glob_pattern="*/emails/*")
```

### Available Resources

#### `rag-config://server`
Provides the current RAG server configuration including Qdrant settings, embedding model configuration, monitored directories, and file extensions.

#### `rag-stats://database`
Returns statistics about the RAG database including collection information, total vectors, vector size, embedding model, and number of indexed files.

### Configuration

The MCP server uses the same `config.yaml` file as the file monitor. Make sure your Qdrant server is running and accessible before starting the MCP server.

### Integration Examples

#### With Claude Desktop

Add to your Claude Desktop configuration:

```json
{
  "mcpServers": {
    "rag-documents": {
      "command": "python",
      "args": ["/path/to/your/mcp_server.py"],
      "env": {}
    }
  }
}
```

#### With Other MCP Clients

The server implements the full MCP specification and can be used with any MCP-compatible client via stdio transport.