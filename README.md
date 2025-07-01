# RAG File Monitor

A Python tool that monitors directories for file changes and automatically indexes documents in a Qdrant vector database for RAG (Retrieval-Augmented Generation) systems.

## Features

- **File Monitoring**: Watches multiple directories for file changes in real-time
- **Multi-format Support**: Processes HTML, PDF, TXT, DOCX, and other text files
- **Vector Storage**: Generates embeddings and stores them in Qdrant vector database
- **Change Detection**: Only reprocesses files when they actually change
- **Chunking**: Intelligently splits large documents into smaller chunks
- **Configurable**: YAML-based configuration for all settings
- **Search CLI**: Comprehensive command-line tool for searching and managing documents
- **MCP Server**: Model Context Protocol server for integration with AI assistants

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
   - Update `directories` to point to folders you want to monitor (see Directory Configuration below)
   - Adjust `file_extensions` for the file types you need (global defaults)
   - Modify Qdrant settings if your server runs on different host/port
   - Set `delete_embeddings_on_file_deletion` to control deletion behavior:
     - `true`: Delete embeddings when files are deleted (traditional behavior)
     - `false`: Keep embeddings but mark as deleted (allows restoration)

2. Make sure your Qdrant server is running at `http://localhost:6633`

### Directory Configuration

The tool supports two configuration formats for directories:

#### New Format (Recommended): Per-Directory Settings
```yaml
directories:
  "/path/to/documents":
    ignore_extensions: []  # Use all global file_extensions
    max_filesize: 0  # Use global max_file_size_mb setting
  "/path/to/emails":
    ignore_extensions: [".xlsx", ".pptx"]  # Skip spreadsheets and presentations
    max_filesize: 5  # Maximum 5MB per file in this directory
  "/path/to/downloads":
    ignore_extensions: [".html", ".htm", ".rtf"]  # Skip web files and RTF
    max_filesize: 2  # Maximum 2MB per file in downloads (temporary files)
```

Each directory can specify:
- `ignore_extensions`: Array of file extensions to skip for this directory
- `max_filesize`: Maximum file size in MB for files in this directory (0 = use global default)
- Extensions are applied on top of the global `file_extensions` list
- Empty array for ignore_extensions means all global extensions will be processed

#### Legacy Format (Still Supported)
```yaml
directories:
  - "/path/to/documents"
  - "/path/to/emails"
  - "/path/to/downloads"
```

This format will process all global `file_extensions` in all directories.

### Configuration Examples

Here are some practical examples of how to configure directories with per-directory settings:

#### Example 1: Academic/Research Setup
```yaml
directories:
  "/Users/username/Documents/Papers":
    ignore_extensions: []  # Process all file types
    max_filesize: 20  # Allow large academic papers up to 20MB
  "/Users/username/Documents/Email_Attachments":
    ignore_extensions: [".html", ".htm"]  # Skip HTML emails
    max_filesize: 5  # Email attachments usually smaller
  "/Users/username/Downloads":
    ignore_extensions: [".html", ".htm", ".rtf", ".xlsx"]  # Skip web files and spreadsheets
    max_filesize: 2  # Downloads are often temporary, keep small
  "/Users/username/Desktop":
    ignore_extensions: [".pptx", ".xlsx"]  # Skip presentations and spreadsheets on desktop
    max_filesize: 10  # Medium size limit for desktop files
```

#### Example 2: Business Setup
```yaml
directories:
  "/Users/username/OneDrive/Documents":
    ignore_extensions: []  # Process all document types
    max_filesize: 15  # Business documents can be larger
  "/Users/username/OneDrive/Shared":
    ignore_extensions: [".xlsx", ".pptx"]  # Skip large presentation files
    max_filesize: 8  # Shared files medium size
  "/Users/username/Downloads":
    ignore_extensions: [".html", ".htm", ".rtf"]  # Skip temporary web downloads
    max_filesize: 3  # Keep downloads small
```

#### Example 3: Selective Processing
```yaml
directories:
  "/Users/username/Important_Docs":
    ignore_extensions: []  # Process everything important
    max_filesize: 50  # Large limit for important documents
  "/Users/username/Archive":
    ignore_extensions: [".docx", ".xlsx", ".pptx"]  # Only text and PDFs in archive
    max_filesize: 25  # Medium limit for archived content
  "/Users/username/Temp":
    ignore_extensions: [".pdf", ".docx", ".xlsx", ".pptx"]  # Only plain text in temp
    max_filesize: 1  # Very small limit for temporary files
```

The key benefits of per-directory configuration:
- **Reduce noise**: Skip irrelevant file types in specific directories
- **Improve performance**: Don't process large files where they're not needed
- **Control file sizes**: Set different size limits based on directory purpose
- **Better organization**: Tailor processing to the content type of each directory
- **Fine-grained control**: Different rules for different use cases

## Usage

After installation, you have access to several command-line tools:

- **`rag-monitor`**: Main file monitoring and indexing tool
- **`rag-search`**: Search and manage documents in the database
- **`rag-manage`**: Manage deleted documents and database maintenance
- **`rag-mcp-server`**: Run the Model Context Protocol server

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

### Scheduled Scanning with Cron

For periodic scanning without continuous monitoring, you can set up a cron job to run `rag-monitor --scan-only` at regular intervals. This is useful for systems where you want to index files periodically rather than monitoring in real-time.

#### Setting up a Cron Job

1. **Create a wrapper script** (recommended approach for complex paths):

Create a file called `rag-scan.sh` in your project directory:

```bash
#!/bin/bash
# RAG Monitor Scan Script
# This script ensures proper environment setup for cron execution

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Activate virtual environment and run scan
source .venv/bin/activate
rag-monitor -c ./config.yaml --scan-only
```

Make it executable:
```bash
chmod +x rag-scan.sh
```

2. **Edit your crontab**:
```bash
crontab -e
```

3. **Add the cron job** (example runs every 4 hours):
```bash
# RAG Monitor periodic scan - runs every 4 hours
0 */4 * * * /path/to/your/rag-file-monitor/rag-scan.sh > ~/Library/Logs/rag-monitor-last-scan.log 2>&1
```

#### Alternative: Direct crontab entry

If you prefer a single-line crontab entry without a wrapper script:

```bash
# RAG Monitor scan-only job - runs every 4 hours
0 */4 * * * cd /path/to/your/rag-file-monitor && .venv/bin/rag-monitor -c ./config.yaml --scan-only > ~/Library/Logs/rag-monitor-last-scan.log 2>&1
```

#### Cron Schedule Examples

```bash
# Every 4 hours
0 */4 * * *

# Every 6 hours at the top of the hour
0 */6 * * *

# Daily at 2 AM
0 2 * * *

# Every weekday at 9 AM
0 9 * * 1-5

# Every Sunday at midnight
0 0 * * 0
```

#### Important Notes

- Replace `/path/to/your/rag-file-monitor` with your actual installation path
- The log file `~/Library/Logs/rag-monitor-last-scan.log` will be overwritten each time (only keeps the last run)
- Make sure your Qdrant server is running when the cron job executes
- Consider using absolute paths in cron jobs to avoid PATH issues
- The wrapper script approach is more robust as it handles environment activation automatically

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

### Searching Documents

The `rag-search` CLI tool provides comprehensive search capabilities:

```bash
# Semantic search
rag-search --query "machine learning algorithms"

# Find similar documents
rag-search --example ~/Documents/research_paper.pdf

# Search by file pattern
rag-search --glob "*.pdf" --start-date "2024-01-01"

# Delete documents matching a pattern
rag-search --glob "*/temp/*" --delete --dry-run
```

For detailed usage examples, see [RAG_SEARCH_CLI.md](RAG_SEARCH_CLI.md).

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