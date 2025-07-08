#!/usr/bin/env python3
"""
MCP Server for RAG Document Search

This server provides access to the RAG document database through the Model Context Protocol.
It implements a search tool that allows querying documents indexed in Qdrant.
"""

import logging
from pathlib import Path
from typing import List, Dict, Optional, Any
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
from pathlib import Path
from rag_file_monitor.text_extractors import TextExtractor

try:
    import yaml
except ImportError:
    import ruamel.yaml as yaml

try:
    from mcp.server.fastmcp import FastMCP
    from pydantic import BaseModel, Field
except ImportError:
    print("MCP dependencies not installed. Please run: pip install 'mcp[cli]'")
    raise

from rag_file_monitor.embedding_manager import EmbeddingManager


class RAGSearchResult(BaseModel):
    """Structured result for RAG search"""

    file_path: str = Field(description="Path to the source file")
    document: str = Field(description="Document content chunk")
    score: float = Field(description="Similarity score (0-1, higher is better)")
    chunk_index: int = Field(description="Index of the chunk within the document")
    is_deleted: bool = Field(description="Whether the source file has been deleted")
    deletion_timestamp: Optional[str] = Field(default=None, description="When the file was deleted, if applicable")
    rerank_score: Optional[float] = Field(default=None, description="Re-ranking score if re-ranker is enabled")
    original_score: Optional[float] = Field(default=None, description="Original embedding similarity score before re-ranking")


class RAGSearchResponse(BaseModel):
    """Complete response for RAG search including metadata"""

    results: List[RAGSearchResult] = Field(description="List of matching document chunks")
    query: str = Field(description="The original search query")
    total_results: int = Field(description="Number of results returned")
    filtered_by_pattern: Optional[str] = Field(default=None, description="Glob pattern used for filtering, if any")


@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[Dict[str, Any]]:
    """Manage application lifecycle with RAG components"""
    # Load configuration with smart discovery
    config_candidates = [Path.cwd() / "config.yaml", Path(__file__).parent / "config.yaml"]
    config_path = None
    for candidate in config_candidates:
        if candidate.exists():
            config_path = candidate
            break

    if config_path is None:
        raise FileNotFoundError(f"Configuration file not found. Tried: {config_candidates}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Setup logging
    logging_config = config.get("logging", {})
    log_level = getattr(logging, logging_config.get("level", "INFO").upper())

    # Configure logging
    log_file = logging_config.get("mcp_logfile", "mcp_rag_server.log")

    # Create file handler with explicit settings
    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    file_handler.setLevel(log_level)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))

    # Create stream handler
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(log_level)
    stream_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(stream_handler)

    logger = logging.getLogger(__name__)
    logger.info("Starting MCP RAG Server")

    try:
        # Initialize embedding manager
        embedding_manager = EmbeddingManager(config, slim_mode=False)
        logger.info("RAG system initialized successfully")

        yield {"embedding_manager": embedding_manager, "config": config, "logger": logger}
    except Exception as e:
        logger.error(f"Failed to initialize RAG system: {e}")
        raise
    finally:
        logger.info("Shutting down MCP RAG Server")


# Create MCP server
mcp = FastMCP("RAG Document Search", lifespan=app_lifespan)


@mcp.tool()
def rag_search(query: str, number_docs: int = 10, glob_pattern: Optional[str] = None) -> RAGSearchResponse:
    """
    Search for relevant documents in the RAG database.

    This tool searches through indexed documents on this conputer using semantic similarity
    to find the most relevant content chunks for a given query.

    Args:
        query: The search string used to find relevant documents
        number_docs: Number of documents to return (default: 10)
        glob_pattern: Optional glob pattern to filter results by file path (case insensitive)
                     Examples: "*.pdf", "*/emails/*", "*report*"

    Returns:
        RAGSearchResponse: Structured response containing matching document chunks
    """
    # Get application context
    ctx = mcp.get_context()
    embedding_manager = ctx.request_context.lifespan_context["embedding_manager"]
    logger = ctx.request_context.lifespan_context["logger"]

    try:
        logger.info(f"RAG search query: '{query}' (limit: {number_docs}, pattern: {glob_pattern})")

        # Use the existing search_similar method with glob pattern support
        raw_results = embedding_manager.search_similar(
            query=query,
            limit=number_docs,
            include_deleted=True,  # Don't include deleted documents by default
            glob_pattern=glob_pattern,
        )

        # Convert to structured results
        structured_results = []
        for result in raw_results:
            structured_result = RAGSearchResult(
                file_path=result.get("file_path", ""),
                document=result.get("document", ""),
                score=result.get("score", 0.0),
                chunk_index=result.get("chunk_index", 0),
                is_deleted=result.get("is_deleted", False),
                deletion_timestamp=result.get("deletion_timestamp"),
                rerank_score=result.get("rerank_score"),
                original_score=result.get("original_score"),
            )
            structured_results.append(structured_result)

        response = RAGSearchResponse(
            results=structured_results, query=query, total_results=len(structured_results), filtered_by_pattern=glob_pattern
        )

        logger.info(f"RAG search completed: {len(structured_results)} results returned")
        return response

    except Exception as e:
        logger.error(f"Error during RAG search: {e}")
        # Return empty results on error rather than failing completely
        return RAGSearchResponse(results=[], query=query, total_results=0, filtered_by_pattern=glob_pattern)


@mcp.resource("rag-config://server", title="RAG Server Configuration")
def get_rag_config() -> str:
    """Get the current RAG server configuration"""
    ctx = mcp.get_context()
    config = ctx.request_context.lifespan_context["config"]

    config_summary = {
        "qdrant": config.get("qdrant", {}),
        "mcp": __file__,
        "embedding": config.get("embedding", {}),
        "directories": config.get("directories", {}),
        "file_extensions": config.get("file_extensions", []),
    }

    return yaml.dump(config_summary, default_flow_style=False)


@mcp.resource("rag-stats://database", title="RAG Database Statistics")
def get_database_stats() -> str:
    """Get statistics about the RAG database"""
    ctx = mcp.get_context()
    embedding_manager = ctx.request_context.lifespan_context["embedding_manager"]

    try:
        # Get collection information
        collection_info = embedding_manager.client.get_collection(collection_name=embedding_manager.collection_name)

        # Get memory statistics
        memory_stats = embedding_manager.get_memory_stats()

        stats = {
            "collection_name": embedding_manager.collection_name,
            "collection_status": str(collection_info.status),
            "vector_size": embedding_manager.vector_size,
            "embedding_model": embedding_manager.model_name,
            "indexed_files": len(embedding_manager.file_hashes),
            "memory_stats": memory_stats,
        }

        return yaml.dump(stats, default_flow_style=False)

    except Exception as e:
        return f"Error retrieving database statistics: {str(e)}"


@mcp.prompt()
def find_files_about(topic: str) -> str:
    """Find specific files about a particular topic or content.

    This prompt helps locate relevant documents that can be opened directly.
    Use this when you need to find files containing specific information.

    Args:
        topic: What you're looking for (e.g., "machine learning papers", "budget reports", "meeting notes")
    """
    return f"""I need to find files about "{topic}". Please use the rag_search tool to search for relevant documents.

After finding the results:
1. List the most relevant files with their paths
2. If the files exist and are accessible, offer to open or read specific ones
3. Provide file:// links for files that can be opened directly
4. Summarize what types of content were found

Search query: {topic}"""


@mcp.prompt()
def summarize_documents_about(topic: str, file_pattern: str = None) -> str:
    """Summarize information from multiple local documents about a specific topic.

    This prompt helps gather and synthesize information from various sources.
    Use this when you need a comprehensive overview from multiple documents available locally,
    or when requested by the user specifically.

    Args:
        topic: The subject to summarize (e.g., "quarterly financial performance", "project status updates")
        file_pattern: Optional glob pattern to filter specific file types or locations
    """
    pattern_instruction = f"\nUse glob pattern: {file_pattern}" if file_pattern else ""

    return f"""I need a comprehensive summary about "{topic}" from available documents.{pattern_instruction}

Please follow this process:
1. First, use rag_search to find relevant documents about "{topic}"
2. Identify the most relevant and recent documents
3. For key documents, read their full content to get complete context if that is required
4. Synthesize the information into a structured summary covering:
   - Key findings or main points
   - Important dates and timelines
   - Relevant numbers, statistics, or metrics
   - Action items or next steps (if applicable)
   - Sources used (with file paths)

Search query: {topic}"""


@mcp.prompt()
def find_emails_about(subject_or_content: str, date_range: str = None) -> str:
    """Find specific emails based on subject or content.

    This prompt helps locate emails which are stored as HTML files in OneDrive.
    Use this when you need to find email communications about specific topics.

    Args:
        subject_or_content: What to search for in emails (subject line or content)
        date_range: Optional date range (e.g., "last month", "2024", "January 2025")
    """
    email_pattern = "*/OneDrive-UniversityofLincoln/Emails/*/*.html"
    date_instruction = f" from {date_range}" if date_range else ""

    return f"""I need to find emails about "{subject_or_content}"{date_instruction}.

Please search for emails using these steps:
1. Use rag_search with glob_pattern="{email_pattern}" to find relevant emails
2. Look for emails matching the subject or containing the specified content
3. Present the results showing:
   - Email subject/title (from filename or content)
   - Date information (if available)
   - Sender/recipient information (if found in content)
   - Brief preview of relevant content
   - Full file path for reference

Search in: /Users/mhanheide/Library/CloudStorage/OneDrive-UniversityofLincoln/Emails
File type: all stoared email files end in `.html`
Search query: {subject_or_content}"""


@mcp.prompt()
def comprehensive_search(query: str, search_strategy: str = "broad") -> str:
    """Perform a comprehensive search across all available documents.

    This prompt provides a flexible search strategy that adapts based on the type of information needed.
    Use this for complex queries that might require multiple search approaches.

    Args:
        query: The search query or question
        search_strategy: Search approach - "broad" (cast wide net), "focused" (specific results), or "deep" (detailed analysis)
    """
    if search_strategy == "focused":
        strategy_instruction = """
Use a focused search approach:
1. Search with specific keywords
2. Limit results to most relevant matches
3. Provide precise, targeted information"""
    elif search_strategy == "deep":
        strategy_instruction = """
Use a deep analysis approach:
1. Search broadly first to identify relevant documents
2. Read full content of key documents
3. Cross-reference information across sources
4. Provide detailed analysis with citations"""
    else:  # broad
        strategy_instruction = """
Use a broad search approach:
1. Search with multiple keyword variations
2. Include different file types and locations
3. Cast a wide net to capture all relevant information"""

    return f"""I need to search for information about: "{query}"

Search Strategy: {search_strategy}
{strategy_instruction}

Please help me find and organize information by:
1. Using rag_search with appropriate keywords and patterns
2. Identifying the most relevant documents
3. Organizing results by relevance and type
4. Providing actionable next steps or direct links where possible

For emails, remember to use the pattern: */OneDrive-UniversityofLincoln/Emails/*/*.html
For specific file types, use appropriate glob patterns

Search query: {query}"""


@mcp.tool()
def scan_file(file_path: str) -> Dict[str, Any]:
    """
    Scan and index a specific file into the RAG database.

    This tool allows ad-hoc scanning of individual files without waiting for the file monitor.
    The file will always be re-indexed when requested.

    Args:
        file_path: Absolute path to the file to scan and index

    Returns:
        Dict containing scan results and status information
    """
    # Get application context
    ctx = mcp.get_context()
    embedding_manager = ctx.request_context.lifespan_context["embedding_manager"]
    logger = ctx.request_context.lifespan_context["logger"]

    try:

        # Validate file path
        file_path = Path(file_path).resolve()

        if not file_path.exists():
            return {
                "success": False,
                "error": f"File not found: {file_path}",
                "file_path": str(file_path),
            }

        if not file_path.is_file():
            return {
                "success": False,
                "error": f"Path is not a file: {file_path}",
                "file_path": str(file_path),
            }

        logger.info(f"Ad-hoc file scan requested: {file_path}")

        # Process the file - let embedding_manager handle hashing internally
        file_path_str = str(file_path)
        text_extractor = TextExtractor()

        # Extract text from file
        extracted_text = text_extractor.extract_text(file_path_str)

        if not extracted_text or not extracted_text.strip():
            return {
                "success": False,
                "error": "No text could be extracted from file - may be unsupported file type or empty",
                "file_path": file_path_str,
            }

        # Index the document - embedding_manager will handle hash calculation internally
        # Pass empty string as hash since we want to force re-indexing
        embedding_manager.index_document(file_path_str, extracted_text, "")

        # Get chunk count by checking how many chunks would be generated
        chunks = embedding_manager.chunk_text(extracted_text)
        chunks_added = len(chunks) if chunks else 0

        message = f"File successfully scanned and indexed with {chunks_added} chunks"
        logger.info(f"Ad-hoc file scan completed: {file_path} ({chunks_added} chunks)")

        return {
            "success": True,
            "message": message,
            "file_path": file_path_str,
            "action": "indexed",
            "chunks_added": chunks_added,
            "file_size": file_path.stat().st_size,
            "file_modified": file_path.stat().st_mtime,
        }

    except Exception as e:
        logger.error(f"Error during ad-hoc file scan: {e}")
        return {
            "success": False,
            "error": f"Error scanning file: {str(e)}",
            "file_path": (
                file_path_str if "file_path_str" in locals() else str(file_path) if "file_path" in locals() else "unknown"
            ),
        }


def main():
    """Main entry point for the MCP server"""
    # Run the FastMCP server
    mcp.run()


if __name__ == "__main__":
    main()
