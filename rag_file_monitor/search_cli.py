#!/usr/bin/env python3
"""
RAG Search CLI Tool

A comprehensive command-line interface for searching and managing documents
in the RAG (Retrieval-Augmented Generation) database.

Features:
- Semantic search by query text
- Search by example document (find similar files)
- Search by file glob patterns
- Filter by date range and minimum match score
- Configurable output formats (brief or detailed)
- Bulk delete operations with confirmation
"""

import sys
import argparse
import logging
import fnmatch
import os
from pathlib import Path
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
import json

try:
    import yaml
except ImportError:
    import ruamel.yaml as yaml

from .embedding_manager import EmbeddingManager
from .text_extractors import TextExtractor


class RAGSearchCLI:
    """Command-line interface for RAG document search"""

    def __init__(self, config_path: str = None):
        """Initialize the CLI with configuration"""
        if config_path is None:
            # Look for config.yaml in the current directory, then in the package directory
            config_candidates = [Path.cwd() / "config.yaml", Path(__file__).parent.parent / "config.yaml"]
            config_path = None
            for candidate in config_candidates:
                if candidate.exists():
                    config_path = candidate
                    break

            if config_path is None:
                raise FileNotFoundError(f"Configuration file not found. Tried: {config_candidates}")

        self.config_path = Path(config_path)
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        with open(self.config_path, "r") as f:
            self.config = yaml.safe_load(f)

        # Setup logging
        logging.basicConfig(
            level=logging.WARNING, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"  # Keep quiet by default
        )
        self.logger = logging.getLogger(__name__)

        # Initialize embedding manager
        self.embedding_manager = EmbeddingManager(self.config, slim_mode=False)
        self.text_extractor = TextExtractor()

    def search_by_query(
        self,
        query: str,
        limit: int = 10,
        min_score: float = 0.0,
        include_deleted: bool = False,
        glob_pattern: str = None,
        start_date: str = None,
        end_date: str = None,
    ) -> List[Dict]:
        """Search documents by semantic query"""
        self.logger.info(f"Searching for query: '{query}'")

        # Get results from embedding manager
        results = self.embedding_manager.search_similar(
            query=query, limit=limit, include_deleted=include_deleted, glob_pattern=glob_pattern
        )

        # Apply additional filters
        filtered_results = []
        for result in results:
            # Filter by minimum score
            if result.get("score", 0) < min_score:
                continue

            # Filter by date range (if provided)
            if start_date or end_date:
                # Try to get timestamp from result or file modification time
                file_timestamp = self._get_file_timestamp(result)
                if file_timestamp:
                    if start_date and file_timestamp < self._parse_date(start_date):
                        continue
                    if end_date and file_timestamp > self._parse_date(end_date):
                        continue

            filtered_results.append(result)

        return filtered_results

    def search_by_example(
        self,
        example_file: str,
        limit: int = 10,
        min_score: float = 0.0,
        include_deleted: bool = False,
        glob_pattern: str = None,
        start_date: str = None,
        end_date: str = None,
    ) -> List[Dict]:
        """Search for documents similar to an example file"""
        if not os.path.exists(example_file):
            raise FileNotFoundError(f"Example file not found: {example_file}")

        self.logger.info(f"Searching for files similar to: {example_file}")

        # Extract text from the example file
        try:
            example_text = self.text_extractor.extract_text(example_file)
            if not example_text or len(example_text.strip()) < 10:
                raise ValueError(f"Could not extract meaningful text from {example_file}")
        except Exception as e:
            raise ValueError(f"Error extracting text from {example_file}: {e}")

        # Use the extracted text as a query
        return self.search_by_query(
            query=example_text[:2000],  # Limit to first 2000 chars for efficiency
            limit=limit,
            min_score=min_score,
            include_deleted=include_deleted,
            glob_pattern=glob_pattern,
            start_date=start_date,
            end_date=end_date,
        )

    def search_by_glob(self, glob_pattern: str, start_date: str = None, end_date: str = None) -> List[Dict]:
        """Search documents by file glob pattern"""
        self.logger.info(f"Searching for files matching pattern: {glob_pattern}")

        # Use scroll to get all documents matching the pattern
        try:
            # Convert to case-insensitive matching for better results
            scroll_result = self.embedding_manager.client.scroll(
                collection_name=self.embedding_manager.collection_name,
                limit=10000,  # Large limit to get all results
                with_payload=True,
            )

            results = []
            seen_files = set()

            for point in scroll_result[0]:
                file_path = point.payload.get("file_path", "")
                if not file_path:
                    continue

                # Check if file matches glob pattern (case insensitive)
                if not fnmatch.fnmatch(file_path.lower(), glob_pattern.lower()):
                    continue

                # Avoid duplicate files
                if file_path in seen_files:
                    continue
                seen_files.add(file_path)

                result = {
                    "file_path": file_path,
                    "document": point.payload.get("document", ""),
                    "score": 1.0,  # Perfect match for glob search
                    "chunk_index": point.payload.get("chunk_index", 0),
                    "is_deleted": point.payload.get("is_deleted", False),
                    "deletion_timestamp": point.payload.get("deletion_timestamp"),
                }

                # Filter by date range
                if start_date or end_date:
                    file_timestamp = self._get_file_timestamp(result)
                    if file_timestamp:
                        if start_date and file_timestamp < self._parse_date(start_date):
                            continue
                        if end_date and file_timestamp > self._parse_date(end_date):
                            continue

                results.append(result)

            # Sort by file path for consistent output
            results.sort(key=lambda x: x["file_path"])
            return results

        except Exception as e:
            self.logger.error(f"Error in glob search: {e}")
            return []

    def bulk_delete(self, results: List[Dict], force: bool = False, dry_run: bool = False):
        """Delete vectors for all documents in the results"""
        if not results:
            print("No documents to delete.")
            return

        # Group results by file path to avoid duplicate deletions
        files_to_delete = {}
        for result in results:
            file_path = result.get("file_path")
            if file_path and file_path not in files_to_delete:
                files_to_delete[file_path] = result

        total_files = len(files_to_delete)
        print(f"Found {total_files} unique files to delete from the database.")

        if dry_run:
            print("\nDRY RUN - Files that would be deleted:")
            for file_path in sorted(files_to_delete.keys()):
                print(f"  {file_path}")
            return

        if not force:
            print("\nFiles to be deleted from the database:")
            for file_path in sorted(files_to_delete.keys()):
                print(f"  {file_path}")

            confirm = input(f"\nAre you sure you want to delete {total_files} files from the database? (yes/no): ")
            if confirm.lower() not in ["yes", "y"]:
                print("Deletion cancelled.")
                return

        # Perform deletions
        deleted_count = 0
        failed_count = 0

        for file_path in files_to_delete:
            try:
                if not force:
                    # Interactive confirmation for each file
                    confirm = input(f"Delete {file_path}? (y/n/a/q): ").lower()
                    if confirm == "q":
                        print("Deletion process stopped.")
                        break
                    elif confirm == "a":
                        force = True  # Switch to automatic mode
                    elif confirm not in ["y", "a"]:
                        continue

                self.embedding_manager.delete_document(file_path, force_delete=True)
                deleted_count += 1
                print(f"✓ Deleted: {file_path}")

            except Exception as e:
                failed_count += 1
                print(f"✗ Failed to delete {file_path}: {e}")

        print(f"\nDeletion complete: {deleted_count} successful, {failed_count} failed")

    def _get_file_timestamp(self, result: Dict) -> Optional[datetime]:
        """Get timestamp for a file result"""
        file_path = result.get("file_path")
        if not file_path:
            return None

        try:
            # Try to get file modification time
            if os.path.exists(file_path):
                return datetime.fromtimestamp(os.path.getmtime(file_path))

            # For deleted files, try to parse deletion timestamp
            if result.get("deletion_timestamp"):
                try:
                    return datetime.fromisoformat(result["deletion_timestamp"])
                except:
                    pass

        except Exception:
            pass

        return None

    def _parse_date(self, date_str: str) -> datetime:
        """Parse date string to datetime object"""
        # Handle relative dates
        date_str = date_str.lower().strip()

        if date_str in ["today"]:
            return datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        elif date_str in ["yesterday"]:
            return (datetime.now() - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        elif date_str.endswith(" days ago"):
            try:
                days = int(date_str.split()[0])
                return (datetime.now() - timedelta(days=days)).replace(hour=0, minute=0, second=0, microsecond=0)
            except:
                pass
        elif date_str.endswith(" weeks ago"):
            try:
                weeks = int(date_str.split()[0])
                return (datetime.now() - timedelta(weeks=weeks)).replace(hour=0, minute=0, second=0, microsecond=0)
            except:
                pass
        elif date_str.endswith(" months ago"):
            try:
                months = int(date_str.split()[0])
                return (datetime.now() - timedelta(days=months * 30)).replace(hour=0, minute=0, second=0, microsecond=0)
            except:
                pass

        # Try standard date formats
        for fmt in ["%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y", "%d/%m/%Y", "%Y-%m-%d %H:%M:%S"]:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue

        # Try ISO format
        try:
            return datetime.fromisoformat(date_str)
        except ValueError:
            pass

        raise ValueError(f"Could not parse date: {date_str}")

    def format_results(
        self, results: List[Dict], verbose: bool = False, files_only: bool = True, document_mode: bool = False
    ) -> str:
        """Format search results for display"""
        if not results:
            return "No results found." if not files_only else ""

        if files_only:
            # Return only unique file paths
            file_paths = sorted(set(result.get("file_path", "") for result in results))
            return "\n".join(file_paths)

        if document_mode:
            # Generate LLM-friendly markdown document
            return self._format_as_markdown_document(results)

        output = []
        output.append(f"Found {len(results)} results:\n")

        for i, result in enumerate(results, 1):
            file_path = result.get("file_path", "Unknown")
            score = result.get("score", 0.0)
            is_deleted = result.get("is_deleted", False)

            # Basic info
            status = " [DELETED]" if is_deleted else ""
            output.append(f"{i:3}. {file_path}{status}")

            if verbose:
                output.append(f"     Score: {score:.4f}")
                output.append(f"     Chunk: {result.get('chunk_index', 0)}")

                if is_deleted and result.get("deletion_timestamp"):
                    output.append(f"     Deleted: {result.get('deletion_timestamp')}")

                # Show document preview
                document = result.get("document", "")
                if document:
                    preview = document[:200] + "..." if len(document) > 200 else document
                    preview = preview.replace("\n", " ").replace("\r", " ")
                    output.append(f"     Preview: {preview}")

                output.append("")  # Empty line for spacing
            else:
                # Brief format - just show score
                output.append(f"     Score: {score:.4f}")

        return "\n".join(output)

    def _format_as_markdown_document(self, results: List[Dict]) -> str:
        """Format results as a comprehensive markdown document for LLM consumption"""
        if not results:
            return "# Search Results\n\nNo results found."

        # Group results by file to avoid duplicates
        files_dict = {}
        for result in results:
            file_path = result.get("file_path", "Unknown")
            if file_path not in files_dict:
                files_dict[file_path] = []
            files_dict[file_path].append(result)

        output = []
        output.append("# Search Results\n")
        output.append(f"Found {len(results)} document chunks from {len(files_dict)} unique files.\n")

        for file_path, file_results in files_dict.items():
            # Sort chunks by chunk index
            file_results.sort(key=lambda x: x.get("chunk_index", 0))

            output.append(f"## [`{file_path}`](file://{file_path})")
            output.append("")  # Empty line after header

            # File metadata
            first_result = file_results[0]
            is_deleted = first_result.get("is_deleted", False)

            if is_deleted:
                output.append("**Status:** DELETED")
                if first_result.get("deletion_timestamp"):
                    output.append(f"**Deleted:** {first_result.get('deletion_timestamp')}")

            output.append(f"**Number of chunks:** {len(file_results)}\n")

            # Show all chunks for this file
            for i, result in enumerate(file_results):
                chunk_index = result.get("chunk_index", 0)
                score = result.get("score", 0.0)
                document = result.get("document", "")

                output.append(f"\n### Chunk {chunk_index} (Similarity Score: {score:.4f})")

                if document:
                    # Format the document content as a markdown quote
                    # Split into lines and prefix each with >
                    document_lines = document.split("\n")
                    quoted_lines = [f"> {line}" if line.strip() else ">" for line in document_lines]
                    output.append("\n".join(quoted_lines))
                else:
                    output.append("> *No content available*")

                output.append("")  # Empty line for spacing

            output.append("---\n")  # Separator between files

        return "\n".join(output)


def create_parser():
    """Create the argument parser for the CLI"""
    parser = argparse.ArgumentParser(
        description="RAG Search CLI - Search and manage documents in the RAG database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic semantic search (default mode)
  rag-search "machine learning algorithms"
  rag-search --query "machine learning algorithms"
  echo "machine learning" | rag-search
  
  # Search with glob filter
  rag-search "budget report" --glob "*.pdf"
  
  # Find similar files
  rag-search --example ~/Documents/report.pdf
  
  # Search by file pattern only
  rag-search --glob-only "*.pdf"
  
  # With additional options
  rag-search "project status" --limit 5 --min-score 0.7
  
  # Different output formats
  rag-search "test" --verbose
  rag-search "test" --brief
  rag-search "test" --document
  rag-search "test" --json
  
  # Bulk delete with confirmation
  rag-search "temp file" --delete --dry-run
        """,
    )

    # Positional argument for query (default mode)
    parser.add_argument("query", nargs="?", help="Search query for semantic search (default mode)")

    # Search type options
    search_group = parser.add_mutually_exclusive_group()
    search_group.add_argument("--example", "-e", type=str, help="Path to example file for similarity search")
    search_group.add_argument("--glob-only", type=str, help="Search only by glob pattern (no semantic search)")

    # Filtering options
    parser.add_argument("--glob", "-g", type=str, help="Glob pattern to filter results by file path")
    parser.add_argument(
        "--limit", "-l", type=int, default=10, help="Maximum number of results (default: 10, ignored for --glob-only)"
    )
    parser.add_argument(
        "--min-score",
        "-s",
        type=float,
        default=0.0,
        help="Minimum similarity score (0.0-1.0, default: 0.0, ignored for --glob-only)",
    )
    parser.add_argument("--include-deleted", action="store_true", help="Include deleted documents in results")

    # Date filtering
    parser.add_argument("--start-date", type=str, help='Start date filter (YYYY-MM-DD, "today", "1 week ago", etc.)')
    parser.add_argument("--end-date", type=str, help='End date filter (YYYY-MM-DD, "today", "1 week ago", etc.)')

    # Output options (mutually exclusive)
    output_group = parser.add_mutually_exclusive_group()
    output_group.add_argument("--verbose", "-v", action="store_true", help="Show detailed information for each result")
    output_group.add_argument("--brief", "-b", action="store_true", help="Show brief format with scores")
    output_group.add_argument(
        "--document", action="store_true", help="Output as markdown document with full content (LLM-friendly)"
    )
    output_group.add_argument("--json", action="store_true", help="Output results as JSON")
    # Note: files-only is now the default, no flag needed

    # Deletion options
    parser.add_argument("--delete", "-d", action="store_true", help="Delete found documents from the database")
    parser.add_argument("--force", action="store_true", help="Skip confirmation prompts for deletion")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be deleted without actually deleting")

    # Configuration
    parser.add_argument("--config", "-c", type=str, help="Path to configuration file (default: config.yaml)")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    return parser


def main():
    """Main CLI entry point"""
    parser = create_parser()
    args = parser.parse_args()

    # Setup logging
    if args.debug:
        logging.basicConfig(level=logging.DEBUG, force=True)

    try:
        # Determine the query source
        query_text = None

        # Priority: positional argument, then stdin
        if args.query:
            query_text = args.query
        elif not args.example and not args.glob_only and not sys.stdin.isatty():
            # Read from stdin if no other search method specified and stdin has data
            query_text = sys.stdin.read().strip()
            if not query_text:
                print("Error: No query provided via argument or stdin", file=sys.stderr)
                sys.exit(1)

        # Validate that we have a search method
        if not query_text and not args.example and not args.glob_only:
            print("Error: Must provide a search query, example file, or glob pattern", file=sys.stderr)
            parser.print_help()
            sys.exit(1)

        # Initialize CLI
        cli = RAGSearchCLI(config_path=args.config)

        # Perform search based on type
        results = []

        if query_text:
            # Semantic search (default mode)
            results = cli.search_by_query(
                query=query_text,
                limit=args.limit,
                min_score=args.min_score,
                include_deleted=args.include_deleted,
                glob_pattern=args.glob,  # Use glob as filter for semantic search
                start_date=args.start_date,
                end_date=args.end_date,
            )
        elif args.example:
            # Similarity search
            results = cli.search_by_example(
                example_file=args.example,
                limit=args.limit,
                min_score=args.min_score,
                include_deleted=args.include_deleted,
                glob_pattern=args.glob,  # Use glob as filter for similarity search
                start_date=args.start_date,
                end_date=args.end_date,
            )
        elif args.glob_only:
            # Pure glob search
            results = cli.search_by_glob(glob_pattern=args.glob_only, start_date=args.start_date, end_date=args.end_date)

        # Handle deletion if requested
        if args.delete:
            cli.bulk_delete(results, force=args.force, dry_run=args.dry_run)
            return

        # Determine output format
        files_only = True  # Default mode
        verbose = False
        document_mode = False

        if args.verbose:
            files_only = False
            verbose = True
        elif args.brief:
            files_only = False
            verbose = False
        elif args.document:
            files_only = False
            document_mode = True

        # Output results
        if args.json:
            print(json.dumps(results, indent=2, default=str))
        else:
            output = cli.format_results(results, verbose=verbose, files_only=files_only, document_mode=document_mode)
            if output:  # Only print if there's content
                print(output)

    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        if args.debug:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
