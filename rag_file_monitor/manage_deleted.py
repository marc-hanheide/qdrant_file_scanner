#!/usr/bin/env python3
"""
Utility script to manage deleted documents in Qdrant
"""

import yaml
import click
import logging
from pathlib import Path
from .embedding_manager import EmbeddingManager


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file with smart discovery"""
    # If the provided path is just "config.yaml" (default), try smart discovery
    if config_path == "config.yaml":
        config_candidates = [Path.cwd() / "config.yaml", Path(__file__).parent.parent / "config.yaml"]
        actual_config_path = None
        for candidate in config_candidates:
            if candidate.exists():
                actual_config_path = candidate
                break

        if actual_config_path is None:
            raise FileNotFoundError(f"Configuration file not found. Tried: {config_candidates}")

        config_path = str(actual_config_path)

    # Load the configuration file
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


@click.group()
def cli():
    """Manage deleted documents in RAG system"""
    pass


@cli.command()
@click.option("--config", "-c", default="config.yaml", help="Path to config file")
def list_deleted(config):
    """List all documents marked as deleted, check if they actually exist, and restore/reindex if needed"""
    import os
    from .text_extractors import TextExtractor
    
    # Load config
    config_data = load_config(config)

    # Setup logging
    log_level = getattr(logging, config_data.get('logging', {}).get('level', 'INFO').upper())
    logging.basicConfig(level=log_level)

    # Initialize components
    embedding_manager = EmbeddingManager(config_data)
    text_extractor = TextExtractor()

    # Get deleted documents
    deleted_docs = embedding_manager.get_deleted_documents()

    if not deleted_docs:
        click.echo("No deleted documents found.")
        return

    actually_deleted = []
    restored_count = 0
    reindexed_count = 0

    click.echo(f"Found {len(deleted_docs)} documents marked as deleted. Checking actual file status...")
    click.echo()

    for doc in deleted_docs:
        file_path = doc['file_path']
        
        # Check if file actually exists
        if os.path.exists(file_path):
            click.echo(f"File exists: {file_path}")
            
            try:
                # Get current file hash
                current_hash = embedding_manager.get_file_hash(file_path)
                
                # Check if file has changed by comparing with stored hash
                if not embedding_manager.is_file_unchanged(file_path, current_hash):
                    click.echo(f"  File has changed - reindexing...")
                    
                    # Extract text and reindex
                    text_content = text_extractor.extract_text(file_path)
                    embedding_manager.index_document(file_path, text_content, current_hash)
                    
                    click.echo(f"  Successfully reindexed: {file_path}")
                    reindexed_count += 1
                else:
                    # File unchanged, just restore it (remove deleted flag)
                    click.echo(f"  File unchanged - restoring...")
                    text_content = text_extractor.extract_text(file_path)
                    embedding_manager.index_document(file_path, text_content, current_hash)
                    
                    click.echo(f"  Successfully restored: {file_path}")
                    restored_count += 1
                    
            except Exception as e:
                click.echo(f"  Error processing file: {str(e)}")
                actually_deleted.append(doc)
        else:
            # File is actually deleted
            actually_deleted.append(doc)

    click.echo()
    click.echo(f"Summary:")
    click.echo(f"  Files restored (unchanged): {restored_count}")
    click.echo(f"  Files reindexed (changed): {reindexed_count}")
    click.echo(f"  Files actually deleted: {len(actually_deleted)}")

    if actually_deleted:
        click.echo()
        click.echo("Actually deleted files:")
        for doc in actually_deleted:
            click.echo(f"  File: {doc['file_path']}")
            click.echo(f"    Chunks: {doc['chunk_count']}")
            click.echo(f"    Originally indexed: {doc['original_timestamp']}")
            click.echo(f"    Marked deleted: {doc['deletion_timestamp']}")
            click.echo()


@cli.command()
@click.option("--config", "-c", default="config.yaml", help="Path to config file")
@click.argument("file_path")
def restore_deleted(config, file_path):
    """Mark a deleted document as active (if file exists again)"""
    import os
    from .text_extractors import TextExtractor

    # Check if file exists
    if not os.path.exists(file_path):
        click.echo(f"Error: File {file_path} does not exist. Cannot restore.")
        return

    # Load config
    config_data = load_config(config)

    # Setup logging
    log_level = getattr(logging, config_data.get('logging', {}).get('level', 'INFO').upper())
    logging.basicConfig(level=log_level)

    # Initialize components
    embedding_manager = EmbeddingManager(config_data)
    text_extractor = TextExtractor()

    click.echo(f"Restoring file: {file_path}")

    try:
        # Extract text and get file hash
        text_content = text_extractor.extract_text(file_path)
        file_hash = embedding_manager.get_file_hash(file_path)

        # Reindex the document (this will overwrite any deleted version)
        embedding_manager.index_document(file_path, text_content, file_hash)

        click.echo("File successfully restored!")

    except Exception as e:
        click.echo(f"Error restoring file: {str(e)}")


@cli.command()
@click.option("--config", "-c", default="config.yaml", help="Path to config file")
@click.argument("file_path")
@click.option("--force", is_flag=True, help="Force deletion without confirmation")
def purge_deleted(config, file_path, force):
    """Permanently delete a document marked as deleted (or restore/reindex if file still exists)"""
    import os
    from .text_extractors import TextExtractor
    
    # Check if file exists
    if os.path.exists(file_path):
        # File exists - check if it needs to be restored/reindexed instead
        # Load config first
        config_data = load_config(config)

        # Setup logging
        log_level = getattr(logging, config_data.get('logging', {}).get('level', 'INFO').upper())
        logging.basicConfig(level=log_level)

        # Initialize components
        embedding_manager = EmbeddingManager(config_data)
        text_extractor = TextExtractor()
        
        click.echo(f"File {file_path} exists. Checking if it needs restoration instead of deletion...")
        
        try:
            current_hash = embedding_manager.get_file_hash(file_path)
            
            if not embedding_manager.is_file_unchanged(file_path, current_hash):
                click.echo(f"File has changed - reindexing instead of deleting...")
                text_content = text_extractor.extract_text(file_path)
                embedding_manager.index_document(file_path, text_content, current_hash)
                click.echo("File successfully reindexed!")
                return
            else:
                click.echo(f"File unchanged - restoring instead of deleting...")
                text_content = text_extractor.extract_text(file_path)
                embedding_manager.index_document(file_path, text_content, current_hash)
                click.echo("File successfully restored!")
                return
                
        except Exception as e:
            click.echo(f"Error processing file: {str(e)}")
            if not force:
                click.confirm(f"File exists but cannot be processed. Continue with deletion?", abort=True)

    # Load config
    config_data = load_config(config)

    # Setup logging
    log_level = getattr(logging, config_data.get('logging', {}).get('level', 'INFO').upper())
    logging.basicConfig(level=log_level)

    # Initialize embedding manager
    embedding_manager = EmbeddingManager(config_data)

    if not force:
        click.confirm(f"Are you sure you want to permanently delete all chunks for {file_path}?", abort=True)

    # Delete the document
    embedding_manager.delete_document(file_path, force_delete=True)
    click.echo(f"Permanently deleted: {file_path}")


@cli.command()
@click.option("--config", "-c", default="config.yaml", help="Path to config file")
@click.option("--older-than-days", type=int, help="Delete documents older than N days")
@click.option("--force", is_flag=True, help="Force deletion without confirmation")
def cleanup_deleted(config, older_than_days, force):
    """Clean up old deleted documents after checking if they actually exist and restoring/reindexing as needed"""
    import os
    from datetime import datetime, timedelta
    from .text_extractors import TextExtractor

    # Load config
    config_data = load_config(config)

    # Setup logging
    log_level = getattr(logging, config_data.get('logging', {}).get('level', 'INFO').upper())
    logging.basicConfig(level=log_level)

    # Initialize components
    embedding_manager = EmbeddingManager(config_data)
    text_extractor = TextExtractor()

    # Get deleted documents
    deleted_docs = embedding_manager.get_deleted_documents()

    if not deleted_docs:
        click.echo("No deleted documents found.")
        return

    # First, check which files are actually deleted vs still exist
    actually_deleted = []
    restored_count = 0
    reindexed_count = 0

    click.echo(f"Found {len(deleted_docs)} documents marked as deleted. Verifying actual file status...")

    for doc in deleted_docs:
        file_path = doc['file_path']
        
        if os.path.exists(file_path):
            try:
                # File exists - check if it has changed
                current_hash = embedding_manager.get_file_hash(file_path)
                
                if not embedding_manager.is_file_unchanged(file_path, current_hash):
                    # File has changed - reindex it
                    text_content = text_extractor.extract_text(file_path)
                    embedding_manager.index_document(file_path, text_content, current_hash)
                    reindexed_count += 1
                    click.echo(f"Reindexed changed file: {file_path}")
                else:
                    # File unchanged - restore it
                    text_content = text_extractor.extract_text(file_path)
                    embedding_manager.index_document(file_path, text_content, current_hash)
                    restored_count += 1
                    click.echo(f"Restored unchanged file: {file_path}")
                    
            except Exception as e:
                click.echo(f"Error processing {file_path}: {str(e)}")
                actually_deleted.append(doc)
        else:
            # File is actually deleted
            actually_deleted.append(doc)

    if restored_count > 0 or reindexed_count > 0:
        click.echo(f"Restored {restored_count} files and reindexed {reindexed_count} changed files.")

    # Filter by age if specified for actually deleted files
    if older_than_days:
        cutoff_date = datetime.now() - timedelta(days=older_than_days)
        docs_to_delete = []

        for doc in actually_deleted:
            if doc["deletion_timestamp"]:
                deletion_date = datetime.fromisoformat(doc["deletion_timestamp"].replace("Z", "+00:00"))
                if deletion_date < cutoff_date:
                    docs_to_delete.append(doc)

        click.echo(f"Found {len(docs_to_delete)} actually deleted documents older than {older_than_days} days.")
    else:
        docs_to_delete = actually_deleted
        click.echo(f"Found {len(docs_to_delete)} actually deleted documents.")

    if not docs_to_delete:
        return

    if not force:
        click.echo("Documents to be permanently deleted:")
        for doc in docs_to_delete:
            click.echo(f"  - {doc['file_path']} (deleted: {doc['deletion_timestamp']})")
        click.echo()
        click.confirm("Are you sure you want to permanently delete these documents?", abort=True)

    # Delete documents
    for doc in docs_to_delete:
        embedding_manager.delete_document(doc["file_path"], force_delete=True)
        click.echo(f"Deleted: {doc['file_path']}")

    click.echo(f"Cleanup complete. Deleted {len(docs_to_delete)} documents.")


@cli.command()
@click.option("--config", "-c", default="config.yaml", help="Path to config file")
@click.option("--dry-run", is_flag=True, help="Show what would be done without making changes")
def check_and_fix_deleted(config, dry_run):
    """Check all documents marked as deleted and restore/reindex those that still exist"""
    import os
    from .text_extractors import TextExtractor
    
    # Load config
    config_data = load_config(config)

    # Setup logging
    log_level = getattr(logging, config_data.get('logging', {}).get('level', 'INFO').upper())
    logging.basicConfig(level=log_level)

    # Initialize components
    embedding_manager = EmbeddingManager(config_data)
    text_extractor = TextExtractor()

    # Get deleted documents
    deleted_docs = embedding_manager.get_deleted_documents()

    if not deleted_docs:
        click.echo("No deleted documents found.")
        return

    actually_deleted = []
    to_restore = []
    to_reindex = []

    click.echo(f"Found {len(deleted_docs)} documents marked as deleted. Checking actual file status...")

    for doc in deleted_docs:
        file_path = doc['file_path']
        
        if os.path.exists(file_path):
            try:
                current_hash = embedding_manager.get_file_hash(file_path)
                
                if not embedding_manager.is_file_unchanged(file_path, current_hash):
                    to_reindex.append((file_path, current_hash))
                else:
                    to_restore.append((file_path, current_hash))
                    
            except Exception as e:
                click.echo(f"Error processing {file_path}: {str(e)}")
                actually_deleted.append(doc)
        else:
            actually_deleted.append(doc)

    # Show summary
    click.echo()
    click.echo("Summary:")
    click.echo(f"  Files to restore (unchanged): {len(to_restore)}")
    click.echo(f"  Files to reindex (changed): {len(to_reindex)}")
    click.echo(f"  Files actually deleted: {len(actually_deleted)}")

    if dry_run:
        click.echo()
        click.echo("DRY RUN - No changes will be made")
        
        if to_restore:
            click.echo("Files that would be restored:")
            for file_path, _ in to_restore:
                click.echo(f"  - {file_path}")
                
        if to_reindex:
            click.echo("Files that would be reindexed:")
            for file_path, _ in to_reindex:
                click.echo(f"  - {file_path}")
        return

    # Perform the fixes
    restored_count = 0
    reindexed_count = 0
    
    for file_path, current_hash in to_restore:
        try:
            text_content = text_extractor.extract_text(file_path)
            embedding_manager.index_document(file_path, text_content, current_hash)
            restored_count += 1
            click.echo(f"Restored: {file_path}")
        except Exception as e:
            click.echo(f"Error restoring {file_path}: {str(e)}")

    for file_path, current_hash in to_reindex:
        try:
            text_content = text_extractor.extract_text(file_path)
            embedding_manager.index_document(file_path, text_content, current_hash)
            reindexed_count += 1
            click.echo(f"Reindexed: {file_path}")
        except Exception as e:
            click.echo(f"Error reindexing {file_path}: {str(e)}")

    click.echo()
    click.echo(f"Complete! Restored {restored_count} files and reindexed {reindexed_count} files.")


if __name__ == "__main__":
    cli()
