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
    """List all documents marked as deleted"""
    # Load config
    config_data = load_config(config)

    # Setup logging
    log_level = getattr(logging, config_data.get('logging', {}).get('level', 'INFO').upper())
    logging.basicConfig(level=log_level)

    # Initialize embedding manager
    embedding_manager = EmbeddingManager(config_data)

    # Get deleted documents
    deleted_docs = embedding_manager.get_deleted_documents()

    if not deleted_docs:
        click.echo("No deleted documents found.")
        return

    click.echo(f"Found {len(deleted_docs)} deleted documents:")
    click.echo()

    for doc in deleted_docs:
        click.echo(f"File: {doc['file_path']}")
        click.echo(f"  Chunks: {doc['chunk_count']}")
        click.echo(f"  Originally indexed: {doc['original_timestamp']}")
        click.echo(f"  Marked deleted: {doc['deletion_timestamp']}")
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
    """Permanently delete a document marked as deleted"""
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
    """Clean up old deleted documents"""
    from datetime import datetime, timedelta

    # Load config
    config_data = load_config(config)

    # Setup logging
    log_level = getattr(logging, config_data.get('logging', {}).get('level', 'INFO').upper())
    logging.basicConfig(level=log_level)

    # Initialize embedding manager
    embedding_manager = EmbeddingManager(config_data)

    # Get deleted documents
    deleted_docs = embedding_manager.get_deleted_documents()

    if not deleted_docs:
        click.echo("No deleted documents found.")
        return

    # Filter by age if specified
    if older_than_days:
        cutoff_date = datetime.now() - timedelta(days=older_than_days)
        docs_to_delete = []

        for doc in deleted_docs:
            if doc["deletion_timestamp"]:
                deletion_date = datetime.fromisoformat(doc["deletion_timestamp"].replace("Z", "+00:00"))
                if deletion_date < cutoff_date:
                    docs_to_delete.append(doc)

        click.echo(f"Found {len(docs_to_delete)} documents deleted more than {older_than_days} days ago.")
    else:
        docs_to_delete = deleted_docs
        click.echo(f"Found {len(docs_to_delete)} deleted documents.")

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


if __name__ == "__main__":
    cli()
