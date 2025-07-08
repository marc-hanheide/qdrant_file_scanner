#!/usr/bin/env python3
"""
Enable or disable the re-ranker in the RAG system
"""

import yaml
from pathlib import Path
import argparse


def update_reranker_config(config_path: str, enable: bool, model_name: str = None):
    """Update the re-ranker configuration"""
    config_file = Path(config_path)

    if not config_file.exists():
        print(f"Configuration file not found: {config_path}")
        return False

    # Load current configuration
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    # Update re-ranker settings
    if "reranker" not in config:
        config["reranker"] = {}

    config["reranker"]["enabled"] = enable

    if model_name:
        config["reranker"]["model_name"] = model_name

    # Set reasonable defaults if enabling for the first time
    if enable:
        config["reranker"].setdefault("top_k_retrieve", 50)
        config["reranker"].setdefault("score_threshold", 0.0)
        config["reranker"].setdefault("unload_after_idle_minutes", 15)
        config["reranker"].setdefault("model_name", "cross-encoder/ms-marco-MiniLM-L-6-v2")

    # Write updated configuration
    with open(config_file, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    status = "enabled" if enable else "disabled"
    print(f"Re-ranker {status} in {config_path}")

    if enable:
        print(f"Re-ranker model: {config['reranker']['model_name']}")
        print(f"Top-k retrieve: {config['reranker']['top_k_retrieve']}")
        print(f"Score threshold: {config['reranker']['score_threshold']}")
        print("\nNote: You may need to restart the MCP server for changes to take effect.")

    return True


def main():
    parser = argparse.ArgumentParser(description="Enable or disable the RAG re-ranker")
    parser.add_argument("action", choices=["enable", "disable"], help="Action to perform")
    parser.add_argument("--config", "-c", default="config.yaml", help="Path to configuration file")
    parser.add_argument("--model", "-m", help="Re-ranker model name (only when enabling)")

    args = parser.parse_args()

    enable = args.action == "enable"
    success = update_reranker_config(args.config, enable, args.model)

    if not success:
        exit(1)


if __name__ == "__main__":
    main()
