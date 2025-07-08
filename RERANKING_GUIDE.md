# Re-ranking for Improved Search Results

## Overview

The RAG system now supports **re-ranking** to significantly improve search result quality. Re-ranking uses a two-stage approach:

1. **Initial Retrieval**: Uses your existing embedding model (`all-MiniLM-L6-v2`) to quickly find potentially relevant documents
2. **Re-ranking**: Uses a cross-encoder model to provide more accurate relevance scoring by understanding the relationship between query and document content

## Benefits

- **Better Precision**: More relevant results ranked higher
- **Improved Context Understanding**: Cross-encoders can better understand query-document relationships
- **Configurable**: Can be enabled/disabled and tuned per your needs
- **Minimal Performance Impact**: Only re-ranks the top candidates, not all documents

## Configuration

Re-ranking is configured in `config.yaml`:

```yaml
reranker:
  enabled: true  # Set to false to disable
  model_name: "cross-encoder/ms-marco-MiniLM-L-6-v2"  # Cross-encoder model
  top_k_retrieve: 50  # Retrieve this many candidates before re-ranking
  score_threshold: 0.0  # Minimum re-ranking score (0.0 = no threshold)
  unload_after_idle_minutes: 15  # Unload model after idle time
```

## Quick Setup

### Enable Re-ranking

```bash
# Enable with default settings
python enable_reranker.py enable

# Enable with custom model
python enable_reranker.py enable --model "cross-encoder/ms-marco-TinyBERT-L-2-v2"
```

### Disable Re-ranking

```bash
python enable_reranker.py disable
```

## Model Options

### Recommended Models

1. **cross-encoder/ms-marco-MiniLM-L-6-v2** (Default)
   - Good balance of quality and speed
   - ~80MB download
   - Works well for general text

2. **cross-encoder/ms-marco-TinyBERT-L-2-v2** (Faster)
   - Faster inference
   - ~50MB download
   - Slightly lower quality but much faster

3. **cross-encoder/ms-marco-electra-base** (Higher Quality)
   - Better quality for complex queries
   - ~400MB download
   - Slower inference

## How It Works

### Without Re-ranking
```
Query → Embedding → Vector Search → Top-K Results
```

### With Re-ranking
```
Query → Embedding → Vector Search → Top-N Candidates → Cross-Encoder → Re-ranked Top-K Results
```

### Example Flow

1. User searches for "budget analysis quarterly report"
2. System retrieves 50 candidate documents using embedding similarity
3. Re-ranker scores each candidate against the query
4. Returns top 10 results ordered by re-ranking score

## Performance Considerations

### Memory Usage
- Re-ranker models are loaded on-demand
- Models are automatically unloaded after idle time
- Memory usage: ~100-500MB depending on model

### Latency
- Initial retrieval: ~same as before
- Re-ranking: adds ~50-200ms depending on candidate count and model
- Overall: typically 2-3x slower but much better results

### Optimization Tips

1. **Adjust `top_k_retrieve`**: Lower values = faster, higher values = better recall
2. **Use score thresholds**: Filter out low-quality results
3. **Choose appropriate model**: Balance quality vs speed for your use case

## Usage Examples

### MCP Server
The MCP server automatically uses re-ranking when enabled:

```python
# Returns re-ranked results if enabled
result = rag_search("machine learning optimization", number_docs=10)

# Results now include rerank_score and original_score
for item in result.results:
    print(f"File: {item.file_path}")
    print(f"Original score: {item.original_score}")
    print(f"Re-rank score: {item.rerank_score}")
```

### CLI Search
The CLI shows both scores when re-ranking is enabled:

```bash
# Verbose output shows both scores
python -m rag_file_monitor.search_cli "quarterly budget" --verbose

# Output:
# 1. /path/to/budget_q4.pdf
#    Score: 0.742 (re-ranked: 0.891)
#    Chunk: 0
#    Preview: This quarterly budget analysis shows...
```

## Monitoring

### Memory Stats
The system tracks re-ranker memory usage:

```python
# Via MCP server resource
stats = get_database_stats()
# Includes re-ranker model status

# Or via embedding manager
memory_stats = embedding_manager.get_memory_stats()
# Shows if re-ranker model is loaded
```

### Logs
Re-ranking activity is logged:

```
INFO - Loading cross-encoder model: cross-encoder/ms-marco-MiniLM-L-6-v2
INFO - Re-ranked 45 -> 10 results (threshold: 0.0)
INFO - Unloading idle cross-encoder after 15 minutes
```

## Troubleshooting

### Common Issues

1. **ImportError: CrossEncoder not found**
   ```bash
   pip install torch sentence-transformers
   ```

2. **Model download fails**
   - Check internet connection
   - Verify model name is correct
   - Check available disk space

3. **High memory usage**
   - Reduce `unload_after_idle_minutes`
   - Use smaller model (TinyBERT variant)
   - Reduce `top_k_retrieve`

4. **Slow performance**
   - Use faster model (TinyBERT)
   - Reduce `top_k_retrieve`
   - Increase `score_threshold`

### Performance Tuning

```yaml
# Fast setup (prioritize speed)
reranker:
  enabled: true
  model_name: "cross-encoder/ms-marco-TinyBERT-L-2-v2"
  top_k_retrieve: 30
  score_threshold: 0.2
  unload_after_idle_minutes: 5

# Quality setup (prioritize accuracy)
reranker:
  enabled: true
  model_name: "cross-encoder/ms-marco-electra-base"
  top_k_retrieve: 100
  score_threshold: 0.0
  unload_after_idle_minutes: 30
```

## Testing

Run the re-ranking tests:

```bash
python -m pytest tests/test_reranker.py -v
```

## Integration Notes

### Existing Code Compatibility
- All existing search functionality remains unchanged
- Re-ranking is transparent to existing clients
- Results include additional score fields when re-ranking is enabled

### Backward Compatibility
- Configuration is fully backward compatible
- Disabling re-ranking reverts to original behavior
- No changes required to existing search queries
