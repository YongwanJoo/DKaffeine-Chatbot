"""Token counting and truncation utilities."""
import logging
from typing import List, Tuple, Any

import tiktoken

logger = logging.getLogger(__name__)

# Global tokenizer instance (lazy loaded)
_tokenizer = None

def get_tokenizer():
    """Get or create the shared tokenizer instance (cl100k_base)."""
    global _tokenizer
    if _tokenizer is None:
        try:
            _tokenizer = tiktoken.get_encoding("cl100k_base")
        except Exception as e:
            logger.error(f"Failed to load tiktoken encoding: {e}")
            # Fallback to a simple character-based approximation if tiktoken fails
            # This is just to prevent crashes, though accuracy will be poor
            return None
    return _tokenizer

def count_tokens(text: str) -> int:
    """Count tokens in a text string.
    
    Args:
        text: Input text
        
    Returns:
        Number of tokens
    """
    if not text:
        return 0
        
    tokenizer = get_tokenizer()
    if tokenizer:
        try:
            return len(tokenizer.encode(text))
        except Exception as e:
            logger.warning(f"Token counting failed: {e}")
            return len(text) // 4  # Rough approximation
    else:
        return len(text) // 4

def truncate_text(text: str, max_tokens: int) -> str:
    """Truncate text to fit within max_tokens.
    
    Args:
        text: Input text
        max_tokens: Maximum allowed tokens
        
    Returns:
        Truncated text
    """
    if not text or max_tokens <= 0:
        return ""
        
    current_tokens = count_tokens(text)
    if current_tokens <= max_tokens:
        return text
        
    tokenizer = get_tokenizer()
    if tokenizer:
        try:
            tokens = tokenizer.encode(text)
            truncated_tokens = tokens[:max_tokens]
            return tokenizer.decode(truncated_tokens)
        except Exception as e:
            logger.warning(f"Token truncation failed: {e}")
            # Fallback: character truncation
            return text[:max_tokens * 4]
    else:
        return text[:max_tokens * 4]

def truncate_documents(
    docs: List[Tuple[Any, float, int]], 
    max_tokens: int,
    overhead_per_doc: int = 50
) -> List[Tuple[Any, float, int]]:
    """Truncate a list of documents to fit within max_tokens.
    
    Preserves the order of documents (assuming they are sorted by relevance).
    If a document cannot fit fully, it might be partially included or skipped
    depending on the remaining space.
    
    Args:
        docs: List of (doc, score, doc_number) tuples
        max_tokens: Maximum allowed tokens for all documents combined
        overhead_per_doc: Estimated tokens for formatting (e.g. "[Document 1] ... Source: ...")
        
    Returns:
        List of (doc, score, doc_number) tuples that fit
    """
    if not docs or max_tokens <= 0:
        return []
        
    current_tokens = 0
    truncated_docs = []
    
    for doc, score, doc_number in docs:
        content = doc.page_content
        source = doc.metadata.get('source', 'unknown')
        
        # Calculate tokens for this document including overhead
        # Overhead accounts for: f"[문서 {doc_number}] {content}\n출처: {source}"
        # We count content tokens exactly, and add a buffer for the wrapper
        content_tokens = count_tokens(content)
        source_tokens = count_tokens(source)
        doc_total_tokens = content_tokens + source_tokens + overhead_per_doc
        
        if current_tokens + doc_total_tokens <= max_tokens:
            # Fits entirely
            truncated_docs.append((doc, score, doc_number))
            current_tokens += doc_total_tokens
        else:
            # Doesn't fit entirely. Check if we can fit a partial chunk.
            remaining_tokens = max_tokens - current_tokens - source_tokens - overhead_per_doc
            
            # If we can fit at least a meaningful amount (e.g. 50 tokens), truncate and add
            if remaining_tokens > 50:
                truncated_content = truncate_text(content, remaining_tokens)
                # Create a copy of the doc with truncated content to avoid mutating original
                # Note: LangChain Document objects might be mutable or not, safer to copy
                from langchain_core.documents import Document
                new_doc = Document(
                    page_content=truncated_content,
                    metadata=doc.metadata
                )
                truncated_docs.append((new_doc, score, doc_number))
                current_tokens += remaining_tokens + source_tokens + overhead_per_doc
                
            # Stop processing further documents as we are full
            break
            
    return truncated_docs
