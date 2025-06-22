"""Knowledge Retrieval MCP Server with LLM integration for support documentation.

This module implements an MCP server that manages and searches a knowledge base
of support documentation. It provides intelligent search capabilities to find
relevant solutions and procedures for technical issues, and integrates with LLMs
to generate contextual recommendations based on the retrieved knowledge.

The server maintains a collection of markdown-based knowledge resources with
metadata for categorization and relevance scoring. It's designed to support
the support agent by providing quick access to troubleshooting guides,
runbooks, and best practices.
"""


import json
import re
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from mcp.server.fastmcp import FastMCP

# Vector embeddings imports
try:
    import openai
    import os
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    OPENAI_EMBEDDINGS_AVAILABLE = bool(os.getenv("OPENAI_API_KEY"))
    EMBEDDINGS_AVAILABLE = OPENAI_EMBEDDINGS_AVAILABLE
except ImportError:
    OPENAI_EMBEDDINGS_AVAILABLE = False
    EMBEDDINGS_AVAILABLE = False
    print("Warning: openai not available. Install with: pip install openai")

# Fallback to sentence-transformers if OpenAI not available
if not OPENAI_EMBEDDINGS_AVAILABLE:
    try:
        from sentence_transformers import SentenceTransformer
        EMBEDDINGS_AVAILABLE = True
        print("Using sentence-transformers as fallback")
    except ImportError:
        print("Warning: Neither OpenAI nor sentence-transformers available")

# Create FastMCP server instance
mcp = FastMCP("KnowledgeRetrievalServer")

# Initialize resources directory
RESOURCES_DIR = Path("./knowledge_resources")
RESOURCES_DIR.mkdir(parents=True, exist_ok=True)

class VectorKnowledgeBase:
    """Vector-based knowledge base with semantic search capabilities."""
    
    def __init__(self):
        self.embeddings_model = None
        self.section_embeddings = []
        self.sections = []
        self.indexed = False
        self.use_openai = OPENAI_EMBEDDINGS_AVAILABLE
        
        if EMBEDDINGS_AVAILABLE:
            try:
                if self.use_openai:
                    # Use OpenAI's superior semantic embedding model
                    self.embeddings_model = "text-embedding-3-small"  # Model identifier
                    print("Using OpenAI text-embedding-3-small for superior semantic search")
                else:
                    # Fallback to sentence-transformers
                    import logging
                    logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
                    self.embeddings_model = SentenceTransformer('all-MiniLM-L6-v2')
                    print("Using sentence-transformers fallback")
            except Exception as e:
                print(f"Failed to load embeddings model: {e}")
                self.embeddings_model = None
    
    def _generate_embedding(self, text: str):
        """Generate embedding using either OpenAI or sentence-transformers."""
        if self.use_openai and isinstance(self.embeddings_model, str):
            # Use OpenAI API
            client = openai.OpenAI()
            response = client.embeddings.create(
                model=self.embeddings_model,
                input=text
            )
            return np.array(response.data[0].embedding)
        else:
            # Use sentence-transformers
            return self.embeddings_model.encode(text)
    
    async def _analyze_query_intent(self, query: str) -> Dict[str, str]:
        """Analyze query to understand intent using LLM reasoning without hardcoded categories."""
        if not self.use_openai:
            return {"reasoning": "LLM not available", "semantic_focus": ""}
        
        try:
            client = openai.OpenAI()
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{
                    "role": "system", 
                    "content": """You are an expert at understanding technical support queries. Analyze the query and provide semantic understanding to help with document retrieval.

Provide:
1. A semantic summary of what the user is asking about
2. Key concepts that should be prioritized in document search
3. Any concepts that should be deprioritized or avoided

Do NOT use predefined categories. Focus on understanding the user's actual problem.

Respond with JSON: {
  "semantic_focus": "brief description of what user needs help with",
  "key_concepts": ["concept1", "concept2", "concept3"],
  "avoid_concepts": ["concept1", "concept2"],
  "reasoning": "explanation of the user's actual problem"
}"""
                },
                {
                    "role": "user",
                    "content": f"Query: {query}"
                }],
                temperature=0.1,
                max_tokens=300
            )
            
            import json
            result = json.loads(response.choices[0].message.content)
            return result
            
        except Exception as e:
            print(f"LLM query analysis failed: {e}")
            return {"semantic_focus": "", "key_concepts": [], "avoid_concepts": [], "reasoning": f"LLM analysis failed: {e}"}

    def cleanup(self):
        """Clean up model resources to prevent semaphore leaks."""
        if self.embeddings_model is not None and not self.use_openai:
            try:
                # Clean up any torch multiprocessing contexts
                import torch
                if hasattr(torch.multiprocessing, 'set_sharing_strategy'):
                    torch.multiprocessing.set_sharing_strategy('file_system')
                
                # Try to clean up model internals
                if hasattr(self.embeddings_model, '_modules'):
                    for module in self.embeddings_model._modules.values():
                        if hasattr(module, 'pool'):
                            try:
                                module.pool.close()
                                module.pool.join()
                            except:
                                pass
                
                # Clear model from memory
                del self.embeddings_model
            except Exception:
                # Ignore cleanup errors, just clear the reference
                pass
            finally:
                self.embeddings_model = None
                
        # Clean up embeddings data
        if hasattr(self.section_embeddings, 'clear'):
            self.section_embeddings.clear()
        else:
            self.section_embeddings = []
        self.sections.clear()
    
    def parse_markdown_sections(self, content: str, doc_name: str) -> List[Tuple[str, str, str]]:
        """Parse markdown content into sections.
        
        Args:
            content: Full markdown content
            doc_name: Name of the source document
            
        Returns:
            List of tuples: (section_title, section_content, hierarchical_path)
        """
        lines = content.split('\n')
        sections = []
        current_section = {"title": doc_name, "content": [], "level": 0, "path": [doc_name]}
        section_stack = [current_section]
        
        for line in lines:
            # Check if line is a header
            header_match = re.match(r'^(#{1,6})\s+(.+)$', line)
            if header_match:
                # Save current section if it has content
                if current_section["content"]:
                    full_title = " > ".join(current_section["path"])
                    content_text = '\n'.join(current_section["content"]).strip()
                    if content_text:  # Only add non-empty sections
                        sections.append((full_title, content_text, '/'.join(current_section["path"])))
                
                # Process new header
                level = len(header_match.group(1))
                title = header_match.group(2).strip()
                
                # Update section stack to maintain hierarchy
                while len(section_stack) > level:
                    section_stack.pop()
                
                # Create new section
                current_path = section_stack[-1]["path"][:] if section_stack else [doc_name]
                if level > len(section_stack):
                    current_path.append(title)
                else:
                    current_path = current_path[:level-1] + [title]
                
                current_section = {
                    "title": title,
                    "content": [],
                    "level": level,
                    "path": current_path
                }
                section_stack.append(current_section)
            else:
                # Add content to current section
                current_section["content"].append(line)
        
        # Don't forget the last section
        if current_section["content"]:
            full_title = " > ".join(current_section["path"])
            content_text = '\n'.join(current_section["content"]).strip()
            if content_text:
                sections.append((full_title, content_text, '/'.join(current_section["path"])))
        
        return sections
    
    async def index_documents(self):
        """Pre-compute embeddings for all knowledge document sections."""
        if not self.embeddings_model:
            return False
            
        try:
            self.sections = []
            self.section_embeddings = []
            
            # Load all markdown documents
            for file_path in RESOURCES_DIR.glob("*.md"):
                # Read document content
                with open(file_path, 'r') as f:
                    content = f.read()
                
                # Parse into sections using filename as base
                doc_name = file_path.stem.replace("_", " ").title()
                sections = self.parse_markdown_sections(content, doc_name)
                
                # Create embeddings for each section
                for section_title, section_content, section_path in sections:
                    # Create section record
                    section = {
                        "uri": f"knowledge://{file_path.name}#{section_path}",
                        "doc_name": file_path.stem,
                        "section_title": section_title,
                        "content": section_content,
                        "section_path": section_path
                    }
                    
                    # Generate embedding for content only (no filename contamination)
                    # Remove document name from section title for embedding
                    clean_section_title = section_title.split(" > ", 1)[-1] if " > " in section_title else section_title
                    if clean_section_title == doc_name:  # If it's just the document name, use content only
                        text_to_embed = section_content[:1000]
                    else:
                        text_to_embed = f"{clean_section_title} {section_content[:1000]}"
                    embedding = self._generate_embedding(text_to_embed)
                    
                    self.sections.append(section)
                    self.section_embeddings.append(embedding)
            
            self.section_embeddings = np.array(self.section_embeddings)
            self.indexed = True
            # Don't print indexing message to keep demo clean
            return True
            
        except Exception as e:
            print(f"Error indexing documents: {e}")
            return False
    
    async def semantic_search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Find most semantically similar document sections with query understanding."""
        if not self.indexed or not self.embeddings_model:
            return []
        
        try:
            # Solution #5: Analyze query intent first using LLM
            query_intent = await self._analyze_query_intent(query)
            
            # Generate query embedding with better model
            query_embedding = self._generate_embedding(query)
            if query_embedding.ndim == 1:
                query_embedding = query_embedding.reshape(1, -1)
            
            # Calculate cosine similarity with all sections
            similarities = cosine_similarity(query_embedding, self.section_embeddings)[0]
            
            # Get top-k most similar sections
            top_indices = np.argsort(similarities)[::-1][:top_k]
            results = []
            
            # Apply LLM-based semantic filtering
            key_concepts = query_intent.get("key_concepts", [])
            avoid_concepts = query_intent.get("avoid_concepts", [])
            semantic_focus = query_intent.get("semantic_focus", "")
            
            for idx in top_indices:
                similarity_score = float(similarities[idx])
                
                # Only include results above relevance threshold
                if similarity_score > 0.2:
                    section = self.sections[idx].copy()
                    section_text = f"{section['section_title']} {section['content']}".lower()
                    
                    # Generic concept-based boosting/penalizing
                    concept_adjustment = 0.0
                    
                    # Boost if key concepts are present
                    for concept in key_concepts:
                        if concept.lower() in section_text:
                            concept_adjustment += 0.05  # Small boost per matching key concept
                    
                    # Penalize if avoid concepts are present
                    for concept in avoid_concepts:
                        if concept.lower() in section_text:
                            concept_adjustment -= 0.1  # Penalty for concepts to avoid
                    
                    # Apply semantic adjustments
                    adjusted_score = similarity_score + concept_adjustment
                    
                    # Create result in expected format
                    result = {
                        "content": section["content"],
                        "source": section["section_title"],
                        "relevance_score": max(adjusted_score, 0.0),  # Don't go negative
                        "metadata": {
                            "uri": section["uri"],
                            "title": section["section_title"],
                            "mime_type": "text/markdown",
                            "section_path": section["section_path"],
                            "semantic_focus": semantic_focus,
                            "concept_adjustment": concept_adjustment
                        }
                    }
                    results.append(result)
            
            # Re-sort by adjusted relevance scores
            results.sort(key=lambda x: x["relevance_score"], reverse=True)
            
            return results
            
        except Exception as e:
            print(f"Error in semantic search: {e}")
            return []

# Global vector knowledge base instance
vector_kb = VectorKnowledgeBase()

def _create_default_resources():
    """Create default knowledge resources for financial services support.
    
    Initializes the knowledge base with essential documentation covering
    common support scenarios in financial services. Each resource includes:
    - Structured troubleshooting procedures
    - SQL queries and code snippets for investigation
    - Step-by-step resolution processes
    - Quality assurance and documentation requirements
    
    Resources are stored as simple markdown files that are automatically
    parsed into sections for granular search capabilities.
    """
    
    resources = [
        {
            "filename": "data_reconciliation.md",
            "content": """# Data Reconciliation Issue Resolution

## Overview
This runbook covers procedures for investigating and resolving data reconciliation discrepancies.

## Common Reconciliation Issues

### 1. Position Report Discrepancies
**Symptoms:**
- Position totals don't match between systems
- Missing trades in position calculations
- Incorrect mark-to-market valuations

**Investigation Steps:**
1. **Identify Scope:**
   - Which positions are affected?
   - What time period shows discrepancies?
   - Which systems are involved?

2. **Data Source Analysis:**
   ```sql
   -- Check trade booking completeness
   SELECT booking_date, COUNT(*) as trade_count 
   FROM trades 
   WHERE booking_date = CURRENT_DATE
   GROUP BY booking_date;
   
   -- Verify position calculations
   SELECT portfolio, SUM(quantity * price) as total_value
   FROM positions 
   WHERE as_of_date = CURRENT_DATE
   GROUP BY portfolio;
   ```

3. **Reconciliation Validation:**
   - Run data validation rules manually
   - Compare record counts between systems
   - Check for data type mismatches or nulls
   - Verify timestamp consistency across systems

### 2. ETL Pipeline Data Issues
**Symptoms:**
- Data not appearing in target systems
- Transformation errors in processing
- Data quality validation failures

**Resolution Process:**
1. **Pipeline Health Check:**
   - Check ETL job execution logs
   - Verify data transformation logic
   - Validate mapping rules and business logic
   - Review data lineage and dependencies

2. **Data Quality Validation:**
   ```bash
   # Check file arrival and processing
   ls -la /data/incoming/{YYYYMMDD}/
   
   # Validate data format
   head -n 10 /data/incoming/trades_20240101.csv
   
   # Check processing status
   SELECT job_name, status, start_time, end_time 
   FROM etl_job_history 
   WHERE run_date = CURRENT_DATE;
   ```

### 3. Regulatory Reporting Discrepancies
**Critical Actions:**
- Identify affected regulatory reports
- Calculate potential exposure and impact
- Coordinate with Compliance team immediately
- Document all investigation steps

## Data Correction Procedures

### 1. Trade Data Corrections
- Verify authorization for data changes
- Document business justification
- Apply corrections in designated maintenance window
- Validate corrections across all downstream systems

### 2. Reprocessing Requirements
- Identify data dependencies and downstream impacts
- Schedule reprocessing during low-activity periods
- Validate data consistency after reprocessing
- Update reconciliation exceptions if valid

## Quality Assurance
- Run full reconciliation after corrections
- Compare before/after data states
- Validate with business users
- Update monitoring to detect similar issues

## Documentation Requirements
- Record all investigation steps and findings
- Document root cause analysis
- Update data lineage documentation
- Create preventive measures for future occurrences
"""
        }
    ]
    
    # Write resources to files
    for resource in resources:
        file_path = RESOURCES_DIR / resource["filename"]
        if not file_path.exists():
            with open(file_path, 'w') as f:
                f.write(resource["content"])

# Initialize resources and vector index on startup
_create_default_resources()

async def _initialize_vector_index():
    """Initialize the vector knowledge base."""
    await vector_kb.index_documents()

# Note: The index will be initialized when the first search is performed

@mcp.tool()
async def search_knowledge(query: str, category: Optional[str] = None, max_results: int = 5) -> List[Dict]:
    """Search knowledge base using vector embeddings for semantic similarity.
    
    Performs semantic search across the knowledge base using pre-computed vector
    embeddings and cosine similarity. Falls back to basic search if embeddings
    are not available.
    
    Args:
        query: Search query describing the issue or topic.
        category: Optional category filter (currently unused in vector search).
        max_results: Maximum number of results to return (default: 5).
    
    Returns:
        List[Dict]: Ranked list of knowledge resources containing:
                   - content: Full text of the knowledge article
                   - source: Resource identifier and title
                   - relevance_score: Cosine similarity score between 0.0 and 1.0
                   - metadata: Additional resource information
    """
    # Initialize vector index if not already done
    if not vector_kb.indexed:
        await _initialize_vector_index()
    
    # Use vector search (required)
    if vector_kb.indexed and EMBEDDINGS_AVAILABLE:
        results = await vector_kb.semantic_search(query, max_results)
        return results
    
    # No fallback - vector embeddings are required
    print("ERROR: Vector embeddings not available - system requires vector search capabilities")
    return []



@mcp.tool()
def prepare_analysis_prompt(query: str, knowledge_content: str, affected_system: Optional[str] = None) -> str:
    """Prepare a focused prompt for LLM analysis using retrieved knowledge.
    
    Constructs a structured prompt that combines the user's issue description
    with relevant knowledge base content to generate specific, actionable
    recommendations. The prompt is designed to produce concise, targeted
    responses focused on immediate resolution steps.
    
    Args:
        query: Original user issue description.
        knowledge_content: Relevant knowledge base content for context.
        affected_system: Optional system identifier for additional context.
    
    Returns:
        str: Formatted prompt ready for LLM processing, structured to elicit
             immediate actions, expected results, and escalation procedures.
    """
    
    return f"""You are a production support specialist. Analyze this specific issue and provide targeted recommendations.

USER ISSUE: {query}
AFFECTED SYSTEM: {affected_system or 'Not specified'}

KNOWLEDGE BASE CONTENT:
{knowledge_content}

CRITICAL INSTRUCTIONS:
1. ONLY use commands, code, or procedures that are EXPLICITLY shown in the knowledge base content above
2. DO NOT invent, create, or suggest any code that is not directly copied from the knowledge base
3. If the knowledge base mentions a concept without providing the exact code, acknowledge this limitation
4. When referencing code, copy it EXACTLY as shown in the knowledge base
5. If additional information is provided from follow-up searches, incorporate that knowledge into your response
6. CONTEXT AWARENESS: Carefully read what the user has already stated or verified. Do NOT recommend steps that contradict or duplicate what the user has explicitly confirmed
7. SKIP REDUNDANT STEPS: If the user has already stated that something is working correctly (e.g., "book2 resolved to 'MarkitWire'"), do not suggest verifying that same thing again
8. CONTEXTUALIZE EXAMPLES: Adapt code examples and procedures to the user's specific situation. Replace generic parameters and placeholder values with the actual systems, services, or identifiers mentioned in the user's request

Based on the user's specific issue and the knowledge base content, provide a concise response with:

## Immediate Actions
- List specific steps using ONLY information from the knowledge base
- Include ONLY the exact commands/code provided in the knowledge base
- SKIP any steps the user has already confirmed or stated are working correctly
- If a step is mentioned but no code is provided, simply describe the step without adding unhelpful messages

## Expected Results
- What to look for when executing these steps
- Key indicators of success or failure

## Next Steps
- When to escalate and to whom (as specified in the knowledge base)
- Follow-up actions if initial steps don't resolve the issue

Remember: Never make up code or commands. Only use what's explicitly provided in the knowledge base content. Pay attention to what the user has already verified to avoid redundant recommendations."""


# Main execution
async def main():
    """Main entry point for the knowledge retrieval server.
    
    Handles command-line argument parsing and server initialization.
    Supports both STDIO and SSE connection methods for MCP communication.
    The server automatically creates default knowledge resources on startup
    if they don't already exist.
    """
    import argparse
    import logging
    import os
    
    # Set logging level based on environment variable
    log_level = os.getenv("MCP_LOG_LEVEL", "INFO")
    logging.getLogger().setLevel(getattr(logging, log_level))
    
    parser = argparse.ArgumentParser(description="Knowledge Retrieval MCP Server")
    parser.add_argument(
        "--connection", 
        choices=["stdio", "sse"],
        default="stdio",
        help="Connection method (default: stdio)"
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host for SSE connections")
    parser.add_argument("--port", type=int, default=8002, help="Port for SSE connections")
    
    args = parser.parse_args()
    
    if args.connection == "stdio":
        await mcp.run_stdio_async()
    elif args.connection == "sse":
        await mcp.run_sse_async(host=args.host, port=args.port)

if __name__ == "__main__":
    import asyncio
    import atexit
    import signal
    
    # Register cleanup handler for proper resource disposal
    def cleanup_handler():
        global vector_kb
        if hasattr(vector_kb, 'cleanup'):
            vector_kb.cleanup()
    
    def signal_handler(sig, frame):
        cleanup_handler()
        exit(0)
    
    atexit.register(cleanup_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    asyncio.run(main())