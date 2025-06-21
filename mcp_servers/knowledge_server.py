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
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from mcp.server.fastmcp import FastMCP

# Create FastMCP server instance
mcp = FastMCP("KnowledgeRetrievalServer")

# Initialize resources directory
RESOURCES_DIR = Path("./knowledge_resources")
RESOURCES_DIR.mkdir(parents=True, exist_ok=True)

def _create_default_resources():
    """Create default knowledge resources for financial services support.
    
    Initializes the knowledge base with essential documentation covering
    common support scenarios in financial services. Each resource includes:
    - Structured troubleshooting procedures
    - SQL queries and code snippets for investigation
    - Step-by-step resolution processes
    - Quality assurance and documentation requirements
    
    Resources are stored as markdown files with accompanying metadata
    for efficient search and categorization.
    """
    
    resources = [
        {
            "filename": "data_reconciliation.md",
            "title": "Data Reconciliation Issue Resolution",
            "description": "Procedures for resolving data quality and reconciliation issues",
            "category": "data_issue", 
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
            
            # Create metadata file
            metadata = {
                "title": resource["title"],
                "description": resource["description"],
                "category": resource["category"],
                "created": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat()
            }
            metadata_path = RESOURCES_DIR / f"{resource['filename']}.meta"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

# Initialize resources on startup
_create_default_resources()

@mcp.tool()
def search_knowledge(query: str, category: Optional[str] = None, max_results: int = 5) -> List[Dict]:
    """Search knowledge base for relevant support documentation.
    
    Performs intelligent search across the knowledge base using relevance
    scoring that considers query terms, category matching, and content
    analysis. Results are ranked by relevance to ensure the most applicable
    solutions are returned first.
    
    Args:
        query: Search query describing the issue or topic.
        category: Optional category filter to improve relevance (e.g., 'data_issue').
        max_results: Maximum number of results to return (default: 5).
    
    Returns:
        List[Dict]: Ranked list of knowledge resources containing:
                   - content: Full text of the knowledge article
                   - source: Resource identifier and title
                   - relevance_score: Score between 0.0 and 1.0
                   - metadata: Additional resource information
    """
    results = []
    
    # Get all available resources
    resources = []
    for file_path in RESOURCES_DIR.glob("*.md"):
        metadata_path = RESOURCES_DIR / f"{file_path.name}.meta"
        
        # Load metadata if exists
        metadata = {}
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        
        # Create resource entry
        resource = {
            "uri": f"knowledge://{file_path.name}",
            "name": file_path.stem,
            "title": metadata.get("title", file_path.stem.replace("_", " ").title()),
            "description": metadata.get("description", f"Knowledge resource: {file_path.stem}"),
            "mime_type": "text/markdown"
        }
        resources.append(resource)
    
    for resource in resources:
        # Calculate relevance based on query
        relevance_score = _calculate_resource_relevance(resource, query, category)
        
        if relevance_score > 0.1:  # Minimum relevance threshold
            # Read resource content for result
            try:
                filename = resource["uri"].replace("knowledge://", "")
                file_path = RESOURCES_DIR / filename
                
                with open(file_path, 'r') as f:
                    content = f.read()
                
                # Create knowledge result
                result = {
                    "content": content,
                    "source": f"Knowledge Resource - {resource['title']}",
                    "relevance_score": relevance_score,
                    "metadata": {
                        "uri": resource["uri"],
                        "title": resource["title"],
                        "description": resource["description"],
                        "mime_type": resource["mime_type"]
                    }
                }
                results.append(result)
                
            except Exception:
                # Skip resources that can't be read
                continue
    
    # Sort by relevance and return top results
    results.sort(key=lambda x: x["relevance_score"], reverse=True)
    return results[:max_results]

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

Based on the user's specific issue and the knowledge base content, provide a concise response with:

## Immediate Actions
- List 3-4 specific steps directly relevant to this exact issue
- Include specific commands or queries from the knowledge base

## Expected Results
- What to look for when executing these steps
- Key indicators of success or failure

## Next Steps
- When to escalate and to whom
- Follow-up actions if initial steps don't resolve the issue

Focus only on what's directly relevant to this specific issue. Be concise and actionable."""

def _calculate_resource_relevance(resource: Dict, query: str, category: Optional[str] = None) -> float:
    """Calculate relevance score for a knowledge resource.
    
    Implements a multi-factor scoring algorithm that considers:
    - Category matching (bonus for matching categories)
    - Term frequency in metadata (title, description)
    - Term presence in content body
    - Query term coverage
    
    The scoring is designed to favor resources that closely match both
    the query terms and the issue category, ensuring highly relevant
    results for support scenarios.
    
    Args:
        resource: Resource dictionary with metadata.
        query: Search query to match against.
        category: Optional category for bonus scoring.
    
    Returns:
        float: Normalized relevance score between 0.0 and 1.0.
    """
    query_terms = query.lower().split()
    score = 0.0
    
    # Check resource metadata for matches
    searchable_text = " ".join([
        resource["title"].lower(),
        resource["description"].lower(),
        resource["name"].lower()
    ])
    
    # Load metadata for category matching
    filename = resource["uri"].replace("knowledge://", "")
    metadata_path = RESOURCES_DIR / f"{filename}.meta"
    resource_category = None
    
    if metadata_path.exists():
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                resource_category = metadata.get("category")
        except:
            pass
    
    # Category matching bonus (if context provided)
    if category and resource_category and category == resource_category:
        score += 2.0
    elif category and resource_category and category != resource_category:
        score -= 0.5
    
    # Term matching in metadata
    for term in query_terms:
        if term in searchable_text:
            score += 0.8
    
    # Load and search content for better matching
    try:
        file_path = RESOURCES_DIR / filename
        with open(file_path, 'r') as f:
            content_text = f.read().lower()
        
        for term in query_terms:
            if term in content_text:
                score += 0.3
                
    except:
        # If can't read content, rely on metadata only
        pass
    
    # Normalize score
    return min(score / len(query_terms) if query_terms else 0.0, 1.0)

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
    asyncio.run(main())