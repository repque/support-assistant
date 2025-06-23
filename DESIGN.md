# AI Production Support Assistant - System Design

## Overview

The AI Production Support Assistant is a production-ready system built using Model Context Protocol (MCP) architecture with vector-based semantic search and intelligent LLM decision making. The system provides context-aware analysis and resolution recommendations for production support requests with zero hardcoded business logic.

## Core Architecture

### High-Level System Flow
```
User Request → Classification → Knowledge Search → LLM Analysis → Recommendations
     ↓              ↓                ↓                ↓              ↓
[CLI/API] → [Classification MCP] → [Knowledge MCP] → [Assistant] → [Formatted Output]
```

### Key Components

1. **Support Assistant** (`support_agent/assistant.py`)
   - Central orchestrator managing all MCP server interactions
   - Implements confidence-based decision framework
   - Handles LLM integration for intelligent analysis
   - Manages token tracking and performance monitoring

2. **Classification MCP Server** (`mcp_servers/classification_server.py`)
   - Team-specific request categorization using LLM analysis
   - JSON-configurable categories and workflows
   - Few-shot learning with training examples
   - Confidence scoring for decision making

3. **Knowledge MCP Server** (`mcp_servers/knowledge_server.py`)
   - Section-level vector search using sentence-transformers
   - Markdown-based knowledge base with automatic indexing
   - LLM-based query intent analysis for better relevance
   - Deterministic follow-up searches for gap identification

4. **Health MCP Server** (`mcp_servers/health_server.py`)
   - System monitoring and health checks
   - MCP server status validation
   - Extension point for additional monitoring

## Key Design Principles

### 1. Zero Hardcoded Business Logic
- All decisions made by LLM analysis, not hardcoded rules
- Generic parameter substitution (`feedType`, `dealName`) in knowledge base
- Team configurations stored as JSON files, not in code
- No special-case handling for specific feeds, systems, or scenarios

### 2. Context-Aware Intelligence
- Skips redundant steps based on user's stated facts
- Intelligent parameter substitution in code examples
- Understands user's current troubleshooting state
- Adapts recommendations to specific user context

### 3. Deterministic Quality Control
- Temperature=0.0 for gap identification to ensure consistency
- 30% relevance threshold for follow-up content inclusion
- Quality filtering prevents low-value results
- Consistent source attribution and prioritization

### 4. Section-Level Knowledge Architecture
- Knowledge base parsed at markdown section level for granular search
- Vector embeddings computed for each section independently
- Enables precise relevance matching and attribution
- Supports complex hierarchical knowledge structures

## Technical Implementation

### Vector Search Architecture
```python
# Section-level embedding generation
for section_title, section_content, section_path in sections:
    clean_title = remove_doc_name_contamination(section_title)
    text_to_embed = f"{clean_title} {section_content[:1000]}"
    embedding = generate_embedding(text_to_embed)
    
# Semantic search with LLM intent analysis
query_intent = await analyze_query_intent(query)
similarities = cosine_similarity(query_embedding, section_embeddings)
apply_concept_boosting_and_filtering(similarities, query_intent)
```

### Follow-Up Search Mechanism
```python
# Deterministic gap identification
gaps = await identify_knowledge_gaps(
    initial_results, 
    user_query, 
    temperature=0.0  # Ensures consistency
)

# Quality-filtered follow-up searches
for gap in gaps:
    followup_results = await search_knowledge(gap.search_term)
    filtered_results = filter_by_relevance(followup_results, threshold=0.3)
    if filtered_results:
        combined_knowledge += filtered_results
```

### LLM Integration Pattern
- **Client-Side LLM Processing**: Avoids MCP sampling deadlocks
- **OpenAI/Anthropic Support**: Automatic API detection and fallback
- **Structured Prompts**: Consistent prompt engineering for reliable outputs
- **Token Tracking**: Comprehensive usage monitoring

## Data Flow

### 1. Request Processing
1. User submits support request via CLI
2. Assistant starts MCP servers if not running
3. Classification server generates structured prompt
4. Assistant calls LLM for request categorization
5. Classification parsed and confidence evaluated

### 2. Knowledge Retrieval
1. Knowledge server performs vector search on user query
2. LLM analyzes query intent for better relevance scoring
3. Top results filtered by relevance threshold (60% primary)
4. Gap identification performed on initial results

### 3. Follow-Up Enhancement
1. LLM identifies missing implementation details (temperature=0.0)
2. Additional searches performed for each identified gap
3. Follow-up results filtered by relevance (30% threshold)
4. Combined knowledge prepared for final synthesis

### 4. Response Generation
1. Structured prompt built with combined knowledge
2. LLM generates context-aware recommendations
3. Source attribution computed (top 3 primary sources)
4. Rich CLI output formatted and displayed

## Configuration Architecture

### Environment Variables
```bash
# LLM Configuration
export OPENAI_API_KEY="your-key"
export LLM_DEFAULT_MODEL="gpt-4o-mini"
export LLM_TEMPERATURE="0.3"
export LLM_MAX_TOKENS="1000"

# Network Configuration  
export MCP_HOST="127.0.0.1"
export MCP_PORT_BASE="8000"
```

### Team Configuration (`mcp_servers/categories/`)
```json
{
  "team_info": {
    "name": "ATRS Platform Support",
    "description": "Application Trading & Risk Services support team"
  },
  "categories": {
    "feed_issue": {
      "description": "Issues with data feeds and downstream processing",
      "subcategories": ["validation", "outbound", "reconciliation"],
      "workflow": "technical_guidance"
    }
  },
  "training_examples": [...]
}
```

### Knowledge Base Structure
```
knowledge_resources/
├── markitwire_feed_troubleshooting.md
├── feed_framework_troubleshooting.md  
├── data_reconciliation.md
└── outage_investigation.md
```

Each file uses hierarchical markdown sections for granular search:
```markdown
# Document Title
## Category
### Specific Procedure
#### Implementation Details
```

## Quality Assurance

### Deterministic Behavior
- **Gap Identification**: Temperature=0.0 ensures consistent results
- **Relevance Thresholds**: Hard limits prevent low-quality content inclusion
- **Source Attribution**: Consistent top-3 primary source selection
- **Parameter Substitution**: Reliable context-aware code adaptation

### Testing Strategy
- **Functional Tests**: End-to-end CLI testing with real scenarios
- **Context Awareness**: Verify redundant step skipping
- **Generic Parameters**: Test feed type substitution across different feeds
- **Human Review Detection**: Validate LLM-based decision making
- **Silent Mode**: Ensure appropriate silence for vague queries

### Performance Monitoring
- **Token Tracking**: Monitor LLM usage and costs
- **Tool Call Counting**: Track MCP server interactions
- **Response Times**: End-to-end latency measurement
- **Health Checks**: Continuous server status monitoring

## Extension Points

### Adding New Knowledge
1. Create markdown files in `knowledge_resources/`
2. Use parameterized examples (`feedType`, `dealName`)
3. Maintain clear hierarchical section structure
4. System automatically indexes on startup

### Team Configuration
1. Add JSON config file to `mcp_servers/categories/`
2. Define team-specific categories and workflows
3. Include training examples for few-shot learning
4. No code changes required

### Additional MCP Servers
1. Implement FastMCP server interface
2. Register tools with appropriate schemas
3. Add to `MCPConfig` server definitions
4. Assistant automatically integrates via MCP protocol

## Security Considerations

- **API Key Management**: Environment variable based, no hardcoded keys
- **Input Validation**: MCP schema validation on all inputs
- **Content Filtering**: No sensitive information in knowledge base
- **Audit Trail**: Comprehensive logging of all operations

## Performance Characteristics

- **Response Time**: 4-8 seconds end-to-end for typical queries
- **Token Usage**: 3-6k tokens per analysis (optimized prompts)
- **Knowledge Search**: <1 second semantic search across full knowledge base
- **Scalability**: Stateless design supports horizontal scaling
- **Memory Usage**: Efficient vector storage with cleanup handlers

## Future Enhancements

1. **Caching Layer**: Redis-based caching for frequent queries
2. **Real-time Learning**: Feedback loop for improving classifications
3. **Advanced Analytics**: Query pattern analysis and optimization
4. **Multi-modal Support**: Image and document analysis capabilities
5. **Integration APIs**: REST/GraphQL endpoints for external systems