# AI Production Support Assistant Demo Guide

## Demo Video

🎥 **[Watch the Demo Video](./Demo.mov)** - See the system in action with real troubleshooting scenarios

## Overview

This demo showcases the production-ready AI Production Support Assistant built using Model Context Protocol (MCP) architecture with vector-based semantic search and intelligent LLM decision making. The system provides context-aware analysis and resolution recommendations for production support requests with no hardcoded business logic.

## Quick Start

### Installation & Setup
```bash
# Install the package
pip install -e .

# Set up API key
export OPENAI_API_KEY="your-openai-key"
# OR
export ANTHROPIC_API_KEY="your-anthropic-key"

# Verify installation
python -m pytest tests/ -v
```

## Demo Commands

### 1. Automated Demo Scenarios
```bash
python -m support_agent.cli demo --no-interactive
```
**Best starting point** - Shows three key scenarios:
- **High Confidence Analysis**: MarkitWire feed troubleshooting with executable Python code
- **Human Review Required**: Code review requests properly deferred to humans  
- **Silent Mode**: Vague queries handled gracefully without recommendations

### 2. Interactive Demo
```bash
python -m support_agent.cli demo
```
Interactive session where you can test any support requests:
- Try: `"I booked a trade in Athena but it didn't show up in the MarkitWire feed"`
- Try: `"DCPP feed not working"`  
- Try: `"my trade has book2 resolved to 'MarkitWire' but validation failed"`
- Try: `"Can you please check this code: https://github.com/sdlc/hydra/pull/1234?"`

### 3. System Health Check
```bash
python -m support_agent.cli health
```
Verifies all MCP servers (Classification, Knowledge, Health) are operational.

### 4. System Information
```bash
python -m support_agent.cli info
```
Shows system capabilities and knowledge base entries.

### 5. Advanced Configuration
```bash
# Enable recursive knowledge search
python -m support_agent.cli demo --search-depth 2

# Test different connection methods
python -m support_agent.cli demo --connection sse
```

## Key Features Demonstrated

### 1. **Context-Aware Analysis**
**Demo**: `"my trade has book2 resolved to 'MarkitWire' but it did not feed outbound"`

**Shows**:
- System recognizes user already verified book2 → skips redundant verification
- Provides targeted recommendations for remaining issues
- Intelligent understanding of user's current state

### 2. **Generic Feed Support** 
**Demo**: Try with different feed types:
- `"DCPP feed issue"` → Provides `FeedState("DCPP")` code
- `"XODS feed status"` → Provides `FeedState("XODS")` code

**Shows**:
- No hardcoded feed types anywhere in system
- LLM intelligently substitutes parameters based on user context
- Works with any feed type without configuration changes

### 3. **Vector-Based Knowledge Search**
**Demo**: `"MarkitWire feed issue"`

**Shows**:
- Semantic search finds relevant knowledge using vector embeddings
- Server-side similarity search with cosine distance
- Provides implementation details: `ds.evInfo()` for listing downstream events

### 4. **LLM-Based Decision Making**
**Demo**: `"Can you please check this trade for compliance approval?"`

**Shows**:
- LLM analyzes request content to determine if human review needed
- No hardcoded categories for review detection
- Graceful deferral to human experts

### 5. **Silent Mode Intelligence**
**Demo**: `"I need help with this thing"`

**Shows**:
- System stays silent when no relevant knowledge available
- Graceful error handling for vague or incomplete requests
- No unhelpful "I don't know" responses

### 6. **Recursive Knowledge Enhancement**
**Demo**: Enable with `--search-depth 2`

**Shows**:
- LLM identifies knowledge gaps (e.g., "check block events" without code)
- Automatically searches for missing implementation details
- Combines multiple knowledge sources for complete guidance
- Intelligent gap detection and knowledge synthesis

## Technical Architecture Highlights

### Vector Embeddings
- **Model**: `sentence-transformers` with `all-MiniLM-L6-v2`
- **Search**: Server-side semantic search with cosine similarity
- **Performance**: Sub-second search across knowledge base
- **Compatibility**: NumPy 1.26.4 for stable vector operations

### LLM Integration  
- **APIs**: OpenAI and Anthropic supported
- **Pattern**: Client-side LLM processing (no server sampling deadlocks)
- **Usage**: ~3-6k tokens per analysis, ~4-8 second response time
- **Intelligence**: Context awareness, parameter substitution, gap detection

### Knowledge Base
- **Format**: Pure markdown with section-level indexing
- **Parameterization**: Generic examples using `feedType`, `dealName` variables
- **Content**: Feed troubleshooting, data reconciliation, outage procedures, etc.
- **Updates**: Automatic section parsing and indexing of new knowledge files
- **Search**: Section-level embeddings with DFS recursive knowledge retrieval

## Demo Scenarios Walkthrough

### Scenario 1: Data Reconciliation (High Confidence)
```
Input: "Data reconciliation shows discrepancies in position reports"

Output:
- Classification: data_issue/reconciliation (High Priority, 80% confidence)
- Provides specific code for investigation
- Details validation procedures and escalation paths
- 100% confidence → comprehensive guidance provided
```

### Scenario 2: Bless Review (Human Review)
```
Input: "Can you please bless this code?"

Output:
- Classification: bless_request (detected by LLM analysis)
- System stays silent - defers to human experts
- No automated bless review decisions (appropriate)
```

### Scenario 3: Vague Query (Silent Mode)
```
Input: "I need help with this thing"

Output:
- Classification: query (attempted but low relevance)
- No relevant knowledge found in section-level vector search
- System stays silent rather than providing unhelpful responses
- Graceful handling prevents noise in production channels
```

## Performance Metrics

Typical demo performance:
- **Response Time**: 4-8 seconds end-to-end
- **Token Usage**: 3-6k tokens per analysis
- **Tool Calls**: 4-6 MCP operations per request
- **Knowledge Search**: <1 second semantic search
- **Accuracy**: High relevance due to vector search + LLM processing

## Troubleshooting Demo Issues

### Common Issues
1. **API Key**: Ensure `OPENAI_API_KEY` is set
2. **Dependencies**: Run `pip install 'numpy>=1.26.4,<2.0' sentence-transformers`
3. **Server Startup**: Check MCP servers start: `python -m support_agent.cli health`

### Demo Tips
1. **Start with automated demo** (`--no-interactive`) to see system capabilities
2. **Test context awareness** with queries that state what you've already verified
3. **Try bless requests** to see human review detection
4. **Use vague queries** to see silent mode behavior

## Extending the Demo

### Adding New Knowledge
1. Create `.md` file in `knowledge_resources/`
2. Use clear header structure (# ## ### for hierarchical sections)
3. Use parameterized examples: `feedType`, `dealName`, etc. whenever possible
4. System automatically parses sections and indexes content upon startup

### Testing New Scenarios
1. Add test cases to `tests/test_functional.py`
2. Run full test suite: `python -m pytest tests/ -v`
3. Test manually: `python -m support_agent.cli demo`
