# AI Production Support Assistant

A production-ready AI support assistant system that provides intelligent analysis and resolution recommendations for production issues across multiple teams using Model Context Protocol (MCP) architecture.

## Quick Start

### 1. Install Dependencies
```bash
pip install -e .
```

### 2. Run Tests
```bash
python run_tests.py
```

### 3. Start Support Assistant
```bash
python -m support_agent.cli
```

### 4. Demo Mode
```bash
python -m support_agent.cli --demo
```

### 5. Interactive Analysis
```bash
python -m support_agent.cli --interactive
```

Example support request:
```
> My MarkitWire feed isn't sending outbound messages

Classification: query/feed_issue (High Priority, 85% confidence)
Resolution Steps:
   1. Check feed status: SELECT * FROM trade_feeds WHERE feed_name = 'MarkitWire'
   2. Verify connectivity: curl -H "Authorization: Bearer $TOKEN" https://api.markitwire.com/status
   3. Review validation results and check downstream systems
Confidence: 100% → Provides comprehensive guidance
Performance: 5 tools, 3,500 tokens, 4.2 seconds
```

## Features

✅ **Intelligent Analysis**: End-to-end support request analysis with confidence scoring  
✅ **Multi-Team Support**: ATRS and Core teams with distinct category sets  
✅ **Knowledge Integration**: Contextual recommendations from knowledge base  
✅ **Health Monitoring**: System status checks and diagnostic guidance  
✅ **Anti-Hallucination**: 60% confidence threshold with silent mode for uncertain cases  
✅ **Client-Side LLM**: Avoids server-initiated sampling deadlocks  
✅ **Production Ready**: Comprehensive error handling and full test coverage  

## Team Categories

### ATRS Team (Financial Services)
- `query` - Technical questions and troubleshooting
- `outage` - System outages and service disruptions  
- `data_issue` - Data quality and reconciliation problems
- `bless_request` - Code deployment approvals
- `review_request` - Code and architecture reviews

### Core Team (Platform Infrastructure)  
- `query` - General technical guidance
- `jobs` - Batch jobs and ETL processes
- `sdlc` - CI/CD, builds, and deployments
- `database` - Database performance and connectivity
- `cloud` - Cloud infrastructure and scaling
- `ai` - AI/ML model deployment and inference

## Core Capabilities

1. **Support Request Analysis** - Comprehensive analysis with confidence scoring
2. **Multi-Team Classification** - ATRS and Core team-specific categorization  
3. **Knowledge Base Integration** - Contextual recommendations from documentation
4. **Health Status Monitoring** - System status checks and diagnostics
5. **Intelligent Decision Making** - Silent mode for low confidence scenarios
6. **Performance Tracking** - Token usage and response time monitoring

## Architecture

```
support_agent/
├── assistant.py               # Main support assistant engine
├── cli.py                    # Command-line interface
├── config.py                 # Configuration management
└── models.py                 # Pydantic data models

mcp_servers/
├── categories/
│   ├── atrs.json              # ATRS team configuration
│   └── core.json              # Core team configuration  
├── classification_server.py   # Multi-team classification MCP server
├── knowledge_server.py        # Knowledge base MCP server
└── health_server.py          # Health monitoring MCP server

knowledge_resources/           # Knowledge base content
└── test_*.py                 # Comprehensive test suite
```

## Testing

The system includes comprehensive tests:

- **`test_system_overview.py`** - Complete system capabilities
- **`test_multi_team_classification.py`** - Multi-team functionality  
- **`test_classification_workflow.py`** - End-to-end workflow
- **`test_error_handling.py`** - Error handling and edge cases
- **`test_knowledge.py`** - Knowledge server integration

Run all tests: `python run_tests.py`

See [TESTING.md](TESTING.md) for detailed testing documentation.

## Configuration

### Adding New Teams

1. Create team configuration file in `mcp_servers/categories/`:
```json
{
  "team_info": {
    "name": "New Team",
    "description": "Team description",
    "domain": "team_domain"
  },
  "categories": {
    "category_name": {
      "description": "Category description",
      "subcategories": ["sub1", "sub2"],
      "workflow": "workflow_name"
    }
  },
  "training_examples": [
    {
      "text": "Example request",
      "category": "category_name", 
      "subcategory": "sub1",
      "priority": "high",
      "reasoning": "Why this classification"
    }
  ]
}
```

2. Restart the server - teams are loaded automatically

### Client-Side LLM Integration

The server uses a 3-step client-side approach to avoid deadlocks:

1. **Get Prompt**: Client requests team-specific classification prompt
2. **Process with LLM**: Client sends prompt to their LLM service  
3. **Parse Result**: Client sends LLM response back for parsing and validation

This architecture ensures the server never initiates sampling requests that could cause deadlocks.

## Documentation

- [DESIGN.md](DESIGN.md) - System design documentation
- [DEMO_GUIDE.md](DEMO_GUIDE.md) - Demo walkthrough guide
- [TESTING.md](TESTING.md) - Testing guide and documentation