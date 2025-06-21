# Multi-Team Classification Server

A production-ready Model Context Protocol (MCP) server that provides intelligent support request classification for multiple teams using end-to-end LLM sampling.

## Quick Start

### 1. Install Dependencies
```bash
pip install -e .
```

### 2. Run Tests
```bash
python run_tests.py
```

### 3. Start Classification Server
```bash
python -m mcp_servers.classification_server
```

### 4. Use with MCP Client
```python
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Connect to classification server
server_params = StdioServerParameters(
    command="python", 
    args=["-m", "mcp_servers.classification_server"]
)

async with stdio_client(server_params) as (read, write):
    async with ClientSession(read, write) as session:
        await session.initialize()
        
        # List available teams
        teams = await session.call_tool("list_teams", {})
        
        # Get classification prompt for ATRS team
        prompt = await session.call_tool("get_classification_prompt", {
            "user_request": "Trade feed is down", 
            "team": "atrs"
        })
        
        # Process with your LLM, then classify
        result = await session.call_tool("classify_support_request", {
            "user_request": "Trade feed is down",
            "team": "atrs", 
            "llm_response": llm_response
        })
```

## Features

✅ **Multi-Team Support**: ATRS and Core teams with distinct category sets  
✅ **End-to-End LLM Sampling**: No pattern-matching fallbacks  
✅ **Client-Side Processing**: Avoids server-initiated sampling deadlocks  
✅ **Dynamic Configuration**: JSON-based team configurations  
✅ **Team Management**: Discovery and introspection tools  
✅ **Comprehensive Error Handling**: Graceful failure modes  
✅ **Production Ready**: Fully tested with comprehensive test suite  

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

## Available Tools

1. **`classify_support_request`** - Classify requests by team with LLM
2. **`get_classification_prompt`** - Generate team-specific prompts  
3. **`list_teams`** - Discover available teams
4. **`get_team_info`** - Get team details and descriptions
5. **`get_team_categories`** - Get categories for specific team
6. **`get_category_details`** - Get detailed category information

## Architecture

```
mcp_servers/
├── categories/
│   ├── atrs.json              # ATRS team configuration
│   └── core.json              # Core team configuration  
├── classification_server.py   # Multi-team classification server
├── knowledge_server.py        # Knowledge base server
└── health_server.py          # Health monitoring server

support_agent/
├── assistant.py               # Main support assistant
├── cli.py                    # Command-line interface
├── config.py                 # Configuration management
└── models.py                 # Pydantic data models
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