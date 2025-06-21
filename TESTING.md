# Classification Server Testing Guide

This directory contains comprehensive tests for the multi-team classification server.

## Quick Start

Run all tests with the test runner:
```bash
python run_tests.py
```

## Individual Tests

### Core Tests

1. **`test_system_overview.py`** - Complete system overview
   - Lists all available tools and teams
   - Shows team categories and capabilities
   - Demonstrates end-to-end functionality

2. **`test_multi_team_classification.py`** - Multi-team functionality
   - Tests ATRS and Core team classifications
   - Verifies team-specific categories and workflows
   - Tests team management tools

3. **`test_classification_workflow.py`** - Client-side workflow
   - Tests the 3-step classification process
   - Verifies team-specific prompts
   - Tests client-side LLM integration

4. **`test_error_handling.py`** - Error handling and edge cases
   - Tests missing parameters
   - Tests invalid team/category names
   - Tests malformed responses
   - Verifies graceful error handling

### Integration Tests

5. **`test_knowledge.py`** - Knowledge server integration
   - Tests knowledge server functionality
   - Verifies resource retrieval

## Classification Server Features

### Team Configuration
- **ATRS Team**: `[query, outage, data_issue, bless_request, review_request]`
- **Core Team**: `[query, jobs, sdlc, database, cloud, ai]`

### Available Tools
1. `classify_support_request` - Classify requests by team
2. `get_classification_prompt` - Get team-specific prompts
3. `list_teams` - List available teams
4. `get_team_info` - Get team details
5. `get_team_categories` - Get team categories
6. `get_category_details` - Get category details

### Key Features
✅ **Multi-tenant**: Supports multiple teams with different categories  
✅ **End-to-end LLM sampling**: No pattern-matching fallbacks  
✅ **Client-side processing**: Avoids server-initiated sampling deadlocks  
✅ **Dynamic configuration**: JSON-based team configs  
✅ **Comprehensive error handling**: Graceful failure modes  
✅ **Team-specific prompts**: Customized classification prompts per team  

## Expected Test Results

All tests should pass:
- ✅ System Overview & Capabilities
- ✅ Multi-Team Classification
- ✅ Classification Workflow  
- ✅ Error Handling & Edge Cases
- ✅ Knowledge Server Integration

## Troubleshooting

If tests fail:
1. Ensure all dependencies are installed: `pip install -e .`
2. Check that MCP servers can start: `python -m mcp_servers.classification_server --help`
3. Verify team configurations exist: `ls mcp_servers/categories/`
4. Run individual tests for detailed error messages