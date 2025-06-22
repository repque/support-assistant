# Support Agent Testing Guide

This document covers testing for the AI Production Support Assistant system.

## Quick Start

Run all tests:
```bash
python -m pytest tests/ -v
```

Run tests directly:
```bash
python tests/test_functional.py
```

## Test Suite Overview

The system uses functional testing that validates end-to-end functionality through the CLI interface.

### Current Tests (`tests/test_functional.py`)

1. **`test_cli_health_check`** - CLI health command functionality
   - Verifies MCP servers start correctly
   - Tests health status reporting
   - Validates server connectivity

2. **`test_cli_info_command`** - CLI info command functionality  
   - Tests system information display
   - Verifies knowledge base entry counts
   - Validates category and workflow reporting

3. **`test_cli_demo_scenarios`** - Automated demo scenarios
   - Tests all three demo scenarios without crashes
   - Verifies proper server startup and shutdown
   - Validates end-to-end processing

4. **`test_markitwire_context_awareness`** - Core context awareness
   - Tests the specific MarkitWire context-aware functionality
   - Verifies book2 verification is skipped when user states it's resolved
   - Validates block events code (`ds.evInfo()`) is provided
   - Tests vector search functionality

5. **`test_vague_query_handling`** - Edge case handling
   - Tests vague queries like "I need help with this thing"
   - Verifies system stays silent for queries with no relevant knowledge
   - Tests graceful error handling for empty/minimal queries
   - Ensures no crashes occur

6. **`test_human_review_detection`** - LLM-based decision making
   - Tests compliance request detection
   - Verifies system properly defers review requests to humans
   - Validates LLM-based decision making (no hardcoded logic)

7. **`test_no_hardcoded_logic_in_codebase`** - Code quality validation
   - Scans codebase for hardcoded patterns
   - Ensures no hardcoded feed types or business logic
   - Validates LLM-based approach throughout

## Key System Features Tested

### ✅ **Vector Embeddings & Semantic Search**
- Server-side vector search using `sentence-transformers`
- Semantic similarity with cosine distance
- No fallback to keyword matching

### ✅ **Context Awareness** 
- Skips redundant verification steps
- Considers user's stated facts
- Adapts recommendations to user context

### ✅ **Generic Feed Support**
- Works with any feed type (MarkitWire, DCPP, XODS, Bloomberg, Reuters)
- Intelligent parameter substitution
- No hardcoded feed-specific logic

### ✅ **LLM-Based Decision Making**
- Request handling decisions made by LLM
- Gap detection and recursive knowledge search
- Human review detection without hardcoded categories

### ✅ **Graceful Error Handling**
- Silent mode for low-confidence scenarios
- Proper handling of empty search results
- No crashes on edge cases

## Test Results Expected

All 7 tests should pass:
```
tests/test_functional.py::test_cli_health_check PASSED
tests/test_functional.py::test_cli_info_command PASSED  
tests/test_functional.py::test_cli_demo_scenarios PASSED
tests/test_functional.py::test_markitwire_context_awareness PASSED
tests/test_functional.py::test_vague_query_handling PASSED
tests/test_functional.py::test_human_review_detection PASSED
tests/test_functional.py::test_no_hardcoded_logic_in_codebase PASSED
```

Typical runtime: ~80 seconds (includes full system startup/shutdown cycles)

## Manual Testing

### Demo Scenarios
```bash
# Automated scenarios - tests all key functionality
python -m support_agent.cli demo --no-interactive

# Interactive testing
python -m support_agent.cli demo
```

### Health Check
```bash
python -m support_agent.cli health
```

### System Information
```bash  
python -m support_agent.cli info
```

## Test Coverage

### Core Functionality
- ✅ **MCP Server Integration** - All three servers (classification, knowledge, health)
- ✅ **Vector Search** - Semantic search with embeddings
- ✅ **LLM Integration** - OpenAI/Anthropic API integration
- ✅ **Context Processing** - User context awareness
- ✅ **Parameter Substitution** - Feed type and other parameter adaptation

### Edge Cases
- ✅ **Empty Queries** - Graceful handling of minimal input
- ✅ **Vague Requests** - Appropriate silent mode behavior  
- ✅ **Review Requests** - Human review detection and deferral
- ✅ **Unknown Feed Types** - Generic handling of any feed type

### Architecture Validation
- ✅ **No Hardcoding** - Scans for hardcoded business logic
- ✅ **LLM Decisions** - Validates LLM-based decision making
- ✅ **Generic Knowledge** - Tests parameterized knowledge base

## Performance Metrics

Typical performance from test results:
- **Response Time**: 4-8 seconds end-to-end
- **Token Usage**: 3-6k tokens per analysis  
- **Tool Calls**: 4-6 MCP tool calls per request
- **Memory Usage**: Minimal (vector embeddings cached)

## Troubleshooting

### Test Failures

1. **Vector Embedding Issues**
   ```bash
   pip install 'numpy>=1.26.4,<2.0' sentence-transformers scikit-learn
   ```

2. **API Key Issues**
   ```bash
   export OPENAI_API_KEY="your-key"
   # OR
   export ANTHROPIC_API_KEY="your-key"
   ```

3. **MCP Server Startup Issues**
   - Check Python path and dependencies
   - Verify knowledge resources exist: `ls knowledge_resources/`
   - Test individual servers: `python mcp_servers/knowledge_server.py --help`

### Expected Warnings
- Numpy compatibility warnings (handled gracefully)
- Pytest async deprecation warnings (non-critical)
- Resource tracker warnings (cleanup-related, non-critical)

## Adding New Tests

When adding functionality, add corresponding tests to `tests/test_functional.py`:

1. **CLI Commands** - Test through subprocess calls
2. **Core Functionality** - Test through async assistant calls  
3. **Edge Cases** - Test error handling and boundary conditions
4. **Integration** - Test end-to-end workflows

Example test structure:
```python
@pytest.mark.asyncio
async def test_new_functionality():
    config = MCPConfig()
    assistant = SupportAssistant(config)
    
    success = await assistant.start_mcp_servers()
    assert success
    
    try:
        # Test code here
        pass
    finally:
        await assistant.stop_mcp_servers()
```

## Legacy Tests

Previous complex unit tests with fixture dependencies have been removed in favor of simpler, more reliable functional tests that test the system as users actually interact with it.