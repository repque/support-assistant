# Multi-Team Classification Server Demo Guide

## Overview

This demo showcases the production-ready Multi-Team Classification Server built using the Model Context Protocol (MCP). The system provides intelligent support request classification for multiple teams (ATRS and Core) using end-to-end LLM sampling without any pattern-matching fallbacks.

## Installation & Setup

```bash
# Install the package
pip install -e .

# Verify installation with comprehensive test suite
python run_tests.py
```

## Demo Commands

### 1. System Overview
```bash
python test_system_overview.py
```
Shows complete system capabilities, available teams, categories, and sample classifications.

### 2. Multi-Team Classification Demo
```bash
python test_multi_team_classification.py
```
Demonstrates team-specific classification for both ATRS and Core teams with realistic scenarios.

### 3. End-to-End Workflow Demo
```bash
python test_classification_workflow.py
```
Shows the complete client-side classification workflow with team-specific prompts.

### 4. Error Handling Demo
```bash
python test_error_handling.py
```
Demonstrates robust error handling for edge cases and invalid inputs.


## Demo Features Showcased

### Multi-Team Support
- **ATRS Team**: Athena Trade and Risk Services support (query, outage, data_issue, bless_request, review_request)
- **Core Team**: Platform infrastructure support (query, jobs, sdlc, database, cloud, ai)
- **Team Management**: Dynamic team discovery and configuration
- **Team-Specific Prompts**: Customized classification prompts per team with examples

### End-to-End LLM Sampling
- **Client-Side Processing**: 3-step workflow avoiding server-initiated sampling deadlocks
- **Rich Prompts**: Team-specific prompts with categories, examples, and context
- **Structured Output**: JSON classification results with confidence, priority, and workflows

### ATRS Team Classification Examples
- **Query**: "Trade settlement feed from clearing house stopped working"
- **Outage**: "Risk system is completely down"  
- **Data Issue**: "Position reconciliation showing $2M discrepancy"
- **Bless Request**: "Can you bless this code?"
- **Review Request**: "Please review this code change trade model enhancement"

### Core Team Classification Examples  
- **Jobs**: "Nightly bob job failed with timeout error"
- **SDLC**: "Docker build is failing in the deployment pipeline"
- **Database**: "Hydra queries are running extremely slow"
- **Cloud**: "Kubernetes cluster is hitting memory limits"
- **AI**: "ML model inference API is returning 500 errors"

### Dynamic Configuration
- **JSON-Based Teams**: Easy addition of new teams via configuration files
- **Category Management**: Team-specific categories with subcategories and workflows
- **Training Examples**: Few-shot learning examples for accurate classification
- **Workflow Assignment**: Automatic workflow assignment based on category

## Sample Test Issues

### ATRS Team Scenarios
Try these with `team: "atrs"`:

1. **Query**: "Sales trade did not feed to MarkitWire, can you please assist?"
2. **Outage**: "Trade booking service is down and not responding"  
3. **Data Issue**: "Data reconciliation shows discrepancies in position reports"
4. **Bless Request**: "Can you bless this new trading strategy for production deployment?"
5. **Review Request**: "Please review this code change for the risk calculation engine"

### Core Team Scenarios  
Try these with `team: "core"`:

1. **Jobs**: "Nightly ETL job failed with timeout error"
2. **SDLC**: "Docker build is failing in the deployment pipeline"
3. **Database**: "PostgreSQL queries are running extremely slow"
4. **Cloud**: "Kubernetes cluster is hitting memory limits"
5. **AI**: "ML model inference API is returning 500 errors"

### Manual Testing with MCP Client

```python
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def test_classification():
    server_params = StdioServerParameters(
        command="python", 
        args=["-m", "mcp_servers.classification_server"]
    )
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            # List available teams
            teams = await session.call_tool("list_teams", {})
            print("Teams:", teams.content[0].text)
            
            # Get ATRS classification prompt
            prompt = await session.call_tool("get_classification_prompt", {
                "user_request": "Trade feed is down",
                "team": "atrs"
            })
            
            # Simulate LLM response and classify
            llm_response = '{"category": "outage", "subcategory": "system_down", "priority": "critical", "confidence": 0.9, "reasoning": "Trade feed outage is critical"}'
            
            result = await session.call_tool("classify_support_request", {
                "user_request": "Trade feed is down",
                "team": "atrs",
                "llm_response": llm_response
            })
            print("Classification:", result.content[0].text)
```

## Demo Architecture

The system demonstrates a production-ready MCP-based architecture:

### **Multi-Team Classification Server**
- **Team Configuration Manager**: Dynamic loading of JSON team configs
- **Classification Engine**: End-to-end LLM sampling without fallbacks  
- **Team Management Tools**: Discovery and introspection capabilities
- **Error Handling**: Comprehensive validation and graceful failures

### **Supporting MCP Servers**
- **Knowledge Server**: Financial services runbooks and procedures
- **Health Server**: System status monitoring and health checks

### **Client-Side Architecture**
- **3-Step Workflow**: Get prompt → Process with LLM → Parse result
- **MCP Client**: Standard MCP protocol client with sampling callbacks
- **Test Automation**: Comprehensive test suite with automated runner

## Key Demo Metrics

- **Teams Supported**: 2 (ATRS and Core)
- **Total Categories**: 11 across both teams
- **Tools Available**: 6 classification and team management tools
- **Test Coverage**: 5 comprehensive test suites
- **Classification Accuracy**: LLM-powered with confidence scoring
- **Response Time**: Sub-second for prompt generation and parsing
- **Scalability**: Easy addition of new teams via JSON configuration

## Business Value Demonstration

This demo shows how the system provides:

1. **Multi-Tenant Support**: Different teams with specialized categories
2. **Production-Ready Architecture**: No pattern-matching fallbacks
3. **Scalable Design**: Easy addition of new teams and categories
4. **Robust Error Handling**: Graceful handling of edge cases
5. **Comprehensive Testing**: Full test coverage with automated verification
6. **Standards-Based**: Built on Model Context Protocol (MCP)

## Production Readiness

The current implementation is production-ready with:

### **✅ Implemented Features**
- Multi-team classification system
- End-to-end LLM sampling 
- Dynamic team configuration loading
- Comprehensive error handling and validation
- Full test suite coverage
- Team management and discovery tools

### **Ready for Extension**
- Easy addition of new teams via JSON configs
- Configurable category hierarchies and workflows
- Custom sampling callbacks for different LLM providers
- Integration with ticketing systems and workflows

## Technical Highlights

- **Type-Safe**: Full Pydantic model validation throughout
- **Async Architecture**: Non-blocking MCP operations
- **Standards-Based**: Built on Model Context Protocol specification
- **Client-Side Sampling**: Avoids server-initiated sampling deadlocks
- **Comprehensive Testing**: 5 test suites covering all functionality
- **Error Resilient**: Graceful handling of malformed inputs and edge cases
- **Documentation**: Complete documentation with examples and guides