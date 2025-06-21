"""AI Production Support Assistant - Phase 1 Demo.

This package implements an intelligent support assistant for production systems
that leverages MCP (Model Context Protocol) servers and LLM capabilities to
provide automated analysis and recommendations for technical support requests.

The assistant coordinates multiple specialized services:
- Classification: Categorizes and prioritizes support requests
- Knowledge Retrieval: Searches documentation for relevant solutions
- Health Monitoring: Checks system status and analyzes logs

Key features:
- Confidence-based decision making
- Intelligent silence when confidence is low
- Multi-team support with configurable categories
- Rich CLI interface for demonstrations
"""

__version__ = "0.1.0"