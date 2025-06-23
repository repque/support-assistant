"""Configuration module for MCP server connections and settings.

This module defines the configuration structures for the support agent's
MCP (Model Context Protocol) server connections. It provides a flexible
configuration system that supports multiple connection methods and server
endpoints, allowing the system to adapt to different deployment scenarios.
"""

import os
from enum import Enum
from typing import Dict, Optional

from pydantic import BaseModel


class ConnectionMethod(Enum):
    """Supported MCP connection methods for server communication.

    Defines the available protocols for connecting to MCP servers. Each method
    has different characteristics suitable for various deployment scenarios:

    - STDIO: Process-based communication via standard input/output (default)
    - SSE: HTTP-based Server-Sent Events for network communication
    - HTTP: Standard HTTP REST API (placeholder for future use)
    """

    STDIO = "stdio"
    SSE = "sse"
    HTTP = "http"


class LLMConfig(BaseModel):
    """Configuration for LLM provider settings.

    Centralizes all LLM-related configuration parameters to avoid hardcoding
    throughout the application. Supports environment variable overrides for
    flexible deployment across different environments.

    Attributes:
        default_model: Default LLM model to use for analysis.
        temperature: Sampling temperature for LLM responses (0.0-1.0).
        max_tokens: Maximum tokens per LLM request.
        timeout: Request timeout in seconds for LLM calls.
        api_base: Base URL for LLM API endpoint.
    """

    default_model: str = os.getenv("LLM_DEFAULT_MODEL", "gpt-4o-mini")
    temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.3"))
    max_tokens: int = int(os.getenv("LLM_MAX_TOKENS", "1000"))
    timeout: float = float(os.getenv("LLM_TIMEOUT", "30.0"))
    api_base: Optional[str] = os.getenv(
        "LLM_API_BASE"
    )  # None means use provider default


class ServerConfig(BaseModel):
    """Configuration for a single MCP server instance.

    Defines all necessary parameters for connecting to and managing an MCP server.
    This configuration supports multiple connection methods and allows for
    flexible server deployment options.

    Attributes:
        name: Human-readable server name for identification.
        script_path: Relative path to the Python script implementing the server.
        connection_method: Protocol to use for server communication.
        host: Network host for SSE/HTTP connections (default: localhost).
        port: Network port for SSE/HTTP connections.
    """

    name: str
    script_path: str
    connection_method: ConnectionMethod = ConnectionMethod.STDIO
    host: str = os.getenv("MCP_HOST", "127.0.0.1")
    port: int = int(os.getenv("MCP_PORT_BASE", "8000"))


class MCPConfig(BaseModel):
    """Main configuration for the MCP server ecosystem.

    Central configuration object that defines all MCP servers used by the
    support agent. This configuration establishes the default connection
    method and individual server settings for the three core services:
    classification, knowledge retrieval, and health monitoring.

    The configuration is designed to be easily extended with additional
    servers or modified for different deployment environments.

    Attributes:
        connection_method: Default connection method for all servers.
        servers: Dictionary mapping server roles to their configurations.
        knowledge_search_depth: Maximum levels of recursive knowledge enhancement (1 = no recursion).
        llm: LLM provider configuration settings.
        default_team: Default team for classification requests.
    """

    connection_method: ConnectionMethod = ConnectionMethod.STDIO
    knowledge_search_depth: int = 1
    llm: LLMConfig = LLMConfig()
    default_team: str = os.getenv("DEFAULT_TEAM", "atrs")
    servers: Dict[str, ServerConfig] = {
        "classification": ServerConfig(
            name="ClassificationServer",
            script_path="mcp_servers/classification_server.py",
            port=8001,
        ),
        "knowledge": ServerConfig(
            name="KnowledgeRetrievalServer",
            script_path="mcp_servers/knowledge_server.py",
            port=8002,
        ),
        "health": ServerConfig(
            name="HealthMonitoringServer",
            script_path="mcp_servers/health_server.py",
            port=8003,
        ),
    }

    def get_server_config(self, server_name: str) -> ServerConfig:
        """Get configuration for a specific server by name.

        Args:
            server_name: Name of the server (e.g., 'classification', 'knowledge').

        Returns:
            ServerConfig: Configuration object for the requested server.

        Raises:
            KeyError: If the server name is not found in the configuration.
        """
        return self.servers[server_name]
