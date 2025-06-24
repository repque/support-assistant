"""Classification & Triage MCP Server using Native MCP SDK with Sampling Support.

This module implements an MCP (Model Context Protocol) server that provides intelligent
classification and triaging of support requests. It uses LLM-based classification to
categorize requests by team and issue type, enabling automated routing
and workflow suggestions for support systems.

The server supports multiple teams with custom category configurations and provides
both synchronous classification (with client-provided LLM responses) and asynchronous
classification (using MCP sampling).
"""

import argparse
import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from mcp.server import NotificationOptions, Server
from mcp.server.models import InitializationOptions
from mcp.server.session import ServerSession
from mcp.types import (
    CreateMessageRequest,
    CreateMessageRequestParams,
    CreateMessageResult,
    TextContent,
    Tool,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("classification-server")


class TeamConfigManager:
    """Manages team-specific category configurations for support request classification.

    This class is responsible for loading and managing team-specific configuration files
    that define categories, subcategories, workflows, and training examples for each
    support team. It provides a centralized interface for accessing team configurations
    used in the classification process.

    The configuration files are expected to be JSON files in a 'categories' directory,
    with each file named after the team it represents (e.g., 'atrs.json', 'core.json').

    Attributes:
        categories_dir: Path to the directory containing team configuration files.
        team_configs: Dictionary mapping team names to their configuration data.
    """

    def __init__(self, categories_dir: Optional[Path] = None):
        self.categories_dir = categories_dir or Path(__file__).parent / "categories"
        self.team_configs = {}
        self._load_team_configs()

    def _load_team_configs(self):
        """Load all team configuration files from the categories directory.

        This method scans the categories directory for JSON files and loads each one
        as a team configuration. Each configuration file should contain team info,
        categories with subcategories and workflows, and training examples.

        The method gracefully handles missing directories and malformed files,
        logging warnings or errors as appropriate without failing the entire
        initialization process.
        """
        if not self.categories_dir.exists():
            logger.warning(f"Categories directory not found: {self.categories_dir}")
            return

        for config_file in self.categories_dir.glob("*.json"):
            try:
                with open(config_file, "r") as f:
                    config = json.load(f)

                team_name = config_file.stem
                self.team_configs[team_name] = config
                logger.info(f"Loaded configuration for team: {team_name}")

            except Exception as e:
                logger.error(f"Failed to load team config {config_file}: {e}")

    def get_available_teams(self) -> List[str]:
        """Get list of available team names for classification.

        Returns:
            List[str]: Sorted list of team names that have valid configurations loaded.
                      These names correspond to the stem of configuration filenames.
        """
        return sorted(list(self.team_configs.keys()))

    def get_team_info(self, team: str) -> Dict[str, Any]:
        """Get detailed information about a specific team.

        Args:
            team: The team identifier (e.g., 'atrs', 'core').

        Returns:
            Dict[str, Any]: Team information including name and description.

        Raises:
            ValueError: If the specified team is not found in loaded configurations.
        """
        if team not in self.team_configs:
            raise ValueError(f"Unknown team: {team}")
        return self.team_configs[team]["team_info"]

    def get_team_categories(self, team: str) -> Dict[str, Any]:
        """Get category definitions for a specific team.

        Args:
            team: The team identifier (e.g., 'atrs', 'core').

        Returns:
            Dict[str, Any]: Dictionary mapping category names to their definitions,
                          including descriptions, subcategories, and workflows.

        Raises:
            ValueError: If the specified team is not found in loaded configurations.
        """
        if team not in self.team_configs:
            raise ValueError(f"Unknown team: {team}")
        return self.team_configs[team]["categories"]

    def get_team_examples(self, team: str) -> List[Dict[str, Any]]:
        """Get training examples for few-shot learning in classification.

        Training examples are used to improve classification accuracy by providing
        the LLM with examples of properly classified requests for the specific team.

        Args:
            team: The team identifier (e.g., 'atrs', 'core').

        Returns:
            List[Dict[str, Any]]: List of training examples, each containing
                                 text, category, subcategory, and reasoning.

        Raises:
            ValueError: If the specified team is not found in loaded configurations.
        """
        if team not in self.team_configs:
            raise ValueError(f"Unknown team: {team}")
        return self.team_configs[team].get("training_examples", [])


def _build_classification_prompt(
    user_request: str, team_config: TeamConfigManager, team: str
) -> str:
    """Build team-specific classification prompt following Anthropic cookbook best practices.

    Constructs a detailed prompt for the LLM to classify support requests according to
    team-specific categories and guidelines. The prompt includes system instructions,
    category definitions, few-shot examples, and output format specifications.

    The prompt is designed to handle vague requests appropriately by instructing the
    LLM to assign low confidence scores when requests lack specific details.

    Args:
        user_request: The support request text to be classified.
        team_config: TeamConfigManager instance containing team configurations.
        team: The team identifier for which to build the prompt.

    Returns:
        str: Complete prompt ready for LLM processing, including the user request.
    """

    team_info = team_config.get_team_info(team)
    categories = team_config.get_team_categories(team)
    examples = team_config.get_team_examples(team)

    # System prompt with clear instructions
    system_prompt = f"""You are an expert at classifying support requests for the {team_info['name']} team.

Team: {team_info['name']}
Domain: {team_info['description']}

Your task is to classify the following support request into one of these categories:

CATEGORIES:
"""

    # Add category definitions
    for category, info in categories.items():
        subcats = ", ".join(info.get("subcategories", []))
        system_prompt += f"- {category}: {info['description']}\n"
        if subcats:
            system_prompt += f"  Subcategories: {subcats}\n"

    system_prompt += """
INSTRUCTIONS:
1. Read the support request carefully and analyze the BUSINESS CONTEXT
2. Determine which category best matches the request based on the category descriptions above
3. Choose the most appropriate subcategory within that category
4. Provide clear reasoning for your classification decision
5. IMPORTANT: If the request is vague, unclear, or lacks specific details (e.g., "something is broken", "I need help", "there's an issue"), you MUST assign a LOW confidence score (0.3 or less)

Use the category descriptions and examples below to guide your classification. Each category has specific subcategories and workflows defined by the team.

EXAMPLES:
"""

    # Add few-shot examples
    for example in examples:
        system_prompt += f"""
Request: "{example['text']}"
Category: {example['category']}
Subcategory: {example['subcategory']}
Reasoning: {example['reasoning']}
"""

    system_prompt += """
OUTPUT FORMAT:
Respond with a JSON object in this exact format:
{
  "category": "category_name",
  "subcategory": "subcategory_name", 
  "confidence": 0.85,  // MUST be 0.3 or less for vague requests!
  "reasoning": "Brief explanation of classification decision"
}

CONFIDENCE SCORING GUIDELINES:
- 0.9-1.0: Very specific request with clear technical details
- 0.7-0.8: Clear request with good context
- 0.5-0.6: Somewhat unclear but identifiable issue
- 0.3 or less: Vague, unclear, or overly general request

Now classify this request:
"""

    return system_prompt + f'"{user_request}"'


class ClassificationServer:
    """MCP Server for classifying support requests using LLM sampling.

    This server implements the Model Context Protocol (MCP) to provide intelligent
    classification services for support requests. It exposes tools for classifying
    requests, retrieving team information, and managing category configurations.

    The server supports two modes of operation:
    1. Client-provided LLM responses: The client calls the LLM and provides the response
    2. Server-side sampling: The server requests LLM sampling via MCP (may cause deadlock)

    Attributes:
        server: MCP Server instance handling protocol communication.
        session: Optional ServerSession for managing client connections.
        team_config: TeamConfigManager instance for accessing team configurations.
    """

    def __init__(self):
        self.server = Server("classification-server")
        self.session: Optional[ServerSession] = None
        self.team_config = TeamConfigManager()
        self._setup_handlers()

    def _setup_handlers(self):
        """Set up MCP server handlers for tools and requests.

        Registers all available tools with the MCP server and defines their
        input schemas and handlers. This includes tools for classification,
        prompt generation, team listing, and category information retrieval.
        """

        @self.server.list_tools()
        async def handle_list_tools() -> List[Tool]:
            """List all available tools provided by this MCP server.

            Returns:
                List[Tool]: List of tool definitions including names, descriptions,
                           and input schemas for all classification-related tools.
            """
            return [
                Tool(
                    name="classify_support_request",
                    description="Classify a support request into team-specific categories",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "user_request": {
                                "type": "string",
                                "description": "The support request text to classify",
                            },
                            "team": {
                                "type": "string",
                                "description": "The team name to classify for (e.g., 'atrs', 'core')",
                                "enum": self.team_config.get_available_teams(),
                            },
                            "llm_response": {
                                "type": "string",
                                "description": "Optional: LLM response to classification prompt (client-provided)",
                            },
                        },
                        "required": ["user_request", "team"],
                    },
                ),
                Tool(
                    name="get_classification_prompt",
                    description="Get team-specific classification prompt for a support request",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "user_request": {
                                "type": "string",
                                "description": "The support request text to build prompt for",
                            },
                            "team": {
                                "type": "string",
                                "description": "The team name to build prompt for (e.g., 'atrs', 'core')",
                                "enum": self.team_config.get_available_teams(),
                            },
                        },
                        "required": ["user_request", "team"],
                    },
                ),
                Tool(
                    name="list_teams",
                    description="Get list of available support teams",
                    inputSchema={"type": "object", "properties": {}},
                ),
                Tool(
                    name="get_team_info",
                    description="Get detailed information about a specific team",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "team": {
                                "type": "string",
                                "description": "The team name to get info for",
                                "enum": self.team_config.get_available_teams(),
                            }
                        },
                        "required": ["team"],
                    },
                ),
                Tool(
                    name="get_team_categories",
                    description="Get available categories for a specific team",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "team": {
                                "type": "string",
                                "description": "The team name to get categories for",
                                "enum": self.team_config.get_available_teams(),
                            }
                        },
                        "required": ["team"],
                    },
                ),
                Tool(
                    name="get_category_details",
                    description="Get detailed information about a specific category for a team",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "team": {
                                "type": "string",
                                "description": "The team name",
                                "enum": self.team_config.get_available_teams(),
                            },
                            "category": {
                                "type": "string",
                                "description": "The category name to get details for",
                            },
                        },
                        "required": ["team", "category"],
                    },
                ),
            ]

        @self.server.call_tool()
        async def handle_call_tool(
            name: str, arguments: Dict[str, Any]
        ) -> List[TextContent]:
            """Handle incoming tool calls from MCP clients.

            Routes tool calls to appropriate handlers based on the tool name and
            processes the arguments to generate responses.

            Args:
                name: The name of the tool being called.
                arguments: Dictionary of arguments passed to the tool.

            Returns:
                List[TextContent]: Response content formatted as MCP TextContent.

            Raises:
                ValueError: If required arguments are missing or invalid.
            """

            if name == "classify_support_request":
                user_request = arguments.get("user_request", "")
                team = arguments.get("team", "")
                llm_response = arguments.get("llm_response")

                if not team:
                    raise ValueError("Team parameter is required")

                if llm_response:
                    # Client provided LLM response, parse it
                    result = self._parse_llm_response(llm_response, user_request, team)
                else:
                    # No LLM response, use server-side sampling (will deadlock)
                    result = await self._classify_support_request(user_request, team)

                return [TextContent(type="text", text=json.dumps(result, indent=2))]

            elif name == "get_classification_prompt":
                user_request = arguments.get("user_request", "")
                team = arguments.get("team", "")

                if not team:
                    raise ValueError("Team parameter is required")

                prompt = _build_classification_prompt(
                    user_request, self.team_config, team
                )
                return [TextContent(type="text", text=prompt)]

            elif name == "list_teams":
                teams_info = []
                for team_name in self.team_config.get_available_teams():
                    team_info = self.team_config.get_team_info(team_name)
                    teams_info.append(
                        {
                            "name": team_name,
                            "display_name": team_info["name"],
                            "description": team_info["description"],
                        }
                    )
                return [TextContent(type="text", text=json.dumps(teams_info, indent=2))]

            elif name == "get_team_info":
                team = arguments.get("team", "")
                if not team:
                    raise ValueError("Team parameter is required")

                try:
                    team_info = self.team_config.get_team_info(team)
                    return [
                        TextContent(type="text", text=json.dumps(team_info, indent=2))
                    ]
                except ValueError as e:
                    error_result = {"error": str(e)}
                    return [
                        TextContent(
                            type="text", text=json.dumps(error_result, indent=2)
                        )
                    ]

            elif name == "get_team_categories":
                team = arguments.get("team", "")
                if not team:
                    raise ValueError("Team parameter is required")

                try:
                    categories = self.team_config.get_team_categories(team)
                    return [
                        TextContent(type="text", text=json.dumps(categories, indent=2))
                    ]
                except ValueError as e:
                    error_result = {"error": str(e)}
                    return [
                        TextContent(
                            type="text", text=json.dumps(error_result, indent=2)
                        )
                    ]

            elif name == "get_category_details":
                team = arguments.get("team", "")
                category = arguments.get("category", "")

                if not team:
                    raise ValueError("Team parameter is required")
                if not category:
                    raise ValueError("Category parameter is required")

                try:
                    categories = self.team_config.get_team_categories(team)
                    if category not in categories:
                        error_result = {
                            "error": f"Category '{category}' not found for team '{team}'"
                        }
                        return [
                            TextContent(
                                type="text", text=json.dumps(error_result, indent=2)
                            )
                        ]

                    return [
                        TextContent(
                            type="text", text=json.dumps(categories[category], indent=2)
                        )
                    ]
                except ValueError as e:
                    error_result = {"error": str(e)}
                    return [
                        TextContent(
                            type="text", text=json.dumps(error_result, indent=2)
                        )
                    ]

            else:
                raise ValueError(f"Unknown tool: {name}")

    def _parse_llm_response(
        self, llm_response: str, user_request: str, team: str
    ) -> Dict[str, Any]:
        """Parse LLM response to extract classification results.

        Handles various response formats including raw JSON and markdown code blocks.
        Validates the parsed classification and enriches it with workflow suggestions
        based on the team's category configurations.

        Args:
            llm_response: Raw response text from the LLM.
            user_request: Original user request (for context/logging).
            team: Team identifier for category validation.

        Returns:
            Dict[str, Any]: Parsed classification result with category, subcategory,
                          confidence, workflow, and reasoning.

        Raises:
            RuntimeError: If the LLM response cannot be parsed as valid JSON or
                         if classification parsing fails.
        """
        try:
            # Extract JSON from the response (handle potential markdown code blocks)
            response_text = llm_response.strip()

            if "```json" in response_text:
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                response_text = response_text[json_start:json_end].strip()
            elif "```" in response_text:
                json_start = response_text.find("```") + 3
                json_end = response_text.find("```", json_start)
                response_text = response_text[json_start:json_end].strip()

            # Parse the classification result
            classification = json.loads(response_text)

            # Validate and add suggested_workflow based on category
            category = classification.get("category", "query")
            team_categories = self.team_config.get_team_categories(team)

            if category in team_categories:
                classification["suggested_workflow"] = team_categories[category].get(
                    "workflow", "technical_guidance"
                )
            else:
                classification["suggested_workflow"] = "technical_guidance"

            # Ensure all required fields are present
            result = {
                "category": classification.get("category", "query"),
                "subcategory": classification.get("subcategory", "unknown"),
                "confidence": float(classification.get("confidence", 0.5)),
                "suggested_workflow": classification["suggested_workflow"],
                "reasoning": classification.get(
                    "reasoning", "Classification based on LLM analysis"
                ),
                "team": team,
            }

            logger.info(f"Parsed classification result for team {team}: {result}")
            return result

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {str(e)}")
            logger.error(f"LLM response was: {llm_response}")
            raise RuntimeError(f"LLM returned invalid JSON: {str(e)}")
        except Exception as e:
            logger.error(f"Classification parsing failed: {str(e)}")
            raise RuntimeError(f"Classification error: {str(e)}")

    async def _classify_support_request(
        self, user_request: str, team: str
    ) -> Dict[str, Any]:
        """Classify a support request using server-side LLM sampling via MCP.

        This method uses MCP's sampling capability to request classification from
        an LLM. Note that this approach may cause deadlock in certain MCP
        implementations where the server and client share the same process.

        Args:
            user_request: The support request text to classify.
            team: Team identifier for team-specific classification.

        Returns:
            Dict[str, Any]: Classification result with category, subcategory,
                          confidence, workflow, and reasoning.

        Raises:
            RuntimeError: If server session is not initialized, LLM returns
                         invalid JSON, or classification fails.
        """

        if not self.session:
            raise RuntimeError("Server session not initialized")

        # Build the classification prompt
        prompt = _build_classification_prompt(user_request, self.team_config, team)

        try:
            # Create the sampling request
            sampling_request = CreateMessageRequest(
                method="sampling/createMessage",
                params=CreateMessageRequestParams(
                    messages=[
                        {
                            "role": "user",
                            "content": TextContent(type="text", text=prompt),
                        }
                    ],
                    maxTokens=1000,
                    model="claude-3-5-sonnet-20241022",
                    temperature=0.1,
                ),
            )

            # Send sampling request to client
            logger.info(
                f"Sending sampling request for classification: {user_request[:100]}..."
            )
            response = await self.session.send_request(
                sampling_request, CreateMessageResult
            )

            if not isinstance(response, CreateMessageResult):
                raise ValueError(f"Unexpected response type: {type(response)}")

            # Extract the response content
            if not response.content or not response.content.text:
                raise ValueError("Empty response from LLM")

            response_text = response.content.text.strip()
            logger.info(f"Received LLM response: {response_text[:200]}...")

            # Extract JSON from the response (handle potential markdown code blocks)
            if "```json" in response_text:
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                response_text = response_text[json_start:json_end].strip()
            elif "```" in response_text:
                json_start = response_text.find("```") + 3
                json_end = response_text.find("```", json_start)
                response_text = response_text[json_start:json_end].strip()

            # Parse the classification result
            classification = json.loads(response_text)

            # Validate and add suggested_workflow based on category
            category = classification.get("category", "query")
            if category in CATEGORIES:
                classification["suggested_workflow"] = CATEGORIES[category].get(
                    "workflow", "technical_guidance"
                )
            else:
                classification["suggested_workflow"] = "technical_guidance"

            # Ensure all required fields are present
            result = {
                "category": classification.get("category", "query"),
                "subcategory": classification.get("subcategory", "unknown"),
                "confidence": float(classification.get("confidence", 0.5)),
                "suggested_workflow": classification["suggested_workflow"],
                "reasoning": classification.get(
                    "reasoning", "Classification based on LLM analysis"
                ),
            }

            logger.info(f"Classification result: {result}")
            return result

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {str(e)}")
            raise RuntimeError(f"LLM returned invalid JSON: {str(e)}")
        except Exception as e:
            logger.error(f"Classification failed: {str(e)}")
            raise RuntimeError(f"Classification error: {str(e)}")


async def main():
    """Main entry point for the classification server.

    Handles command-line argument parsing, logging configuration, and server
    initialization. Supports both stdio and SSE connection methods for MCP
    communication.

    The server can be run in two modes:
    - stdio: Standard input/output communication (default)
    - sse: Server-Sent Events for HTTP-based communication
    """

    parser = argparse.ArgumentParser(description="Classification MCP Server")
    parser.add_argument(
        "--connection",
        choices=["stdio", "sse"],
        default="stdio",
        help="Connection method (default: stdio)",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host for SSE connections")
    parser.add_argument(
        "--port", type=int, default=8001, help="Port for SSE connections"
    )

    args = parser.parse_args()

    # Set logging level based on environment variable
    import os

    log_level = os.getenv("MCP_LOG_LEVEL", "INFO")
    logging.getLogger().setLevel(getattr(logging, log_level))

    # Create and run server
    classification_server = ClassificationServer()

    if args.connection == "stdio":
        from mcp.server.stdio import stdio_server

        async with stdio_server() as (read_stream, write_stream):
            init_options = InitializationOptions(
                server_name="classification-server",
                server_version="1.0.0",
                capabilities=classification_server.server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            )

            # Create session and store it in the server
            classification_server.session = ServerSession(
                read_stream, write_stream, init_options
            )

            await classification_server.server.run(
                read_stream, write_stream, init_options
            )
    elif args.connection == "sse":
        from mcp.server.sse import SseServerTransport

        transport = SseServerTransport(f"http://{args.host}:{args.port}/sse")
        await classification_server.server.run(transport, InitializationOptions())


if __name__ == "__main__":
    asyncio.run(main())
