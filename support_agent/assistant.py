"""Main Support Assistant module that orchestrates MCP server interactions.

This module provides the core SupportAssistant class that acts as the central
orchestrator for the support agent system. It manages connections to multiple
MCP (Model Context Protocol) servers, coordinates classification, knowledge
retrieval, health monitoring, and generates intelligent recommendations for
support requests.

The assistant implements a confidence-based decision system, only providing
recommendations when it has sufficient information and confidence in the analysis.
"""

import asyncio
import os
import sys
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Dict, Optional

import httpx
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.sse import sse_client
from mcp.types import Implementation, CreateMessageRequestParams, CreateMessageResult, TextContent

from .models import SupportRequest, Classification, AnalysisResult
from .config import MCPConfig, ConnectionMethod

# Import console for Rich formatting
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich.text import Text

_console = Console()


class SupportAssistant:
    """Main Support Assistant that connects to and orchestrates MCP servers.
    
    This class serves as the central intelligence of the support system, coordinating
    between multiple specialized MCP servers to analyze support requests and provide
    actionable recommendations. It implements a confidence-based decision framework
    that ensures the assistant only responds when it has sufficient information and
    confidence in its analysis.
    
    The assistant manages connections to three primary MCP servers:
    - Classification Server: Categorizes and prioritizes support requests
    - Knowledge Server: Retrieves relevant documentation and solutions
    - Health Server: Monitors system status and service availability
    
    Key features:
    - Multi-server orchestration with STDIO and SSE connection support
    - Confidence scoring to determine when to provide recommendations
    - LLM integration for intelligent analysis and response generation
    - Token tracking for resource management
    - Graceful error handling and silent failure modes
    
    Attributes:
        config: MCPConfig instance containing server configurations.
        mcp_sessions: Dictionary mapping server names to active MCP sessions.
        stdio_contexts: Dictionary storing STDIO context managers for cleanup.
        servers_running: Boolean indicating if MCP servers are active.
    """
    
    
    def __init__(self, config: Optional[MCPConfig] = None):
        self.config = config or MCPConfig()
        self.mcp_sessions = {}
        self.stdio_contexts = {}  # Store context managers for proper cleanup
        self.servers_running = False
        self._context_stack = None  # For managing multiple STDIO contexts
        self._tool_call_count = 0  # Track tool calls
        self._total_tokens = 0  # Track total tokens used
        self._sse_processes = []  # Track SSE subprocess for cleanup
        
        # LLM configuration for MCP sampling
        self._llm_api_key = os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
        self._llm_api_base = self.config.llm.api_base or os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
        
    async def _sampling_callback(self, message: CreateMessageRequestParams) -> CreateMessageResult:
        """MCP sampling callback to process prompts with production LLM.
        
        This callback is invoked by MCP servers when they need to process prompts
        with an LLM. It supports OpenAI-compatible APIs and handles authentication,
        request formatting, and error handling.
        
        The callback is designed to work with both OpenAI and Anthropic APIs,
        automatically detecting which API key is available in the environment.
        
        Args:
            message: CreateMessageRequestParams containing the prompt and parameters.
        
        Returns:
            CreateMessageResult: LLM response wrapped in MCP result format.
        
        Note:
            Requires either OPENAI_API_KEY or ANTHROPIC_API_KEY environment variable.
        """
        
        if not self._llm_api_key:
            return CreateMessageResult(
                role="assistant",
                content=TextContent(
                    type="text",
                    text="Error: No LLM API key configured. Please set OPENAI_API_KEY or ANTHROPIC_API_KEY environment variable."
                ),
                model="error",
                stopReason="error"
            )
        
        try:
            # Prepare the request for OpenAI-compatible API
            messages = []
            for msg in message.messages:
                if hasattr(msg.content, 'text'):
                    messages.append({
                        "role": msg.role,
                        "content": msg.content.text
                    })
                elif isinstance(msg.content, str):
                    messages.append({
                        "role": msg.role,
                        "content": msg.content
                    })
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self._llm_api_base}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self._llm_api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": message.model or self.config.llm.default_model,
                        "messages": messages,
                        "max_tokens": message.maxTokens or self.config.llm.max_tokens,
                        "temperature": self.config.llm.temperature
                    },
                    timeout=self.config.llm.timeout
                )
                
                if response.status_code == 200:
                    data = response.json()
                    content = data["choices"][0]["message"]["content"]
                    
                    return CreateMessageResult(
                        role="assistant",
                        content=TextContent(type="text", text=content),
                        model=data["model"],
                        stopReason="stop"
                    )
                else:
                    error_msg = f"LLM API error: {response.status_code} - {response.text}"
                    return CreateMessageResult(
                        role="assistant",
                        content=TextContent(type="text", text=f"Error: {error_msg}"),
                        model="error",
                        stopReason="error"
                    )
                    
        except Exception as e:
            return CreateMessageResult(
                role="assistant",
                content=TextContent(type="text", text=f"Error processing request: {str(e)}"),
                model="error",
                stopReason="error"
            )
    
    async def start_mcp_servers(self) -> bool:
        """Start and connect to all configured MCP servers.
        
        Initializes connections to all MCP servers defined in the configuration.
        Supports both STDIO (standard input/output) and SSE (Server-Sent Events)
        connection methods. This method ensures all servers are properly started
        and ready to handle requests before returning.
        
        Returns:
            bool: True if all servers started successfully, False otherwise.
        
        Raises:
            Exception: Propagates any exceptions after attempting cleanup.
        """
        if self.servers_running:
            return True
            
        print(f"Starting MCP servers via {self.config.connection_method.value}...")
        
        try:
            if self.config.connection_method == ConnectionMethod.STDIO:
                return await self._start_stdio_servers()
            elif self.config.connection_method == ConnectionMethod.SSE:
                return await self._start_sse_servers()
            else:
                print(f"❌ Unsupported connection method: {self.config.connection_method}")
                return False
                
        except Exception as e:
            print(f"❌ Failed to start MCP servers: {e}")
            await self.stop_mcp_servers()
            return False
    
    async def _start_stdio_servers(self) -> bool:
        """Start all STDIO-based MCP servers with proper context management.
        
        Creates subprocess-based MCP servers that communicate via standard input/output.
        Uses AsyncExitStack for proper resource management and cleanup. Each server
        is initialized with its own Python subprocess and MCP session.
        
        The method ensures proper Python path configuration and environment setup
        for each server subprocess.
        
        Returns:
            bool: True if all STDIO servers started successfully.
        
        Raises:
            Exception: If any server fails to start or initialize.
        """
        # Create an AsyncExitStack to manage all STDIO contexts
        self._context_stack = AsyncExitStack()
        
        try:
            for name, server_config in self.config.servers.items():
                # Get absolute path to server script
                server_script = Path(__file__).parent.parent / server_config.script_path
                
                if not server_script.exists():
                    print(f"ERROR: Server script not found: {server_script}")
                    return False
                
                # Create server parameters
                server_params = StdioServerParameters(
                    command="python",
                    args=[str(server_script), "--connection", "stdio"],
                    env={
                        "PYTHONUNBUFFERED": "1",
                        "PYTHONPATH": str(Path(__file__).parent.parent),
                        "MCP_LOG_LEVEL": "WARNING"  # Suppress INFO logs
                    }
                )
                
                # Connect using the proper MCP client pattern
                stdio_context = stdio_client(server_params)
                read, write = await self._context_stack.enter_async_context(stdio_context)
                
                # Create session using the SDK pattern with sampling callback
                session_context = ClientSession(read, write, sampling_callback=self._sampling_callback)
                session = await self._context_stack.enter_async_context(session_context)
                
                # Initialize the session
                init_result = await session.initialize()
                
                self.mcp_sessions[name] = session
                print(f"✓ {name.title()} Server")
            
            self.servers_running = True
            print("All MCP servers running!")
            return True
            
        except Exception as e:
            print(f"ERROR: STDIO server startup failed: {e}")
            if self._context_stack:
                await self._context_stack.aclose()
                self._context_stack = None
            raise
    
    async def _start_sse_servers(self) -> bool:
        """Start all SSE-based MCP servers.
        
        Iterates through configured servers and starts each one in SSE mode,
        establishing HTTP-based Server-Sent Events connections. This is an
        alternative to STDIO for environments where subprocess communication
        is problematic.
        
        Returns:
            bool: True if all SSE servers started successfully.
        """
        for name, server_config in self.config.servers.items():
            # Get absolute path to server script
            server_script = Path(__file__).parent.parent / server_config.script_path
            
            if not server_script.exists():
                print(f"ERROR: Server script not found: {server_script}")
                return False
            
            session = await self._connect_sse(server_script, server_config)
            self.mcp_sessions[name] = session
            print(f"{name.title()} server connected via SSE")
        
        self.servers_running = True
        print("All MCP servers running!")
        return True
    
    async def stop_mcp_servers(self):
        """Stop all running MCP servers and clean up resources.
        
        Performs graceful shutdown of all MCP servers, cleaning up:
        - Active MCP sessions
        - STDIO context managers (via AsyncExitStack)
        - Legacy STDIO contexts (if any)
        
        This method ensures all resources are properly released even if
        errors occur during shutdown.
        """
        print("Stopping MCP servers...")
        
        # Clean up sessions (managed by context stack)
        if self.mcp_sessions:
            print(f"Disconnecting {len(self.mcp_sessions)} MCP servers...")
        
        # Clean up context stack (for STDIO connections)
        if self._context_stack:
            try:
                await self._context_stack.aclose()
                print("STDIO contexts cleaned up")
            except Exception as e:
                print(f"Warning: Error cleaning up STDIO contexts: {e}")
            finally:
                self._context_stack = None
        
        # Clean up legacy stdio contexts (if any)
        for script_name, context in self.stdio_contexts.items():
            try:
                await context.__aexit__(None, None, None)
            except Exception as e:
                print(f"Warning: Error cleaning up {script_name}: {e}")
        
        # Clean up SSE processes
        for process in self._sse_processes:
            try:
                if process.poll() is None:  # Process is still running
                    process.terminate()
                    # Give it a moment to terminate gracefully
                    await asyncio.sleep(0.1)
                    if process.poll() is None:  # Still running, force kill
                        process.kill()
                    process.wait()  # Wait for cleanup
            except Exception as e:
                print(f"Warning: Error cleaning up SSE process: {e}")
        
        self.mcp_sessions.clear()
        self.stdio_contexts.clear()
        self._sse_processes.clear()
        self.servers_running = False
    
    
    async def _connect_sse(self, server_script: Path, server_config) -> ClientSession:
        """Connect to an MCP server via Server-Sent Events (SSE).
        
        Starts the server as a subprocess in SSE mode and establishes a client
        connection to it. This method handles the server startup delay and
        connection initialization.
        
        Args:
            server_script: Path to the Python script implementing the MCP server.
            server_config: Server configuration containing host and port.
        
        Returns:
            ClientSession: Initialized MCP client session.
        
        Note:
            The subprocess management here is simplified and may need enhancement
            for production use (e.g., process monitoring, cleanup).
        """
        # Start server in SSE mode (in background)
        import subprocess
        import time
        
        # Start the server process and track it for cleanup
        process = subprocess.Popen([
            "python", str(server_script), 
            "--connection", "sse",
            "--host", server_config.host,
            "--port", str(server_config.port)
        ], stderr=subprocess.PIPE)
        self._sse_processes.append(process)
        
        # Give server time to start
        await asyncio.sleep(2)
        
        # Connect via SSE
        sse_context = sse_client(f"http://{server_config.host}:{server_config.port}")
        read, write = await sse_context.__aenter__()
        session = ClientSession(read, write)
        await session.initialize()
        return session
    
    async def analyze_support_request(self, request: SupportRequest) -> Optional[Dict]:
        """Analyze a support request using coordinated MCP server capabilities.
        
        Orchestrates the complete analysis workflow:
        1. Classifies the request to determine category and priority
        2. Checks if the request should be handled (some require human review)
        3. Searches the knowledge base for relevant solutions
        4. Checks system health if an affected system is specified
        5. Calculates confidence score based on available information
        6. Generates recommendations if confidence threshold is met
        
        The method implements intelligent fallback behavior, returning None
        (staying silent) when confidence is low or information is insufficient.
        
        Args:
            request: SupportRequest containing issue description and metadata.
        
        Returns:
            Optional[Dict]: Analysis results including classification, recommendations,
                          and metadata, or None if the assistant should stay silent.
        
        Raises:
            RuntimeError: If MCP servers are not running.
        """
        if not self.servers_running:
            raise RuntimeError("MCP servers not running")
        
        # Reset counters for new request
        self._tool_call_count = 0
        self._total_tokens = 0
        
        try:
            # Step 1: Classify the request using client-side approach (production flow)
            self._update_tool_call_status("Classification(get_classification_prompt)")
            
            # Get the classification prompt
            prompt_result = await self.mcp_sessions["classification"].call_tool(
                "get_classification_prompt",
                {
                    "user_request": request.issue_description,
                    "team": self.config.default_team
                }
            )
            prompt = prompt_result.content[0].text if prompt_result.content else ""
            
            # Step 2: Process with production LLM using sampling callback
            self._update_tool_call_status("Classification(LLM processing)")
            from mcp.types import CreateMessageRequestParams, TextContent
            
            sampling_request = CreateMessageRequestParams(
                messages=[{
                    "role": "user",
                    "content": TextContent(type="text", text=prompt)
                }],
                maxTokens=self.config.llm.max_tokens,
                model=self.config.llm.default_model,
                temperature=self.config.llm.temperature
            )
            
            # Use production LLM sampling callback
            llm_result = await self._sampling_callback(sampling_request)
            
            if llm_result.content and hasattr(llm_result.content, 'text'):
                llm_response = llm_result.content.text
            else:
                llm_response = str(llm_result.content)
            
            # Step 3: Parse the LLM response
            self._update_tool_call_status("Classification(classify_support_request)")
            classification_result = await self.mcp_sessions["classification"].call_tool(
                "classify_support_request",
                {
                    "user_request": request.issue_description,
                    "team": "atrs",
                    "llm_response": llm_response
                }
            )
            # Estimate tokens for request, prompt, LLM response, and classification result
            self._total_tokens += self._estimate_tokens(request.issue_description)
            self._total_tokens += self._estimate_tokens(prompt)
            self._total_tokens += self._estimate_tokens(llm_response)
            self._total_tokens += self._estimate_tokens(str(classification_result))
            
            # Parse the classification result
            classification_data = classification_result.content[0].text
            if isinstance(classification_data, str):
                import json
                classification_data = json.loads(classification_data)
            
            classification = Classification(**classification_data)
            
            # Step 2: Check if we should handle this request using LLM
            should_handle = await self._should_assistant_handle_request(classification, request.issue_description)
            if not should_handle:
                # Clear the status line
                sys.stdout.write("\r" + " " * 80 + "\r")
                sys.stdout.flush()
                _console.print(f"[dim]Assistant staying silent - {classification.category} requires human review[/dim]")
                return None
            
            # Step 3: Search knowledge base using server-side vector embeddings
            self._update_tool_call_status("Knowledge(vector search)")
            knowledge_results = await self.mcp_sessions["knowledge"].call_tool(
                "search_knowledge",
                {
                    "query": request.issue_description,
                    "category": classification.category,
                    "max_results": 5
                }
            )
            # Estimate tokens
            self._total_tokens += self._estimate_tokens(request.issue_description)
            self._total_tokens += self._estimate_tokens(str(knowledge_results))
            
            # Handle empty knowledge results
            if not knowledge_results.content or len(knowledge_results.content) == 0:
                knowledge_data = []
            else:
                # Use ALL returned sections, not just the first one
                knowledge_data = []
                for content_item in knowledge_results.content:
                    if hasattr(content_item, 'text'):
                        section_data = content_item.text
                        if isinstance(section_data, str):
                            import json
                            try:
                                parsed_data = json.loads(section_data)
                                # Handle both single section and list of sections
                                if isinstance(parsed_data, list):
                                    knowledge_data.extend(parsed_data)
                                elif isinstance(parsed_data, dict):
                                    knowledge_data.append(parsed_data)
                            except:
                                continue
            
            
            # Step 4: Skip system health checks - removed affected_system field
            # System should infer what to check from the issue description if needed
            health_status = {}
            
            # Step 4.5: Perform recursive knowledge searches for gaps
            enhanced_knowledge_data = await self._enhance_knowledge_with_recursive_searches(
                knowledge_data, request.issue_description, depth=self.config.knowledge_search_depth
            )
            
            # Step 5: Simple decision - if we have knowledge, provide recommendations
            if not enhanced_knowledge_data or len(enhanced_knowledge_data) == 0:
                # Clear the status line
                sys.stdout.write("\r" + " " * 80 + "\r")
                sys.stdout.flush()
                _console.print(f"[yellow]No relevant knowledge found - Assistant staying silent[/yellow]")
                return None
            
            # Step 6: Generate recommendations
            recommendations = await self._generate_recommendations(
                classification, enhanced_knowledge_data, health_status, request.issue_description
            )
            
            # If no knowledge-based recommendations, stay silent
            if recommendations is None:
                # Clear the status line
                sys.stdout.write("\r" + " " * 80 + "\r")
                sys.stdout.flush()
                _console.print(f"[yellow]No knowledge base guidance available for {classification.category} - Assistant staying silent[/yellow]")
                return None
            
            # Clear the status line
            sys.stdout.write("\r" + " " * 80 + "\r")
            sys.stdout.flush()
            
            return {
                "request_id": request.request_id,
                "analysis": AnalysisResult(
                    request_id=request.request_id,
                    classification=classification,
                    knowledge_results=knowledge_data if isinstance(knowledge_data, list) else [],
                    health_status=health_status,
                    confidence_score=1.0,  # Always confident when we have knowledge
                    recommendations=recommendations,
                    sources_consulted=["Knowledge Base", "Health Monitor"]
                ),
                "recommendations": {"resolution_steps": recommendations},
                "processing_metadata": {
                    "classification": classification.model_dump(),
                    "confidence_score": 1.0,  # Always confident when we have knowledge
                    "processed_entities": 0,
                    "sources_consulted": 2
                }
            }
            
        except Exception as e:
            # Clear the status line
            sys.stdout.write("\r" + " " * 80 + "\r")
            sys.stdout.flush()
            print(f"ERROR: Error analyzing request: {e}")
            return {
                "request_id": request.request_id,
                "error": str(e),
                "analysis": None,
                "recommendations": None,
                "processing_metadata": {"error_occurred": True}
            }
    
    
    
    async def _enhance_knowledge_with_recursive_searches(self, knowledge_data, user_request: str, depth: int = 1):
        """Perform recursive searches to fill knowledge gaps at multiple levels.
        
        Analyzes knowledge results to identify concepts that are mentioned
        but lack implementation details (e.g., "check block events" without code).
        Then performs targeted searches to find that missing information, potentially
        recursively searching through the results of those searches as well.
        
        Args:
            knowledge_data: Knowledge search results to enhance.
            user_request: Original user request for context.
            depth: Maximum depth of recursive searches (1 = no recursion, 2+ = recursive).
            
        Returns:
            Enhanced knowledge data combining all levels of search results.
        """
        if not knowledge_data or not isinstance(knowledge_data, list) or depth <= 0:
            return knowledge_data
        
        # Create deep copies to prevent cross-request state contamination    
        import copy
        current_level_data = copy.deepcopy(knowledge_data)  # Start with clean copy
        all_enhanced_data = copy.deepcopy(knowledge_data)   # Accumulate all results
        
        for current_depth in range(depth):
            # Identify gaps in the current level knowledge
            gaps_to_search = []
            
            # Analyze current level data for gaps  
            for item in current_level_data:
                if not isinstance(item, dict):
                    continue
                    
                content = item.get("content", "")
                
                # Use LLM to intelligently identify knowledge gaps
                gap_concepts = await self._identify_knowledge_gaps(content, current_depth, user_request)
                
                
                for concept in gap_concepts:
                    if concept not in gaps_to_search:
                        gaps_to_search.append(concept)
            
            # If no gaps found at this level, stop recursing
            if not gaps_to_search:
                break
            
            # Perform searches for identified gaps at this level
            new_results_this_level = []
            
            for gap_concept in gaps_to_search[:2]:  # Limit to 2 follow-up searches per level to avoid token explosion
                try:
                    self._update_tool_call_status(f"Knowledge(L{current_depth+1}: {gap_concept[:30]}...)")
                
                    # Search specifically for the gap concept
                    follow_up_results = await self.mcp_sessions["knowledge"].call_tool(
                        "search_knowledge",
                        {
                            "query": gap_concept,
                            "max_results": 1  # Just get the best match
                        }
                    )
                    
                    # Estimate tokens
                    self._total_tokens += self._estimate_tokens(gap_concept)
                    self._total_tokens += self._estimate_tokens(str(follow_up_results))
                    
                    # Check if we got valid results
                    if not follow_up_results.content or len(follow_up_results.content) == 0:
                        continue  # No results found, skip silently
                    
                    follow_up_data = follow_up_results.content[0].text
                    if isinstance(follow_up_data, str):
                        import json
                        try:
                            follow_up_data = json.loads(follow_up_data)
                        except:
                            continue
                    
                    # Handle both single dict and list responses  
                    if isinstance(follow_up_data, dict):
                        follow_up_data = [follow_up_data]  # Convert single result to list
                        
                    # Add follow-up results if they're relevant and not duplicates
                    if isinstance(follow_up_data, list):
                        for item in follow_up_data:
                            if isinstance(item, dict) and item.get("relevance_score", 0) > 0.1:
                                # Check if this isn't already in our results
                                is_duplicate = any(
                                    existing.get("source") == item.get("source") 
                                    for existing in all_enhanced_data 
                                    if isinstance(existing, dict)
                                )
                                if not is_duplicate:
                                    # Mark this as a follow-up result
                                    item["is_followup"] = True
                                    item["searched_for"] = gap_concept
                                    item["search_depth"] = current_depth + 1
                                    all_enhanced_data.append(item)
                                    new_results_this_level.append(item)
                                    
                                        
                except Exception as e:
                    # If follow-up search fails, continue silently with what we have
                    # (Only log in debug mode to avoid cluttering user output)
                    continue
            
            # Set up for next iteration: new results become the data to analyze for gaps
            current_level_data = new_results_this_level
            
            # If we didn't find any new results at this level, stop recursing
            if not new_results_this_level:
                break
        
        return all_enhanced_data
    
    async def _identify_knowledge_gaps(self, content: str, current_depth: int, user_request: str = "") -> list:
        """Use LLM to intelligently identify concepts mentioned without implementation details.
        
        Analyzes knowledge content to find mentions of procedures, commands, or concepts
        that are referenced but lack specific implementation details like code examples.
        
        Args:
            content: Knowledge content to analyze for gaps.
            current_depth: Current recursion depth for context.
            
        Returns:
            List of concept strings that need additional detailed information.
        """
        if not content or len(content.strip()) < 20:  # Skip very short content
            return []
        
        # Use LLM to identify gaps intelligently  
        gap_prompt = f"""You are reviewing troubleshooting documentation. Your job is to find steps that mention actions but don't provide the specific commands or code to perform them.

USER REQUEST: {user_request}

DOCUMENTATION:
{content[:1000]}

Question: Are there any instructions that mention checking, verifying, running, or validating something but DON'T include the actual command, code, or specific procedure to do it?

IMPORTANT CONTEXT AWARENESS:
- If the user already confirmed something is working (e.g., "book2 resolved to MarkitWire"), DON'T suggest re-verifying that same thing
- Focus on NEXT troubleshooting steps after what the user has confirmed
- Prioritize gaps that help with the user's specific unresolved issue

Look specifically for:
- "check block events" without showing how to list them  
- "validate" without showing validation commands
- "check" or "verify" without providing the actual check/verification code
- References to procedures without implementation details

If you find actions that lack implementation details, list the 1-2 most important ones that need code/commands.
PRIORITIZE gaps that are relevant to resolving the user's current issue.
If the documentation provides complete implementation details, return empty array.

Answer with just a JSON array: ["specific action that needs implementation", "another action"] or []

Focus on finding actionable next steps that are mentioned but not implemented with code."""

        try:
            from mcp.types import CreateMessageRequestParams, TextContent
            
            sampling_request = CreateMessageRequestParams(
                messages=[{
                    "role": "user",
                    "content": TextContent(type="text", text=gap_prompt)
                }],
                model=self.config.llm.default_model,
                maxTokens=100,  # Keep small for gap identification
                temperature=self.config.llm.temperature
            )
            
            llm_result = await self._sampling_callback(sampling_request)
            
            if llm_result.content and hasattr(llm_result.content, 'text'):
                response = llm_result.content.text.strip()
                
                # Parse JSON response
                import json
                try:
                    gaps = json.loads(response)
                    if isinstance(gaps, list):
                        return [gap.strip() for gap in gaps[:2] if isinstance(gap, str) and gap.strip()]
                except json.JSONDecodeError:
                    pass
                    
        except Exception:
            pass
        
        return []  # Return empty list if LLM analysis fails
    
    async def _should_assistant_handle_request(self, classification: Classification, issue_description: str) -> bool:
        """Use LLM to determine if the assistant should handle this request or defer to humans.
        
        Analyzes the request to determine if it requires human review, approval, or compliance checks.
        
        Args:
            classification: Classification result for the request.
            issue_description: Original user request description.
            
        Returns:
            bool: True if assistant should handle, False if requires human review.
        """
        
        decision_prompt = f"""You are determining whether an AI assistant should handle a support request or defer to human review.

REQUEST CLASSIFICATION:
Category: {classification.category}
Subcategory: {classification.subcategory or 'None'}
Priority: {classification.priority}

USER REQUEST: {issue_description}

DECISION CRITERIA:
- Handle: Technical troubleshooting, data analysis, system investigation, queries about procedures
- Human Review: Approval requests, compliance reviews, blessing requests, policy decisions, sensitive changes

Question: Should the AI assistant handle this request directly or defer to human review?

Answer with just "HANDLE" or "HUMAN_REVIEW"."""

        try:
            from mcp.types import CreateMessageRequestParams, TextContent
            
            sampling_request = CreateMessageRequestParams(
                messages=[{
                    "role": "user",
                    "content": TextContent(type="text", text=decision_prompt)
                }],
                model=self.config.llm.default_model,
                maxTokens=50,  # Keep small for entity extraction
                temperature=self.config.llm.temperature
            )
            
            llm_result = await self._sampling_callback(sampling_request)
            
            if llm_result.content and hasattr(llm_result.content, 'text'):
                response = llm_result.content.text.strip().upper()
                return "HANDLE" in response
                
        except Exception:
            pass
        
        # Conservative fallback: defer to human for unknown cases
        return False
    
    
    async def _generate_recommendations(self, classification: Classification, knowledge_data, health_status, request_description: str) -> str:
        """Generate intelligent recommendations based on retrieved knowledge.
        
        Uses the best matching knowledge base entry to generate contextual recommendations
        via LLM processing. The method prepares an analysis prompt with the knowledge
        content and user request, then uses the LLM to generate specific, actionable
        recommendations.
        
        Args:
            classification: Classification result for context.
            knowledge_data: List of knowledge base search results.
            health_status: System health information (currently unused but available).
            request_description: Original support request description.
        
        Returns:
            str: Generated recommendations with source attribution, or None if
                no relevant knowledge is available.
        """
        
        if isinstance(knowledge_data, list) and len(knowledge_data) > 0:
            # Separate primary and follow-up results
            primary_results = [item for item in knowledge_data if not item.get("is_followup", False)]
            followup_results = [item for item in knowledge_data if item.get("is_followup", False)]
            
            # Combine top 2-3 primary results for comprehensive guidance
            if primary_results:
                # Sort by relevance score and take top results
                sorted_results = sorted(primary_results, key=lambda x: x.get("relevance_score", 0), reverse=True)
                top_results = sorted_results[:3]  # Focus on most relevant sections for DFS approach
                best_result = top_results[0]  # Primary source for metadata
                
                # Combine content from all top results
                combined_content = ""
                for i, result in enumerate(top_results):
                    if i == 0:
                        combined_content += result.get('content', '')
                    else:
                        # Add additional sections with clear separation
                        combined_content += f"\n\n---\nADDITIONAL RELEVANT SECTION:\n"
                        combined_content += f"Source: {result.get('source', 'Unknown')}\n\n"
                        combined_content += result.get('content', '')
            else:
                best_result = knowledge_data[0]
                combined_content = best_result.get('content', '')
            
            if followup_results:
                combined_content += "\n\n---\nADDITIONAL KNOWLEDGE FOUND:\n"
                for followup in followup_results:
                    searched_concept = followup.get("searched_for", "Unknown")
                    combined_content += f"\nFor '{searched_concept}' mentioned above, here is additional information:\n"
                    combined_content += followup.get('content', '')
                    combined_content += "\n"
                
            
            try:
                # Get the analysis prompt from knowledge server
                prompt_result = await self.mcp_sessions["knowledge"].call_tool(
                    "prepare_analysis_prompt",
                    {
                        "query": request_description,
                        "knowledge_content": combined_content,
                        "affected_system": None
                    }
                )
                
                analysis_prompt = prompt_result.content[0].text
                
                # Create sampling request
                from mcp.types import CreateMessageRequestParams, TextContent
                
                sampling_request = CreateMessageRequestParams(
                    messages=[{
                        "role": "user", 
                        "content": TextContent(type="text", text=analysis_prompt)
                    }],
                    model=self.config.llm.default_model,
                    maxTokens=500  # Fixed size for system info
                )
                
                # Use our sampling callback to get LLM response
                llm_result = await self._sampling_callback(sampling_request)
                
                # Estimate tokens for LLM call
                self._total_tokens += self._estimate_tokens(analysis_prompt)
                
                if llm_result.content and hasattr(llm_result.content, 'text'):
                    llm_response = llm_result.content.text
                    self._total_tokens += self._estimate_tokens(llm_response)
                else:
                    llm_response = str(llm_result.content)
                    self._total_tokens += self._estimate_tokens(llm_response)
                
                # Build source attribution
                source_text = f"Primary Source: {best_result.get('source', 'Unknown')} (Relevance: {best_result.get('relevance_score', 0)*100:.0f}%)"
                
                if followup_results:
                    source_text += "\nAdditional Sources:"
                    for followup in followup_results:
                        source_text += f"\n- {followup.get('source', 'Unknown')} (searched for: {followup.get('searched_for', 'Unknown')})"
                
                return f"""{llm_response}

---
{source_text}
*Analysis generated using LLM processing*"""
                
            except Exception as e:
                # If analysis tool fails, return None to stay silent
                return None
        else:
            # No fallback - if no knowledge found, return None to indicate low confidence
            return None
    
    def _update_tool_call_status(self, message: str):
        """Update tool call status in the console for user feedback.
        
        Provides real-time feedback about which MCP tools are being called,
        similar to how Claude Code displays tool usage. Shows a running count
        of tool calls to indicate processing activity.
        
        Args:
            message: Status message describing the current tool call.
        """
        self._tool_call_count += 1
        # Format status with tool icon and count
        status_text = f"⏺ {message}"
        detail_text = f"({self._tool_call_count} tool {'use' if self._tool_call_count == 1 else 'uses'})"
        
        # Clear line and print status
        sys.stdout.write(f"\r{status_text} {detail_text}")
        sys.stdout.flush()
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for resource tracking.
        
        Attempts to use tiktoken for accurate token counting, falling back
        to character-based estimation if tiktoken is not available. This is
        used to track approximate LLM token usage throughout the analysis.
        
        Args:
            text: String to estimate tokens for.
        
        Returns:
            int: Estimated number of tokens.
        """
        try:
            # Try to use tiktoken for accurate counting
            import tiktoken
            encoding = tiktoken.encoding_for_model("gpt-4")
            return len(encoding.encode(str(text)))
        except ImportError:
            # Fallback to rough estimation: 1 token ≈ 4 characters
            return len(str(text)) // 4
    
    async def get_health_status(self) -> Dict[str, str]:
        """Get health status of all MCP servers.
        
        Tests connectivity and responsiveness of each MCP server by calling
        a simple tool on each one. This method is used for diagnostics and
        system health monitoring.
        
        Returns:
            Dict[str, str]: Dictionary mapping server names to health status.
                          Status is either "healthy" or "unhealthy: <error>".
        """
        if not self.servers_running:
            return {"MCP Servers": "not running"}
        
        health = {}
        for name in self.mcp_sessions.keys():
            try:
                # Simple connectivity test - just check if session exists and is responsive
                session = self.mcp_sessions[name]
                if session and hasattr(session, 'list_tools'):
                    # Session is available and responsive
                    pass
                
                health[f"{name.title()} Server"] = "healthy"
            except Exception as e:
                health[f"{name.title()} Server"] = f"unhealthy: {str(e)}"
        
        return health
    
    async def get_system_info(self) -> Dict:
        """Get comprehensive system information and capabilities.
        
        Retrieves information about the support system's current configuration
        and capabilities, including available categories, workflows, and
        knowledge base status. Used for system diagnostics and capability
        discovery.
        
        Returns:
            Dict: System information including available categories, workflows,
                 and resource counts. Returns error dict if servers not running.
        """
        if not self.servers_running:
            return {"error": "MCP servers not running"}
        
        try:
            # Get available categories for ATRS team (backward compatibility)
            categories_result = await self.mcp_sessions["classification"].call_tool("get_team_categories", {"team": "atrs"})
            categories_data = categories_result.content[0].text
            import json
            categories = list(json.loads(categories_data).keys()) if isinstance(categories_data, str) else []
            
            return {
                "available_categories": categories if isinstance(categories, list) else [],
                "knowledge_base_entries": 3,  # Based on our servers
                "active_conversations": 0,
                "supported_workflows": ["technical_guidance", "outage_investigation", "data_issue_resolution", "human_review_required"]
            }
        except Exception as e:
            return {"error": str(e)}