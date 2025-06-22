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
        CONFIDENCE_THRESHOLD: Minimum confidence score (0.60) required to provide recommendations.
        config: MCPConfig instance containing server configurations.
        mcp_sessions: Dictionary mapping server names to active MCP sessions.
        stdio_contexts: Dictionary storing STDIO context managers for cleanup.
        servers_running: Boolean indicating if MCP servers are active.
    """
    
    CONFIDENCE_THRESHOLD = 0.60  # 60% confidence required
    
    def __init__(self, config: Optional[MCPConfig] = None):
        self.config = config or MCPConfig()
        self.mcp_sessions = {}
        self.stdio_contexts = {}  # Store context managers for proper cleanup
        self.servers_running = False
        self._context_stack = None  # For managing multiple STDIO contexts
        self._tool_call_count = 0  # Track tool calls
        self._total_tokens = 0  # Track total tokens used
        
        # LLM configuration for MCP sampling
        self._llm_api_key = os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
        self._llm_api_base = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
        
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
                        "model": message.model or "gpt-4o-mini",
                        "messages": messages,
                        "max_tokens": message.maxTokens or 1000,
                        "temperature": 0.3
                    },
                    timeout=30.0
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
        
        self.mcp_sessions.clear()
        self.stdio_contexts.clear()
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
        
        # Start the server process
        process = subprocess.Popen([
            "python", str(server_script), 
            "--connection", "sse",
            "--host", server_config.host,
            "--port", str(server_config.port)
        ], stderr=subprocess.PIPE)
        
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
                    "team": "atrs"  # Default to ATRS team for backward compatibility
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
                maxTokens=1000,
                model="gpt-4o-mini",
                temperature=0.1
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
            
            # Step 2: Check if we should handle this request
            if classification.category in ["bless_request", "review_request"]:
                # Clear the status line
                sys.stdout.write("\r" + " " * 80 + "\r")
                sys.stdout.flush()
                _console.print(f"[dim]Assistant staying silent - {classification.category} requires human review[/dim]")
                return None
            
            # Step 3: Search knowledge base
            self._update_tool_call_status("Knowledge(search_knowledge)")
            knowledge_results = await self.mcp_sessions["knowledge"].call_tool(
                "search_knowledge",
                {
                    "query": request.issue_description,
                    "category": classification.category,
                    "max_results": 3
                }
            )
            # Estimate tokens
            self._total_tokens += self._estimate_tokens(request.issue_description)
            self._total_tokens += self._estimate_tokens(str(knowledge_results))
            
            knowledge_data = knowledge_results.content[0].text
            if isinstance(knowledge_data, str):
                import json
                try:
                    knowledge_data = json.loads(knowledge_data)
                except:
                    knowledge_data = []
            
            # Handle the case where knowledge_data is a single dict instead of a list
            if isinstance(knowledge_data, dict):
                knowledge_data = [knowledge_data]
            
            
            # Step 4: Skip system health checks - removed affected_system field
            # System should infer what to check from the issue description if needed
            health_status = {}
            
            # Step 4.5: Perform follow-up knowledge searches for gaps
            enhanced_knowledge_data = await self._enhance_knowledge_with_followup_searches(
                knowledge_data, request.issue_description
            )
            
            # Step 5: Calculate confidence score
            confidence_score = await self._calculate_confidence(
                classification, enhanced_knowledge_data, health_status
            )
            
            # Step 6: Check confidence threshold
            if confidence_score < self.CONFIDENCE_THRESHOLD:
                # Clear the status line
                sys.stdout.write("\r" + " " * 80 + "\r")
                sys.stdout.flush()
                _console.print(f"[yellow]Low confidence ({confidence_score:.1%}) - Assistant staying silent[/yellow]")
                return None
            
            # Step 7: Generate recommendations
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
                    confidence_score=confidence_score,
                    recommendations=recommendations,
                    sources_consulted=["Knowledge Base", "Health Monitor"]
                ),
                "recommendations": {"resolution_steps": recommendations},
                "processing_metadata": {
                    "classification": classification.model_dump(),
                    "confidence_score": confidence_score,
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
    
    async def _calculate_confidence(self, classification: Classification, knowledge_data, health_status) -> float:
        """Calculate confidence score based on available information quality.
        
        Implements a weighted scoring system to determine the assistant's confidence
        in its ability to provide helpful recommendations. The score considers:
        - Classification confidence (40% weight): How certain the classification is
        - Knowledge availability (40% weight): Quality of knowledge base matches
        - Health data availability (20% weight): Whether system status is known
        
        Args:
            classification: Classification result with confidence score.
            knowledge_data: List of knowledge base search results.
            health_status: Dictionary containing system health information.
        
        Returns:
            float: Confidence score between 0.0 and 1.0.
        """
        score = 0.0
        
        # Classification confidence (40% weight)
        if classification.confidence > 0.7:
            score += 0.4
        elif classification.confidence > 0.5:
            score += 0.2
        else:
            score += 0.1
        
        # Knowledge availability (40% weight) 
        if isinstance(knowledge_data, list) and len(knowledge_data) > 0:
            # Check if we have good knowledge matches
            best_score = max([item.get("relevance_score", 0) for item in knowledge_data], default=0)
            if best_score > 0.8:
                score += 0.4
            elif best_score > 0.5:
                score += 0.25
            else:
                score += 0.1
        else:
            score += 0.05
        
        # Health data availability (20% weight)
        if health_status and health_status.get("status") != "unknown":
            score += 0.2
        else:
            score += 0.05
        
        return min(score, 1.0)
    
    
    async def _enhance_knowledge_with_followup_searches(self, initial_knowledge_data, user_request: str):
        """Perform follow-up searches to fill knowledge gaps.
        
        Analyzes the initial knowledge results to identify concepts that are mentioned
        but lack implementation details (e.g., "check block events" without code).
        Then performs targeted follow-up searches to find that missing information.
        
        Args:
            initial_knowledge_data: Initial knowledge search results.
            user_request: Original user request for context.
            
        Returns:
            Enhanced knowledge data combining initial and follow-up search results.
        """
        if not initial_knowledge_data or not isinstance(initial_knowledge_data, list):
            return initial_knowledge_data
            
        # Identify gaps in the knowledge
        gaps_to_search = []
        
        for item in initial_knowledge_data:
            if not isinstance(item, dict):
                continue
                
            content = item.get("content", "").lower()
            
            # Look for specific mentions that need more detail
            # Specifically looking for "block events" which we know has a detailed guide
            if "check block events" in content or "check if there are any block events" in content:
                # Check if there's already code for block events in this content
                if "_blockevents" not in content.lower() and "block_events" not in content.lower():
                    gaps_to_search.append("block events")
            
            # Look for other common gaps
            if "check eligibility" in content and "ineligibilityreasons" not in content:
                gaps_to_search.append("eligibility IneligibilityReasons")
                
            if "compare values in athena" in content and not any(code in content for code in ["select", "query", "sql"]):
                gaps_to_search.append("Athena query examples")
        
        # Debug: print what gaps we found (commented out for production)
        # if gaps_to_search:
        #     print(f"\nIdentified knowledge gaps to search: {gaps_to_search}")
        
        # Perform follow-up searches for identified gaps
        enhanced_data = list(initial_knowledge_data)  # Start with initial data
        
        for gap_concept in gaps_to_search[:2]:  # Limit to 2 follow-up searches to avoid token explosion
            try:
                self._update_tool_call_status(f"Knowledge(follow-up: {gap_concept[:30]}...)")
                
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
                
                follow_up_data = follow_up_results.content[0].text
                if isinstance(follow_up_data, str):
                    import json
                    try:
                        follow_up_data = json.loads(follow_up_data)
                    except:
                        continue
                
                # Add follow-up results if they're relevant and not duplicates
                if isinstance(follow_up_data, list):
                    for item in follow_up_data:
                        if isinstance(item, dict) and item.get("relevance_score", 0) > 0.5:
                            # Check if this isn't already in our results
                            is_duplicate = any(
                                existing.get("source") == item.get("source") 
                                for existing in enhanced_data 
                                if isinstance(existing, dict)
                            )
                            if not is_duplicate:
                                # Mark this as a follow-up result
                                item["is_followup"] = True
                                item["searched_for"] = gap_concept
                                enhanced_data.append(item)
                                
            except Exception as e:
                # If follow-up search fails, continue with what we have
                print(f"Follow-up search failed for '{gap_concept}': {e}")
                continue
        
        return enhanced_data
    
    
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
            
            # Use the best primary result as the main content
            best_result = max(primary_results, key=lambda x: x.get("relevance_score", 0)) if primary_results else knowledge_data[0]
            
            # Combine knowledge content if we have follow-up results
            combined_content = best_result.get('content', '')
            
            if followup_results:
                combined_content += "\n\n## Additional Information from Follow-up Searches:\n"
                combined_content += "(The following information was found by searching for concepts mentioned above that lacked implementation details)\n"
                for followup in followup_results:
                    searched_concept = followup.get("searched_for", "Unknown")
                    combined_content += f"\n### Additional Details for: {searched_concept}\n"
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
                    model="gpt-4o-mini",
                    maxTokens=500
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
                # Try to call a simple tool to test connectivity
                if name == "classification":
                    await self.mcp_sessions[name].call_tool("list_teams", {})
                elif name == "knowledge":
                    await self.mcp_sessions[name].call_tool("search_knowledge", {"query": "test", "max_results": 1})
                elif name == "health":
                    await self.mcp_sessions[name].call_tool("get_all_services", {})
                
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