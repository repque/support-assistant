"""Command-line interface for the Support Agent demonstration.

This module provides the CLI interface for interacting with the support agent
system. It offers both interactive and automated demonstration modes, health
checking, and system information display. The CLI is designed to showcase
the capabilities of the AI-powered support assistant in various scenarios.

The interface uses Rich for enhanced terminal output with colors, panels,
and formatted text to provide a professional demonstration experience.
"""

import asyncio
from datetime import datetime

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from typing_extensions import Annotated

from .assistant import SupportAssistant
from .config import ConnectionMethod, MCPConfig
from .models import SupportRequest

app = typer.Typer(
    name="support-agent",
    help="AI Production Support Assistant - Phase 1 Demo",
    rich_markup_mode="rich",
)

console = Console()


@app.command()
def demo(
    no_interactive: Annotated[
        bool,
        typer.Option(
            "--no-interactive", help="Run sample scenarios instead of interactive mode"
        ),
    ] = False,
    engineer_id: Annotated[
        str, typer.Option("--engineer", "-e", help="Engineer ID")
    ] = "demo-engineer",
    lob: Annotated[
        str, typer.Option("--lob", "-l", help="Line of Business")
    ] = "platform",
    connection: Annotated[
        str, typer.Option("--connection", "-c", help="MCP connection method")
    ] = "stdio",
    search_depth: Annotated[
        int,
        typer.Option(
            "--search-depth",
            "-d",
            help="Knowledge search recursion depth (0=disabled, 1=single-level, 2+=recursive)",
        ),
    ] = 1,
):
    """Run the support agent demo in interactive or automated mode.

    Provides two demonstration modes:
    1. Interactive mode: User can input custom support requests
    2. Automated mode: Runs through predefined scenarios showcasing different capabilities

    The demo connects to MCP servers, analyzes support requests, and displays
    intelligent recommendations based on classification, knowledge base search,
    and system health checks.

    Args:
        no_interactive: If True, runs automated scenarios instead of interactive mode.
        engineer_id: Identifier for the engineer submitting requests.
        lob: Line of business context for request routing.
        connection: MCP connection method ('stdio' or 'sse').
        search_depth: Maximum levels of recursive knowledge enhancement (0=disabled, 1=single-level, 2+=recursive).
    """

    # Create MCP configuration
    config = MCPConfig()
    config.knowledge_search_depth = search_depth
    if connection == "sse":
        config.connection_method = ConnectionMethod.SSE
    elif connection == "stdio":
        config.connection_method = ConnectionMethod.STDIO
    else:
        console.print(f"[red]ERROR: Invalid connection method: {connection}[/red]")
        return

    console.print(
        Panel.fit(
            f"[bold yellow]AI Production Support Assistant[/bold yellow]\n"
            f"[dim]Phase 1 Demo - Analysis & Recommendations for Support Engineers[/dim]\n"
            f"[dim]Connection: {connection.upper()} | Knowledge Search Depth: {search_depth}[/dim]",
            style="yellow",
        )
    )

    if no_interactive:
        asyncio.run(run_sample_demo(config))
    else:
        asyncio.run(run_interactive_demo(engineer_id, lob, config))


@app.command()
def health():
    """Check system health status of all MCP servers.

    Performs a comprehensive health check by:
    1. Starting all configured MCP servers
    2. Testing connectivity to each server
    3. Displaying health status for each component

    This command is useful for troubleshooting connection issues and
    verifying that all components of the support system are operational.
    """
    console.print("Checking system health...")

    async def check_health():
        assistant = SupportAssistant()

        # Start servers for health check
        success = await assistant.start_mcp_servers()
        if not success:
            console.print("[red]❌ Failed to start MCP servers[/red]")
            return

        try:
            health_status = await assistant.get_health_status()

            console.print("\n[bold]System Health Status:[/bold]")
            for service, status in health_status.items():
                if "healthy" in status:
                    console.print(f"✓ {service}: [green]{status}[/green]")
                else:
                    console.print(f"✗ {service}: [red]{status}[/red]")
        finally:
            await assistant.stop_mcp_servers()

    asyncio.run(check_health())


@app.command()
def info():
    """Show system information and capabilities.

    Displays comprehensive information about the support system including:
    - Available support categories
    - Supported workflows
    - Knowledge base statistics
    - System configuration details

    This command helps users understand what types of issues the system
    can handle and its current configuration state.
    """

    async def show_info():
        assistant = SupportAssistant()

        # Start servers for info check
        success = await assistant.start_mcp_servers()
        if not success:
            console.print("[red]❌ Failed to start MCP servers[/red]")
            return

        try:
            system_info = await assistant.get_system_info()

            console.print("\n[bold]System Information:[/bold]")
            console.print(
                f"Knowledge Base Entries: {system_info.get('knowledge_base_entries', 0)}"
            )
            console.print(
                f"Active Conversations: {system_info.get('active_conversations', 0)}"
            )

            categories = system_info.get("available_categories", [])
            if categories:
                console.print("\n[bold]Supported Categories:[/bold]")
                for category in categories:
                    console.print(f"  - {category}")

            workflows = system_info.get("supported_workflows", [])
            if workflows:
                console.print("\n[bold]Available Workflows:[/bold]")
                for workflow in workflows:
                    console.print(f"  - {workflow}")
        finally:
            await assistant.stop_mcp_servers()

    asyncio.run(show_info())


async def run_sample_demo(config: MCPConfig):
    """Run automated demo with predefined sample scenarios.

    Demonstrates the support agent's capabilities through three key scenarios:
    1. High confidence analysis with clear technical issue
    2. Human review required for compliance/approval requests
    3. Low confidence leading to silent mode for vague requests

    Each scenario showcases different aspects of the intelligent decision-making
    process, including when the assistant chooses not to respond.

    Args:
        config: MCP configuration for server connections.
    """

    sample_issues = [
        {
            "description": "I booked a trade in Athena but it didn't show up in the MarkitWire feed",
            "demo_type": "High Confidence Analysis",
        },
        {
            "description": "Can you please check this code: https://github.com/sdlc/hydra/pull/1234?",
            "demo_type": "Human Review Required",
        },
        {
            "description": "I need help with this thing",
            "demo_type": "Low Confidence - Stays Silent",
        },
    ]

    assistant = SupportAssistant(config)

    # Start MCP servers
    success = await assistant.start_mcp_servers()
    if not success:
        console.print("[red]ERROR: Failed to start MCP servers for demo[/red]")
        return

    try:
        for i, issue in enumerate(sample_issues, 1):
            console.print(f"\n{'='*60}")
            console.print(
                f"[bold yellow]Demo Scenario #{i}: {issue['demo_type']}[/bold yellow]"
            )
            console.print(f"{'='*60}")

            request = SupportRequest(
                engineer_sid="demo-engineer",
                request_id=f"DEMO-{i:03d}",
                issue_description=issue["description"],
                lob="platform",
            )

            # Highlight the user query
            console.print(
                Panel(
                    f"[bold white]{issue['description']}[/bold white]",
                    title="[bold cyan]User Query[/bold cyan]",
                    style="cyan",
                )
            )

            # Wait before analyzing
            await asyncio.sleep(5)

            console.print("\n[bold yellow]⏺ Analyzing support request[/bold yellow]")

            # Track start time for demo
            import time

            start_time = time.time()

            result = await assistant.analyze_support_request(request)

            # Calculate elapsed time
            elapsed = time.time() - start_time

            # Show completion status if analysis was performed
            if result is not None:
                tool_count = getattr(assistant, "_tool_call_count", 0)
                token_count = getattr(assistant, "_total_tokens", 0)

                # Format token count with k suffix if large
                if token_count >= 1000:
                    token_str = f"{token_count/1000:.1f}k tokens"
                else:
                    token_str = f"{token_count} tokens"

                console.print(
                    f"  ⎓ Done ({tool_count} tool {'use' if tool_count == 1 else 'uses'} · {token_str} · {elapsed:.1f}s)"
                )
            if result is None:
                pass  # Assistant stayed silent - no additional commentary needed
            else:
                await display_analysis_results(result)

            # Wait before moving to next scenario
            if i < len(sample_issues):
                await asyncio.sleep(5)

    finally:
        await assistant.stop_mcp_servers()


async def run_interactive_demo(engineer_id: str, lob: str, config: MCPConfig):
    """Run interactive demo session with user input.

    Provides a conversational interface where users can:
    - Input custom support requests
    - Specify affected systems
    - Receive real-time analysis and recommendations
    - Continue with multiple queries in one session

    The interactive mode showcases the assistant's ability to handle
    diverse, real-world support scenarios with appropriate responses.

    Args:
        engineer_id: Identifier for the engineer using the system.
        lob: Line of business context.
        config: MCP configuration for server connections.
    """

    assistant = SupportAssistant(config)

    # Start MCP servers once
    success = await assistant.start_mcp_servers()
    if not success:
        console.print("[red]ERROR: Failed to start MCP servers for demo[/red]")
        return

    try:
        conversation_id = f"demo-{datetime.now().strftime('%Y%m%d-%H%M%S')}"


        while True:
            console.print("\n" + "=" * 60)

            # Get issue description from user
            issue_description = Prompt.ask(
                "\n[bold yellow]Describe the issue you're investigating[/bold yellow]\n"
                "[dim](or type 'quit' to exit)[/dim]"
            )

            if issue_description.lower() in ["quit", "exit", "q"]:
                console.print("\nThanks for using the Support Agent demo!")
                break

            # Create support request with unique ID for proper isolation
            request = SupportRequest(
                engineer_sid=engineer_id,
                request_id=f"REQ-{datetime.now().strftime('%Y%m%d%H%M%S')}",
                issue_description=issue_description,
                lob=lob,
            )

            # Analyze the request
            console.print("\n[bold yellow]⏺ Analyzing support request[/bold yellow]")

            # Track start time
            import time

            start_time = time.time()

            result = await assistant.analyze_support_request(request)

            # Calculate elapsed time
            elapsed = time.time() - start_time

            # Show completion status
            tool_count = getattr(assistant, "_tool_call_count", 0)
            token_count = getattr(assistant, "_total_tokens", 0)

            # Format token count with k suffix if large
            if token_count >= 1000:
                token_str = f"{token_count/1000:.1f}k tokens"
            else:
                token_str = f"{token_count} tokens"

            console.print(
                f"  ⎓ Done ({tool_count} tool {'use' if tool_count == 1 else 'uses'} · {token_str} · {elapsed:.1f}s)"
            )

            # Display results
            if result is None:
                console.print(
                    "[dim]This request would be handled by human support engineers.[/dim]"
                )
            else:
                await display_analysis_results(result)

    finally:
        await assistant.stop_mcp_servers()


async def display_analysis_results(result: dict):
    """Display analysis results in a formatted, user-friendly way.

    Presents the analysis results using Rich formatting with:
    - Color-coded panels for different information types
    - Markdown rendering for recommendations
    - Clear separation of classification, recommendations, and metadata

    Args:
        result: Dictionary containing analysis results from the support assistant.
    """

    if result.get("error"):
        console.print(
            Panel(
                f"✗ [red]Analysis Error:[/red]\n{result['error']}",
                title="Error",
                style="red",
            )
        )
        return

    analysis = result["analysis"]
    recommendations = result["recommendations"]

    # Classification Results
    if result["processing_metadata"]["classification"]:
        classification = result["processing_metadata"]["classification"]
        confidence_pct = classification["confidence"] * 100

        console.print(
            Panel(
                f"[bold]Category:[/bold] {classification['category']}\n"
                f"[bold]Subcategory:[/bold] {classification.get('subcategory', 'N/A')}\n"
                f"[bold]Priority:[/bold] {classification['priority']}\n"
                f"[bold]Confidence:[/bold] {confidence_pct:.1f}%\n"
                f"[bold]Reasoning:[/bold] {classification['reasoning']}",
                title="Issue Classification",
                style="dim",
            )
        )

    # Troubleshooting Steps
    resolution_content = recommendations["resolution_steps"]
    
    # Split content at the sources separator
    if "---" in resolution_content:
        main_content, sources_content = resolution_content.split("---", 1)
        main_content = main_content.strip()
        sources_content = sources_content.strip()
        
        # Display main troubleshooting content
        console.print(
            Panel(
                Markdown(main_content),
                title="Troubleshooting Steps",
                style="green",
            )
        )
        
        # Display sources content with dim styling
        console.print(
            Panel(
                sources_content,
                title="Sources",
                style="dim",
                border_style="dim",
            )
        )
    else:
        # Fallback if no sources section
        console.print(
            Panel(
                Markdown(resolution_content),
                title="Troubleshooting Steps",
                style="green",
            )
        )



def main():
    """Main entry point for the CLI application.

    Initializes and runs the Typer application, handling command-line
    argument parsing and routing to appropriate command handlers.
    """
    app()


if __name__ == "__main__":
    main()
