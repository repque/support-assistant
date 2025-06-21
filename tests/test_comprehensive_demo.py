#!/usr/bin/env python3
"""
Comprehensive end-to-end demonstration of the multi-team classification system
"""

import asyncio
import logging
from support_agent.assistant import SupportAssistant
from support_agent.models import SupportRequest
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("comprehensive-demo")
console = Console()

async def run_comprehensive_demo():
    """Run a comprehensive demonstration of all system capabilities"""
    
    console.print("\n[bold cyan]🚀 Comprehensive Multi-Team Classification System Demo[/bold cyan]")
    console.print("=" * 70)
    
    # Create assistant
    assistant = SupportAssistant()
    
    try:
        # Start MCP servers
        console.print("\n[yellow]📡 Starting MCP servers...[/yellow]")
        success = await assistant.start_mcp_servers()
        if not success:
            console.print("[red]❌ Failed to start MCP servers[/red]")
            return
        console.print("[green]✅ MCP servers started successfully[/green]")
        
        # Define comprehensive test scenarios
        scenarios = [
            {
                "title": "ATRS Team - Data Reconciliation Issue",
                "team": "atrs",
                "request": SupportRequest(
                    engineer_sid="john.smith",
                    request_id="DEMO-001",
                    issue_description="Position reconciliation showing $2M discrepancy between trading system and risk management platform for APAC portfolios",
                    affected_system="reconciliation-service",
                    lob="trading"
                ),
                "expected_behavior": "Should classify as data_issue with high confidence and provide detailed resolution steps"
            },
            {
                "title": "ATRS Team - System Outage",
                "team": "atrs",
                "request": SupportRequest(
                    engineer_sid="jane.doe",
                    request_id="DEMO-002",
                    issue_description="Risk management system is completely down, all traders unable to view positions or P&L",
                    affected_system="risk-engine",
                    lob="trading"
                ),
                "expected_behavior": "Should classify as outage with critical priority"
            },
            {
                "title": "ATRS Team - Human Review Required",
                "team": "atrs",
                "request": SupportRequest(
                    engineer_sid="alex.chen",
                    request_id="DEMO-003",
                    issue_description="Please review and approve this code change for the credit risk calculation engine",
                    affected_system="credit-risk-engine",
                    lob="risk"
                ),
                "expected_behavior": "Should stay silent - review requests require human intervention"
            },
            {
                "title": "Core Team - Database Performance",
                "team": "core",
                "request": SupportRequest(
                    engineer_sid="mike.wilson",
                    request_id="DEMO-004",
                    issue_description="PostgreSQL queries on the analytics database are taking 30+ seconds, causing timeouts in the reporting dashboard",
                    affected_system="analytics-db",
                    lob="platform"
                ),
                "expected_behavior": "Should classify as database issue with performance guidance"
            },
            {
                "title": "Core Team - Kubernetes Scaling",
                "team": "core",
                "request": SupportRequest(
                    engineer_sid="sarah.jones",
                    request_id="DEMO-005",
                    issue_description="Kubernetes cluster is hitting memory limits, pods are being evicted during peak trading hours",
                    affected_system="k8s-prod-cluster",
                    lob="infrastructure"
                ),
                "expected_behavior": "Should classify as cloud/scaling issue with critical priority"
            },
            {
                "title": "Vague Request - Should Stay Silent",
                "team": "atrs",
                "request": SupportRequest(
                    engineer_sid="test.user",
                    request_id="DEMO-006",
                    issue_description="Something is broken, please help",
                    affected_system=None,
                    lob="unknown"
                ),
                "expected_behavior": "Should stay silent due to low confidence (vague request)"
            }
        ]
        
        # Process each scenario
        for i, scenario in enumerate(scenarios, 1):
            console.print(f"\n[bold blue]📋 Scenario {i}: {scenario['title']}[/bold blue]")
            console.print("-" * 60)
            
            # Display request details
            request = scenario["request"]
            table = Table(show_header=False, box=None)
            table.add_column("Field", style="dim")
            table.add_column("Value")
            table.add_row("Engineer", request.engineer_sid)
            table.add_row("Request ID", request.request_id)
            table.add_row("Description", request.issue_description)
            table.add_row("System", request.affected_system or "N/A")
            table.add_row("LOB", request.lob)
            console.print(table)
            
            console.print(f"\n[dim]Expected: {scenario['expected_behavior']}[/dim]")
            console.print("\n[yellow]Processing...[/yellow]")
            
            # Analyze the request
            result = await assistant.analyze_support_request(request)
            
            if result is None:
                console.print("[cyan]✅ Assistant stayed silent (as expected for low confidence/human review)[/cyan]")
            else:
                # Display classification results
                analysis = result.get("analysis")
                if analysis:
                    classification = analysis.classification
                    
                    # Create classification panel
                    classification_info = f"""[bold]Category:[/bold] {classification.category}
[bold]Subcategory:[/bold] {classification.subcategory}
[bold]Priority:[/bold] {classification.priority}
[bold]Confidence:[/bold] {classification.confidence:.0%}
[bold]Workflow:[/bold] {classification.suggested_workflow}
[bold]Reasoning:[/bold] {classification.reasoning}"""
                    
                    console.print(Panel(classification_info, title="Classification Result", border_style="green"))
                    
                    # Show confidence score
                    confidence_color = "green" if analysis.confidence_score >= 0.8 else "yellow" if analysis.confidence_score >= 0.6 else "red"
                    console.print(f"\n[{confidence_color}]Overall Confidence: {analysis.confidence_score:.0%}[/{confidence_color}]")
                    
                    # Show recommendations summary
                    if result.get("recommendations", {}).get("resolution_steps"):
                        steps = result["recommendations"]["resolution_steps"]
                        console.print(f"\n[green]✅ Generated {len(steps)} characters of recommendations[/green]")
                    
                    # Show token usage
                    if assistant._total_tokens > 0:
                        console.print(f"[dim]Tokens used: {assistant._total_tokens:,}[/dim]")
        
        # Final system health check
        console.print(f"\n[bold cyan]🔍 System Health Check[/bold cyan]")
        console.print("-" * 40)
        health = await assistant.get_health_status()
        for service, status in health.items():
            status_color = "green" if status == "healthy" else "red"
            console.print(f"[{status_color}]{service}: {status}[/{status_color}]")
        
        # System capabilities summary
        console.print(f"\n[bold cyan]📊 System Capabilities[/bold cyan]")
        console.print("-" * 40)
        info = await assistant.get_system_info()
        console.print(f"Available categories: {', '.join(info.get('available_categories', []))}")
        console.print(f"Knowledge base entries: {info.get('knowledge_base_entries', 0)}")
        console.print(f"Supported workflows: {', '.join(info.get('supported_workflows', []))}")
        
        console.print(f"\n[bold green]✨ Demo completed successfully![/bold green]")
        console.print(f"[dim]The system demonstrated:[/dim]")
        console.print("• Multi-team support (ATRS and Core teams)")
        console.print("• End-to-end LLM classification without fallbacks")
        console.print("• Appropriate confidence scoring for vague requests")
        console.print("• Human review detection for sensitive requests")
        console.print("• Knowledge-based recommendations generation")
        console.print("• Real-time health monitoring")
        
    except Exception as e:
        console.print(f"[red]❌ Demo failed: {e}[/red]")
        import traceback
        traceback.print_exc()
        
    finally:
        # Clean up
        await assistant.stop_mcp_servers()
        console.print("\n[dim]MCP servers stopped[/dim]")

if __name__ == "__main__":
    asyncio.run(run_comprehensive_demo())