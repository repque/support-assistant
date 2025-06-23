"""Health Monitoring MCP Server for service status and log analysis.

This module implements an MCP server that monitors the health and performance
of various services in a financial services infrastructure. It provides tools
for checking service status, querying logs, and analyzing system behavior to
support troubleshooting and incident response.

The server simulates a realistic monitoring environment with multiple services,
health metrics, and log entries that represent common operational scenarios
in production systems.
"""

import random
from datetime import datetime, timedelta
from typing import Dict, List

from mcp.server.fastmcp import FastMCP

# Create FastMCP server instance
mcp = FastMCP("HealthMonitoringServer")

# Mock service status data for financial services
SERVICE_STATUS = {}
LOG_ENTRIES = []


def _initialize_mock_data():
    """Initialize mock service status and log data for demonstration.

    Creates realistic mock data representing:
    - Service health status (healthy, degraded, unhealthy)
    - Performance metrics (response time, resource usage)
    - Log entries with various severity levels
    - Common error patterns and operational messages

    The mock data simulates a typical financial services environment with
    services like trade booking, risk engine, and reconciliation systems.
    """
    global SERVICE_STATUS, LOG_ENTRIES

    services = [
        "trade-booking-service",
        "risk-engine",
        "markitwire-gateway",
        "data-pipeline",
        "reconciliation-service",
        "trade-database",
    ]

    # Create mock service status
    for service in services:
        # Randomly assign health status for demo
        health_roll = random.random()
        if health_roll > 0.8:
            status = "unhealthy"
            response_time = random.uniform(5000, 15000)
        elif health_roll > 0.6:
            status = "degraded"
            response_time = random.uniform(1000, 3000)
        else:
            status = "healthy"
            response_time = random.uniform(50, 500)

        SERVICE_STATUS[service] = {
            "status": status,
            "response_time_ms": response_time,
            "last_check": datetime.now() - timedelta(minutes=random.randint(1, 30)),
            "details": {
                "cpu_usage": random.uniform(10, 90),
                "memory_usage": random.uniform(20, 80),
                "disk_usage": random.uniform(15, 70),
                "active_connections": random.randint(5, 100),
                "error_count": random.randint(0, 10),
            },
        }

    # Create mock log entries
    log_levels = ["INFO", "WARN", "ERROR", "DEBUG"]
    # Demo error patterns for mock data generation only (not used for analysis)
    demo_error_patterns = [
        "Connection timeout",
        "Database lock timeout",
        "Invalid request format",
        "Authentication failed",
        "Service unavailable",
        "Memory allocation error",
    ]

    for i in range(200):
        # Generate realistic timestamps
        timestamp = datetime.now() - timedelta(
            minutes=random.randint(1, 120), seconds=random.randint(0, 59)
        )

        level = random.choice(log_levels)
        service = random.choice(services)

        # Generate realistic log messages
        if level == "ERROR":
            message = f"[{random.choice(demo_error_patterns)}] {random.choice(['Request ID: REQ-' + str(random.randint(1000, 9999)), 'User ID: USR-' + str(random.randint(100, 999))])}"
        elif level == "WARN":
            message = f"High {random.choice(['CPU', 'memory', 'disk'])} usage detected: {random.randint(70, 95)}%"
        else:
            message = f"Processing {random.choice(['trade', 'reconciliation', 'feed'])} request successfully"

        LOG_ENTRIES.append(
            {
                "timestamp": timestamp,
                "level": level,
                "message": message,
                "service_name": service,
            }
        )


# Initialize mock data
_initialize_mock_data()


@mcp.tool()
def check_service_health(service_name: str) -> Dict:
    """Check health status of a specific service.

    Retrieves current health metrics for a service including status,
    response time, and resource utilization. This tool is essential
    for diagnosing service-related issues and understanding system state.

    Args:
        service_name: Name of the service to check (e.g., 'trade-booking-service').

    Returns:
        Dict: Health status containing:
             - service_name: Service identifier
             - status: Current status (healthy/degraded/unhealthy/unknown)
             - response_time_ms: Average response time in milliseconds
             - last_check: Timestamp of last health check
             - details: Resource usage metrics (CPU, memory, disk, connections)
    """
    if service_name not in SERVICE_STATUS:
        return {
            "service_name": service_name,
            "status": "unknown",
            "response_time_ms": 0,
            "last_check": None,
            "details": {"error": "Service not found"},
        }

    health_status = SERVICE_STATUS[service_name]
    return {
        "service_name": service_name,
        "status": health_status["status"],
        "response_time_ms": health_status["response_time_ms"],
        "last_check": (
            health_status["last_check"].isoformat()
            if health_status["last_check"]
            else None
        ),
        "details": health_status["details"],
    }


@mcp.tool()
def query_service_logs(
    service_name: str, hours_back: int = 1, max_lines: int = 100
) -> List[Dict]:
    """Query recent logs for a specific service.

    Retrieves log entries for troubleshooting and analysis. Logs are returned
    in reverse chronological order (newest first) to quickly identify recent
    issues or patterns.

    Args:
        service_name: Name of the service to query logs for.
        hours_back: Number of hours to look back (default: 1).
        max_lines: Maximum number of log entries to return (default: 100).

    Returns:
        List[Dict]: Log entries containing:
                   - timestamp: ISO format timestamp
                   - level: Log severity (INFO/WARN/ERROR/DEBUG)
                   - message: Log message content
                   - service: Service name
    """
    end_time = datetime.now()
    start_time = end_time - timedelta(hours=hours_back)

    # Filter logs by service and time range
    filtered_logs = []
    for entry in LOG_ENTRIES:
        if (
            entry["service_name"] == service_name
            and start_time <= entry["timestamp"] <= end_time
        ):
            filtered_logs.append(
                {
                    "timestamp": entry["timestamp"].isoformat(),
                    "level": entry["level"],
                    "message": entry["message"],
                    "service": entry["service_name"],
                }
            )

    # Sort by timestamp (newest first) and limit results
    filtered_logs.sort(key=lambda x: x["timestamp"], reverse=True)
    return filtered_logs[:max_lines]


@mcp.tool()
def analyze_service_logs(service_name: str, hours_back: int = 1) -> Dict:
    """Analyze service logs to identify patterns and issues.

    Performs automated log analysis to summarize service behavior, identify
    error patterns, and calculate error rates. This tool helps quickly
    understand service health trends without manually reviewing individual
    log entries.

    Args:
        service_name: Name of the service to analyze.
        hours_back: Time period to analyze (default: 1 hour).

    Returns:
        Dict: Analysis summary containing:
             - service_name: Service identifier
             - time_range: Analysis period
             - total_entries: Number of logs analyzed
             - level_breakdown: Count of logs by severity level
             - error_patterns: Common error types and frequencies
             - summary: Human-readable analysis summary
    """
    end_time = datetime.now()
    start_time = end_time - timedelta(hours=hours_back)

    # Filter logs by service and time range
    service_logs = []
    for entry in LOG_ENTRIES:
        if (
            entry["service_name"] == service_name
            and start_time <= entry["timestamp"] <= end_time
        ):
            service_logs.append(entry)

    if not service_logs:
        return {
            "service_name": service_name,
            "time_range": f"{start_time.isoformat()} to {end_time.isoformat()}",
            "total_entries": 0,
            "level_breakdown": {},
            "error_messages": [],
            "summary": f"No log entries found for {service_name} in the last {hours_back} hour(s)",
        }

    # Analyze log levels
    level_counts = {}
    error_messages = []

    for entry in service_logs:
        level = entry["level"]
        level_counts[level] = level_counts.get(level, 0) + 1

        if level == "ERROR":
            # Collect error messages for LLM analysis instead of pattern matching
            error_messages.append(entry["message"])

    # Generate summary
    total_logs = len(service_logs)
    error_count = level_counts.get("ERROR", 0)
    warn_count = level_counts.get("WARN", 0)

    summary = f"Analyzed {total_logs} log entries for {service_name}. "

    if error_count > 0:
        error_rate = (error_count / total_logs) * 100
        summary += f"Found {error_count} errors ({error_rate:.1f}% error rate). "

        if error_messages:
            # For demo purposes, show sample error messages instead of pattern analysis
            sample_errors = error_messages[:3]  # Show first 3 error messages
            summary += f"Sample error messages: {'; '.join(sample_errors)}. "

    if warn_count > 0:
        warn_rate = (warn_count / total_logs) * 100
        summary += f"Found {warn_count} warnings ({warn_rate:.1f}% warning rate). "

    if error_count == 0 and warn_count == 0:
        summary += "No errors or warnings detected. System appears healthy."

    return {
        "service_name": service_name,
        "time_range": f"{start_time.isoformat()} to {end_time.isoformat()}",
        "total_entries": total_logs,
        "level_breakdown": level_counts,
        "error_messages": (
            error_messages[:10] if error_messages else []
        ),  # Return up to 10 error messages for LLM analysis
        "summary": summary,
    }


@mcp.tool()
def get_all_services() -> List[str]:
    """Get list of all monitored services.

    Returns the complete list of services being monitored by the health
    system. Useful for discovery and understanding the system landscape.

    Returns:
        List[str]: Names of all monitored services.
    """
    return list(SERVICE_STATUS.keys())


@mcp.tool()
def get_system_overview() -> Dict:
    """Get comprehensive overview of system health.

    Provides a high-level summary of the entire system's health status,
    categorizing services by their current state. This tool is ideal for
    quick system assessment and identifying services needing attention.

    Returns:
        Dict: System overview containing:
             - healthy_services: List of services operating normally
             - degraded_services: List of services with performance issues
             - unhealthy_services: List of services experiencing problems
             - total_services: Total number of monitored services
             - overall_status: System-wide status (healthy/degraded/critical)
    """
    overview = {
        "healthy_services": [],
        "degraded_services": [],
        "unhealthy_services": [],
        "total_services": len(SERVICE_STATUS),
    }

    for service_name, status_info in SERVICE_STATUS.items():
        status = status_info["status"]
        if status == "healthy":
            overview["healthy_services"].append(service_name)
        elif status == "degraded":
            overview["degraded_services"].append(service_name)
        else:
            overview["unhealthy_services"].append(service_name)

    overview["overall_status"] = "healthy"
    if overview["unhealthy_services"]:
        overview["overall_status"] = "critical"
    elif overview["degraded_services"]:
        overview["overall_status"] = "degraded"

    return overview


# Main execution
async def main():
    """Main entry point for the health monitoring server.

    Initializes the MCP server with mock monitoring data and handles
    connection setup for both STDIO and SSE protocols. The server
    provides real-time health monitoring capabilities for support
    and operations teams.
    """
    import argparse
    import logging
    import os

    # Set logging level based on environment variable
    log_level = os.getenv("MCP_LOG_LEVEL", "INFO")
    logging.getLogger().setLevel(getattr(logging, log_level))

    parser = argparse.ArgumentParser(description="Health Monitoring MCP Server")
    parser.add_argument(
        "--connection",
        choices=["stdio", "sse"],
        default="stdio",
        help="Connection method (default: stdio)",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host for SSE connections")
    parser.add_argument(
        "--port", type=int, default=8003, help="Port for SSE connections"
    )

    args = parser.parse_args()

    if args.connection == "stdio":
        await mcp.run_stdio_async()
    elif args.connection == "sse":
        await mcp.run_sse_async(host=args.host, port=args.port)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
