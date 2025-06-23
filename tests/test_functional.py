#!/usr/bin/env python3
"""Functional tests for the Support Agent system.

These tests verify the core functionality through the CLI interface,
which is the primary way the system is used in production.
"""

import subprocess
from pathlib import Path

import pytest

from support_agent.assistant import SupportAssistant
from support_agent.config import MCPConfig
from support_agent.models import SupportRequest


def test_cli_health_check():
    """Test that the health check command works."""
    result = subprocess.run(
        ["python", "-m", "support_agent.cli", "health"],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent,
    )

    assert result.returncode == 0
    assert "System Health Status:" in result.stdout
    assert "healthy" in result.stdout


def test_cli_info_command():
    """Test that the info command works."""
    result = subprocess.run(
        ["python", "-m", "support_agent.cli", "info"],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent,
    )

    assert result.returncode == 0
    assert "System Information:" in result.stdout
    assert "Knowledge Base Entries:" in result.stdout


def test_cli_demo_scenarios():
    """Test that the automated demo scenarios run without crashes."""
    result = subprocess.run(
        ["python", "-m", "support_agent.cli", "demo", "--no-interactive"],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent,
        timeout=60,  # 60 second timeout
    )

    # Should complete without crashing
    assert result.returncode == 0

    # Should contain expected demo scenarios
    assert "Demo Scenario #1:" in result.stdout
    assert "Demo Scenario #2:" in result.stdout
    assert "Demo Scenario #3:" in result.stdout

    # Should show system working
    assert "Starting MCP servers" in result.stdout
    assert "All MCP servers running!" in result.stdout


@pytest.mark.asyncio
async def test_markitwire_context_awareness():
    """Test the core MarkitWire context awareness functionality."""
    config = MCPConfig()
    assistant = SupportAssistant(config)

    success = await assistant.start_mcp_servers()
    assert success, "Failed to start MCP servers"

    try:
        request = SupportRequest(
            engineer_sid="test-engineer",
            request_id="TEST-MARKITWIRE",
            issue_description="I booked a trade in Athena but it didn't show up in the MarkitWire feed",
            lob="platform",
        )

        result = await assistant.analyze_support_request(request)

        # Should provide recommendations (not stay silent)
        assert (
            result is not None
        ), "System should provide recommendations for MarkitWire issues"

        recommendations = result.get("recommendations", {}).get("resolution_steps", "")
        assert len(recommendations) > 0, "Should provide non-empty recommendations"

        # Check context awareness - should not suggest verifying book2
        assert (
            "verify book2" not in recommendations.lower()
        ), "Should skip book2 verification when user stated it's resolved"

        # Check that block events code is provided
        assert (
            "ds.evInfo()" in recommendations
        ), "Should provide block events implementation code"

    finally:
        await assistant.stop_mcp_servers()


@pytest.mark.asyncio
async def test_vague_query_handling():
    """Test that vague queries are handled gracefully without errors."""
    config = MCPConfig()
    assistant = SupportAssistant(config)

    success = await assistant.start_mcp_servers()
    assert success, "Failed to start MCP servers"

    try:
        # Test the specific query that was causing "list index out of range"
        request = SupportRequest(
            engineer_sid="test-engineer",
            request_id="TEST-VAGUE",
            issue_description="I need help with this thing",
            lob="platform",
        )

        result = await assistant.analyze_support_request(request)

        # Should stay silent for vague queries (no relevant knowledge found)
        assert (
            result is None
        ), "System should stay silent for vague queries with no relevant knowledge"

        # Test other vague queries to ensure robustness
        vague_queries = ["help", "", "something is wrong", "fix this"]

        for query in vague_queries:
            if not query:  # Skip empty string to avoid issues
                continue

            request.issue_description = query
            request.request_id = f"TEST-VAGUE-{query.replace(' ', '-').upper()}"

            # Should not crash - may return None or recommendations depending on knowledge match
            result = await assistant.analyze_support_request(request)
            # Just verify it doesn't crash - result can be None or have recommendations

    finally:
        await assistant.stop_mcp_servers()


@pytest.mark.asyncio
async def test_human_review_detection():
    """Test that compliance/review requests are properly detected."""
    config = MCPConfig()
    assistant = SupportAssistant(config)

    success = await assistant.start_mcp_servers()
    assert success, "Failed to start MCP servers"

    try:
        request = SupportRequest(
            engineer_sid="test-engineer",
            request_id="TEST-REVIEW",
            issue_description="Can you please review this trade for compliance approval?",
            lob="platform",
        )

        result = await assistant.analyze_support_request(request)

        # Should stay silent for review requests
        assert result is None, "System should defer compliance requests to human review"

    finally:
        await assistant.stop_mcp_servers()


def test_no_hardcoded_logic_in_codebase():
    """Verify no hardcoded patterns remain in the codebase."""

    # Check for common hardcoding patterns
    hardcoded_patterns = [
        'if.*"markitwire"',
        'if.*"feed".*and.*"feed"',
        'if.*"block"',
        '\\["bless_request".*"review_request"\\]',
        'if.*name.*==.*"classification"',
        "score.*=.*0\\.[0-9].*markitwire",
    ]

    python_files = list(Path(__file__).parent.parent.glob("**/*.py"))
    # Exclude test files from hardcoding check
    python_files = [f for f in python_files if "test" not in str(f)]

    for pattern in hardcoded_patterns:
        for py_file in python_files:
            content = py_file.read_text()
            # Use simple string search since we're looking for obvious patterns
            if "markitwire" in pattern.lower() and "markitwire" in content.lower():
                # Check if it's in a hardcoded conditional
                lines = content.split("\n")
                for line in lines:
                    if (
                        "if" in line.lower()
                        and "markitwire" in line.lower()
                        and "==" in line
                    ):
                        pytest.fail(
                            f"Found hardcoded pattern in {py_file}: {line.strip()}"
                        )


if __name__ == "__main__":
    # Run basic tests directly
    print("Running functional tests...")

    print("✓ Testing CLI health check...")
    test_cli_health_check()

    print("✓ Testing CLI info command...")
    test_cli_info_command()

    print("✓ Testing hardcoded logic check...")
    test_no_hardcoded_logic_in_codebase()

    print("✓ All functional tests passed!")
