#!/usr/bin/env python3
"""Test multi-team classification functionality"""

import asyncio
import json
import logging
from typing import Any, Dict

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("multi-team-test")


async def mock_llm_classify_by_team(prompt: str, request: str, team: str) -> str:
    """Mock LLM that provides team-specific classifications"""
    
    request_lower = request.lower()
    
    if team == "atrs":
        # ATRS team classifications
        if "bless" in request_lower or "push" in request_lower:
            return json.dumps({
                "category": "bless_request",
                "subcategory": "bless",
                "priority": "medium",
                "confidence": 0.95,
                "reasoning": "ATRS blessing request for production deployment"
            })
        elif "review" in request_lower:
            return json.dumps({
                "category": "review_request",
                "subcategory": "code_review",
                "priority": "medium", 
                "confidence": 0.9,
                "reasoning": "ATRS code review request"
            })
        elif any(word in request_lower for word in ["down", "outage", "not responding"]):
            return json.dumps({
                "category": "outage",
                "subcategory": "system_down",
                "priority": "critical",
                "confidence": 0.9,
                "reasoning": "ATRS system outage - critical service disruption"
            })
        elif any(word in request_lower for word in ["reconciliation", "discrepancy"]):
            return json.dumps({
                "category": "data_issue",
                "subcategory": "reconciliation",
                "priority": "high",
                "confidence": 0.85,
                "reasoning": "ATRS data reconciliation issue"
            })
        else:
            return json.dumps({
                "category": "query",
                "subcategory": "feed_issue",
                "priority": "high",
                "confidence": 0.8,
                "reasoning": "ATRS technical feed/trading issue"
            })
    
    elif team == "core":
        # Core team classifications
        if "etl" in request_lower or "batch" in request_lower:
            return json.dumps({
                "category": "jobs",
                "subcategory": "etl_issue",
                "priority": "high",
                "confidence": 0.9,
                "reasoning": "Core team ETL/batch job issue"
            })
        elif any(word in request_lower for word in ["build", "pipeline", "ci/cd", "deployment"]):
            return json.dumps({
                "category": "sdlc",
                "subcategory": "build_failure",
                "priority": "medium",
                "confidence": 0.85,
                "reasoning": "Core team SDLC/build system issue"
            })
        elif any(word in request_lower for word in ["database", "sql", "postgres", "db"]):
            return json.dumps({
                "category": "database",
                "subcategory": "db_performance",
                "priority": "high",
                "confidence": 0.9,
                "reasoning": "Core team database performance issue"
            })
        elif any(word in request_lower for word in ["kubernetes", "cloud", "scaling"]):
            return json.dumps({
                "category": "cloud",
                "subcategory": "scaling_issue",
                "priority": "high",
                "confidence": 0.8,
                "reasoning": "Core team cloud infrastructure issue"
            })
        elif any(word in request_lower for word in ["ml", "model", "ai", "inference"]):
            return json.dumps({
                "category": "ai",
                "subcategory": "model_deployment",
                "priority": "high",
                "confidence": 0.85,
                "reasoning": "Core team AI/ML model issue"
            })
        else:
            return json.dumps({
                "category": "query",
                "subcategory": "technical_guidance",
                "priority": "medium",
                "confidence": 0.7,
                "reasoning": "Core team general technical guidance"
            })
    
    # Fallback
    return json.dumps({
        "category": "query",
        "subcategory": "unknown",
        "priority": "medium",
        "confidence": 0.5,
        "reasoning": "General classification"
    })


async def test_team_classification(session: ClientSession, request: str, team: str) -> Dict[str, Any]:
    """Test classification for a specific team"""
    
    # Get team-specific prompt
    prompt_result = await session.call_tool(
        "get_classification_prompt",
        arguments={"user_request": request, "team": team}
    )
    prompt = prompt_result.content[0].text if prompt_result.content else ""
    
    # Mock LLM processing
    llm_response = await mock_llm_classify_by_team(prompt, request, team)
    
    # Get classification
    result = await session.call_tool(
        "classify_support_request",
        arguments={
            "user_request": request,
            "team": team,
            "llm_response": llm_response
        }
    )
    
    result_text = result.content[0].text if result.content else "{}"
    return json.loads(result_text)


async def run_multi_team_tests():
    """Test multi-team classification scenarios"""
    
    server_params = StdioServerParameters(
        command="python",
        args=["-m", "mcp_servers.classification_server"],
        env={"MCP_LOG_LEVEL": "WARNING"}
    )
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            logger.info("🏢 Testing Multi-Team Classification System")
            logger.info("=" * 60)
            
            # Test team management tools
            logger.info("📋 1. Testing team management tools")
            
            # List teams
            teams_result = await session.call_tool("list_teams", arguments={})
            teams = json.loads(teams_result.content[0].text)
            logger.info(f"Available teams: {[team['name'] for team in teams]}")
            
            # Get team info for each team
            for team_info in teams:
                team_name = team_info['name']
                info_result = await session.call_tool("get_team_info", arguments={"team": team_name})
                info = json.loads(info_result.content[0].text)
                logger.info(f"  {team_name}: {info['description']}")
            
            logger.info("")
            
            # Test ATRS team scenarios
            logger.info("🏦 2. Testing ATRS team classifications")
            atrs_scenarios = [
                "Can you bless this deployment to production?",
                "Risk management system is completely down", 
                "Position reconciliation showing $2M discrepancy",
                "Trade settlement feed from clearing house stopped working"
            ]
            
            for scenario in atrs_scenarios:
                classification = await test_team_classification(session, scenario, "atrs")
                logger.info(f"Request: {scenario[:40]}...")
                logger.info(f"  → Team: {classification['team']}")
                logger.info(f"  → Category: {classification['category']}")
                logger.info(f"  → Subcategory: {classification['subcategory']}")
                logger.info(f"  → Priority: {classification['priority']}")
                logger.info(f"  → Workflow: {classification['suggested_workflow']}")
                logger.info("")
            
            # Test Core team scenarios  
            logger.info("🔧 3. Testing Core team classifications")
            core_scenarios = [
                "Nightly ETL job failed with timeout error",
                "Docker build is failing in the deployment pipeline",
                "PostgreSQL queries are running extremely slow",
                "Kubernetes cluster is hitting memory limits",
                "ML model inference API is returning 500 errors"
            ]
            
            for scenario in core_scenarios:
                classification = await test_team_classification(session, scenario, "core")
                logger.info(f"Request: {scenario[:40]}...")
                logger.info(f"  → Team: {classification['team']}")
                logger.info(f"  → Category: {classification['category']}")
                logger.info(f"  → Subcategory: {classification['subcategory']}")
                logger.info(f"  → Priority: {classification['priority']}")
                logger.info(f"  → Workflow: {classification['suggested_workflow']}")
                logger.info("")
            
            # Test team categories
            logger.info("📊 4. Testing team-specific categories")
            for team_name in ["atrs", "core"]:
                categories_result = await session.call_tool("get_team_categories", arguments={"team": team_name})
                categories = json.loads(categories_result.content[0].text)
                logger.info(f"{team_name.upper()} categories: {list(categories.keys())}")
            
            logger.info("")
            logger.info("🎉 Multi-team classification tests completed successfully!")


async def main():
    try:
        await run_multi_team_tests()
        return True
    except Exception as e:
        logger.error(f"Tests failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    result = asyncio.run(main())
    exit(0 if result else 1)