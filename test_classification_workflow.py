#!/usr/bin/env python3
"""Test updated client-side classification with team parameters"""

import asyncio
import json
import logging
from typing import Any, Dict

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("updated-client-test")


async def mock_llm_call(prompt: str, team: str) -> str:
    """Mock LLM call that returns team-appropriate classification"""
    logger.info(f"Mock LLM processing prompt for team: {team}")
    
    if team == "atrs":
        return json.dumps({
            "category": "query",
            "subcategory": "feed_issue",
            "priority": "high", 
            "confidence": 0.85,
            "reasoning": "ATRS trade feed issue detected - requires technical troubleshooting"
        }, indent=2)
    elif team == "core":
        return json.dumps({
            "category": "database",
            "subcategory": "db_performance",
            "priority": "high",
            "confidence": 0.9,
            "reasoning": "Core database performance issue requiring optimization"
        }, indent=2)
    else:
        return json.dumps({
            "category": "query",
            "subcategory": "unknown",
            "priority": "medium",
            "confidence": 0.6,
            "reasoning": "General technical query"
        }, indent=2)


async def test_team_classification_workflow(session: ClientSession, request: str, team: str):
    """Test the full classification workflow for a team"""
    
    logger.info(f"🔍 Testing workflow for team '{team}': {request}")
    
    # Step 1: Get team-specific prompt
    logger.info("  📝 Step 1: Getting team-specific classification prompt...")
    prompt_result = await session.call_tool(
        "get_classification_prompt",
        arguments={"user_request": request, "team": team}
    )
    
    prompt = prompt_result.content[0].text if prompt_result.content else ""
    logger.info(f"  Received prompt ({len(prompt)} chars) for team {team}")
    
    # Step 2: Process with mock LLM
    logger.info("  🤖 Step 2: Processing with mock LLM...")
    llm_response = await mock_llm_call(prompt, team)
    
    # Step 3: Send back for parsing
    logger.info("  📊 Step 3: Sending LLM response for parsing...")
    classification_result = await session.call_tool(
        "classify_support_request",
        arguments={
            "user_request": request,
            "team": team,
            "llm_response": llm_response
        }
    )
    
    # Parse and display result
    result_text = classification_result.content[0].text if classification_result.content else "{}"
    classification = json.loads(result_text)
    
    logger.info("  ✅ Classification completed:")
    logger.info(f"    Team: {classification.get('team')}")
    logger.info(f"    Category: {classification.get('category')}")
    logger.info(f"    Subcategory: {classification.get('subcategory')}")
    logger.info(f"    Priority: {classification.get('priority')}")
    logger.info(f"    Confidence: {classification.get('confidence')}")
    logger.info(f"    Workflow: {classification.get('suggested_workflow')}")
    logger.info("")
    
    return classification


async def test_team_management_tools(session: ClientSession):
    """Test team management and discovery tools"""
    
    logger.info("🏢 Testing team management tools...")
    
    # List all teams
    teams_result = await session.call_tool("list_teams", arguments={})
    teams = json.loads(teams_result.content[0].text)
    logger.info(f"  Available teams: {[t['name'] for t in teams]}")
    
    # Test each team
    for team_info in teams:
        team_name = team_info['name']
        logger.info(f"  📋 Team: {team_name}")
        
        # Get team info
        info_result = await session.call_tool("get_team_info", arguments={"team": team_name})
        info = json.loads(info_result.content[0].text)
        logger.info(f"    Description: {info['description']}")
        
        # Get team categories
        categories_result = await session.call_tool("get_team_categories", arguments={"team": team_name})
        categories = json.loads(categories_result.content[0].text)
        logger.info(f"    Categories: {list(categories.keys())}")
        
        # Test category details for first category
        if categories:
            first_category = list(categories.keys())[0]
            details_result = await session.call_tool(
                "get_category_details", 
                arguments={"team": team_name, "category": first_category}
            )
            details = json.loads(details_result.content[0].text)
            logger.info(f"    Sample category '{first_category}': {details.get('description', 'No description')}")
    
    logger.info("")


async def test_updated_client_side():
    """Test the updated client-side classification with teams"""
    
    server_params = StdioServerParameters(
        command="python",
        args=["-m", "mcp_servers.classification_server"],
        env={"MCP_LOG_LEVEL": "WARNING"}
    )
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize
            logger.info("🚀 Initializing client session...")
            await session.initialize()
            logger.info("✅ Client session initialized")
            
            # Test team management
            await test_team_management_tools(session)
            
            # Test ATRS team classification
            atrs_request = "Sales trade did not feed to MarkitWire, can you assist?"
            atrs_classification = await test_team_classification_workflow(session, atrs_request, "atrs")
            
            # Verify ATRS results
            assert atrs_classification.get('team') == 'atrs'
            assert atrs_classification.get('category') == 'query'
            assert atrs_classification.get('subcategory') == 'feed_issue'
            
            # Test Core team classification  
            core_request = "PostgreSQL queries are running extremely slow"
            core_classification = await test_team_classification_workflow(session, core_request, "core")
            
            # Verify Core results
            assert core_classification.get('team') == 'core'
            assert core_classification.get('category') == 'database'
            assert core_classification.get('subcategory') == 'db_performance'
            
            logger.info("🎉 All team-based classification tests PASSED!")
            return True


async def main():
    try:
        success = await test_updated_client_side()
        if success:
            logger.info("✅ Updated multi-team classification system working perfectly!")
        return success
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    result = asyncio.run(main())
    exit(0 if result else 1)