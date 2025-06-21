#!/usr/bin/env python3
"""Test error handling and edge cases for multi-team classification"""

import asyncio
import json
import logging

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("error-test")


async def test_error_cases():
    """Test various error conditions and edge cases"""
    
    server_params = StdioServerParameters(
        command="python",
        args=["-m", "mcp_servers.classification_server"],
        env={"MCP_LOG_LEVEL": "WARNING"}
    )
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            logger.info("🧪 Testing error handling and edge cases...")
            logger.info("=" * 50)
            
            # Test 1: Missing team parameter
            logger.info("1. Testing missing team parameter...")
            try:
                result = await session.call_tool(
                    "classify_support_request",
                    arguments={"user_request": "Test request"}
                    # Missing team parameter
                )
                result_text = result.content[0].text if result.content else ""
                if "Team parameter is required" in result_text:
                    logger.info("✅ Correctly returned error message for missing team")
                else:
                    logger.error("❌ Should have returned error for missing team parameter")
                    return False
            except Exception as e:
                logger.info(f"✅ Correctly failed with exception: {str(e)}")
            
            # Test 2: Invalid team name
            logger.info("2. Testing invalid team name...")
            try:
                result = await session.call_tool(
                    "get_team_info",
                    arguments={"team": "nonexistent_team"}
                )
                result_text = result.content[0].text if result.content else "{}"
                result_data = json.loads(result_text)
                if "error" in result_data:
                    logger.info(f"✅ Correctly returned error: {result_data['error']}")
                else:
                    logger.error("❌ Should have returned error for invalid team")
                    return False
            except Exception as e:
                logger.info(f"✅ Correctly failed: {str(e)}")
            
            # Test 3: Invalid category for team
            logger.info("3. Testing invalid category for team...")
            try:
                result = await session.call_tool(
                    "get_category_details",
                    arguments={"team": "atrs", "category": "nonexistent_category"}
                )
                result_text = result.content[0].text if result.content else "{}"
                result_data = json.loads(result_text)
                if "error" in result_data:
                    logger.info(f"✅ Correctly returned error: {result_data['error']}")
                else:
                    logger.error("❌ Should have returned error for invalid category")
                    return False
            except Exception as e:
                logger.info(f"✅ Correctly failed: {str(e)}")
            
            # Test 4: Empty request text
            logger.info("4. Testing empty request text...")
            try:
                result = await session.call_tool(
                    "get_classification_prompt",
                    arguments={"user_request": "", "team": "atrs"}
                )
                prompt = result.content[0].text if result.content else ""
                if len(prompt) > 100:  # Should still return a valid prompt
                    logger.info("✅ Handled empty request gracefully")
                else:
                    logger.error("❌ Should return valid prompt even for empty request")
                    return False
            except Exception as e:
                logger.error(f"❌ Should handle empty request: {str(e)}")
                return False
            
            # Test 5: Malformed LLM response
            logger.info("5. Testing malformed LLM response...")
            try:
                result = await session.call_tool(
                    "classify_support_request",
                    arguments={
                        "user_request": "Test request",
                        "team": "atrs",
                        "llm_response": "This is not valid JSON"
                    }
                )
                result_text = result.content[0].text if result.content else ""
                if "LLM returned invalid JSON" in result_text:
                    logger.info("✅ Correctly returned error for malformed JSON")
                else:
                    logger.error("❌ Should have returned error for invalid JSON")
                    return False
            except Exception as e:
                logger.info(f"✅ Correctly failed with exception: {str(e)}")
            
            # Test 6: Valid workflow - ensure basic functionality still works
            logger.info("6. Testing valid workflow still works...")
            try:
                # Get prompt
                prompt_result = await session.call_tool(
                    "get_classification_prompt",
                    arguments={"user_request": "Test request", "team": "atrs"}
                )
                
                # Valid classification
                llm_response = json.dumps({
                    "category": "query",
                    "subcategory": "technical_guidance",
                    "priority": "medium",
                    "confidence": 0.8,
                    "reasoning": "Test classification"
                })
                
                result = await session.call_tool(
                    "classify_support_request",
                    arguments={
                        "user_request": "Test request",
                        "team": "atrs", 
                        "llm_response": llm_response
                    }
                )
                
                result_text = result.content[0].text if result.content else "{}"
                classification = json.loads(result_text)
                
                if classification.get('team') == 'atrs' and classification.get('category') == 'query':
                    logger.info("✅ Valid workflow still works correctly")
                else:
                    logger.error("❌ Valid workflow failed")
                    return False
                    
            except Exception as e:
                logger.error(f"❌ Valid workflow should work: {str(e)}")
                return False
            
            # Test 7: List operations work
            logger.info("7. Testing list operations...")
            try:
                # List teams
                teams_result = await session.call_tool("list_teams", arguments={})
                teams = json.loads(teams_result.content[0].text)
                
                if len(teams) == 2 and any(t['name'] == 'atrs' for t in teams):
                    logger.info("✅ List teams works correctly")
                else:
                    logger.error("❌ List teams returned unexpected result")
                    return False
                    
            except Exception as e:
                logger.error(f"❌ List operations should work: {str(e)}")
                return False
            
            logger.info("")
            logger.info("🎉 All error handling tests PASSED!")
            return True


async def main():
    try:
        success = await test_error_cases()
        if success:
            logger.info("✅ Error handling and edge cases working correctly!")
        return success
    except Exception as e:
        logger.error(f"❌ Error handling test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    result = asyncio.run(main())
    exit(0 if result else 1)