#!/usr/bin/env python3
"""Test knowledge server search functionality"""

import asyncio
from pathlib import Path
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def test_knowledge_search():
    """Test knowledge server search"""
    
    server_script = Path("mcp_servers/knowledge_server.py")
    
    server_params = StdioServerParameters(
        command="python",
        args=[str(server_script), "--connection", "stdio"],
        env={"PYTHONUNBUFFERED": "1"}
    )
    
    try:
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                
                print("Testing knowledge search...")
                
                # Test the exact query used in demo
                result = await session.call_tool(
                    "search_knowledge",
                    {
                        "query": "Data reconciliation shows discrepancies in position reports",
                        "category": "data_issue",
                        "max_results": 3
                    }
                )
                
                print("Search result:")
                print(result.content[0].text)
                
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_knowledge_search())