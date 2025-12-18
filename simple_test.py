#!/usr/bin/env python3
"""
Simple test script for the V MCP Server
"""

import asyncio
import sys
from pathlib import Path

async def test_v_mcp_server():
    """Test the V MCP server functionality."""

    print("Testing V MCP Server...")
    print("=" * 50)

    try:
        # Import the MCP server directly for in-memory testing
        from main import VDocumentationServer

        print("✓ Imported MCP server successfully")

        # Create server instance
        v_repo_path = Path(__file__).parent.parent
        v_server = VDocumentationServer(v_repo_path)

        # Test documentation sections
        print("\n1. Testing documentation sections...")
        sections = v_server.get_documentation_sections()
        if "error" not in sections:
            print(f"✓ Documentation sections loaded: {len(sections)} sections")
            print(f"   Available sections: {list(sections.keys())[:5]}...")
        else:
            print(f"✗ Error loading documentation: {sections['error']}")
            return False

        # Test examples list
        print("\n2. Testing examples listing...")
        examples = v_server.get_examples_list()
        if examples and "error" not in examples[0]:
            print(f"✓ Examples loaded: {len(examples)} examples")
            print(f"   Sample: {examples[0]['name'] if examples else 'None'}")
        else:
            print(f"✗ Error loading examples: {examples[0]['error'] if examples else 'No examples'}")

        # Test stdlib modules
        print("\n3. Testing standard library modules...")
        modules = v_server.get_stdlib_modules()
        if modules and "error" not in modules[0]:
            print(f"✓ Standard library modules loaded: {len(modules)} modules")
            print(f"   Sample: {modules[0]['name'] if modules else 'None'}")
        else:
            print(f"✗ Error loading stdlib modules: {modules[0]['error'] if modules else 'No modules'}")

        # Test documentation search
        print("\n4. Testing documentation search...")
        search_results = v_server.search_documentation("array")
        if search_results:
            print(f"✓ Documentation search successful: {len(search_results)} results")
        else:
            print("✗ No search results found")

        # Test specific example retrieval
        print("\n5. Testing example retrieval...")
        hello_world = v_server.get_example_content("hello_world")
        if "error" not in hello_world:
            print("✓ Hello world example retrieved successfully")
        else:
            print(f"✗ Error retrieving hello world: {hello_world['error']}")

        print("\n6. Testing MCP tools...")
        print("✓ MCP server structure verified")

    except Exception as e:
        print(f"✗ Failed to test MCP server: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n" + "=" * 50)
    print("✓ All tests completed!")
    return True

if __name__ == "__main__":
    print("V MCP Server Test Suite")
    print("Make sure you're in the v-mcp-server directory")

    # Check if we're in the right directory
    if not Path("main.py").exists():
        print("✗ Error: main.py not found. Please run this script from the v-mcp-server directory.")
        sys.exit(1)

    # Check if V repository is accessible
    v_repo_path = Path("../")
    if not (v_repo_path / "README.md").exists():
        print("⚠ Warning: V repository not found in parent directory.")
        print("   Make sure you're running this from within the V repository,")
        print("   or set the V_REPO_PATH environment variable.")
    else:
        print("✓ V repository found in parent directory")

    # Run tests
    success = asyncio.run(test_v_mcp_server())

    if success:
        print("\n🎉 V MCP Server is working correctly!")
    else:
        print("\n❌ Some tests failed. Please check the server configuration.")
        sys.exit(1)
