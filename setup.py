#!/usr/bin/env python3
"""
Setup script for V MCP Server
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description):
    """Run a command and return success status."""
    print(f"→ {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed:")
        print(f"   Error: {e.stderr}")
        return False

def main():
    """Main setup function."""
    print("V MCP Server Setup")
    print("=" * 40)

    # Check Python version
    python_version = sys.version_info
    if python_version < (3, 10):
        print(f"✗ Python 3.10+ required. You have Python {python_version.major}.{python_version.minor}")
        sys.exit(1)
    else:
        print(f"✓ Python {python_version.major}.{python_version.minor}.{python_version.micro} detected")

    # Check if we're in the right directory
    if not Path("main.py").exists():
        print("✗ Error: main.py not found. Please run this script from the v-mcp-server directory.")
        sys.exit(1)

    # Check if requirements.txt exists
    if not Path("requirements.txt").exists():
        print("✗ Error: requirements.txt not found.")
        sys.exit(1)

    # Install requirements
    if not run_command(f"{sys.executable} -m pip install -r requirements.txt", "Installing dependencies"):
        sys.exit(1)

    # Check if V repository is accessible
    v_repo_path = Path("../")
    if not (v_repo_path / "README.md").exists():
        print("⚠ Warning: V repository not found in parent directory.")
        print("   Please ensure you're running this from within the V repository,")
        print("   or set the V_REPO_PATH environment variable to point to the V repository.")
        print("")
        print("   Example:")
        print("   export V_REPO_PATH=/path/to/v/repository")
    else:
        print("✓ V repository found in parent directory")

    # Test the server
    print("\nTesting MCP server...")
    test_success = run_command(f"{sys.executable} test_server.py", "Running server tests")

    print("\n" + "=" * 40)
    if test_success:
        print("🎉 Setup completed successfully!")
        print("")
        print("To start the MCP server:")
        print("  python main.py")
        print("")
        print("For more information, see README.md")
    else:
        print("❌ Setup completed with warnings.")
        print("   The server may not work correctly.")
        print("   Please check the error messages above.")

if __name__ == "__main__":
    main()
