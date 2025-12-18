# V Language MCP Server

A comprehensive Model Context Protocol (MCP) server that provides LLMs with extensive knowledge about the V programming language, including documentation, code examples, standard library reference, and syntax explanations.

## Features

### 📚 Documentation Access
- **Full V documentation**: Access the complete V programming language documentation
- **Section-specific queries**: Get detailed information about specific language features
- **Search functionality**: Search through documentation for specific topics

### 💡 Code Examples
- **Example browser**: Browse through hundreds of V code examples
- **Code retrieval**: Get complete source code for any example
- **Pattern search**: Search through examples for specific coding patterns

### 🔧 Standard Library Reference
- **Module listing**: Browse all available V standard library modules
- **Module details**: Get comprehensive information about specific modules
- **File listings**: See what functions and types are available in each module

### 🎯 Syntax Explanations
- **Feature explanations**: Detailed explanations of V language features
- **Code examples**: Practical examples for each concept
- **Quick reference**: Fast access to common syntax patterns

### 🚀 Code Execution (Planned)
- **Code validation**: Check V code syntax and semantics
- **Example execution**: Run V examples and see their output
- **Interactive learning**: Learn by running and modifying code

## Installation

### Prerequisites
- Python 3.10+
- Access to the V language repository (this project should be run from within the V repo or with V_REPO_PATH configured)

### Setup
1. Clone or navigate to the V repository
2. Navigate to the MCP server directory:
   ```bash
   cd v-mcp-server
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. (Optional) Configure the V repository path:
   ```bash
   export V_REPO_PATH="/path/to/v/repository"
   ```
   If not set, the server assumes it's running from within the V repository.

## Usage

### Starting the Server

#### STDIO Mode (Default)
```bash
python main.py
```

#### HTTP Mode
```bash
python main.py --transport http --host 127.0.0.1 --port 8000
```

#### SSE Mode
```bash
python main.py --transport sse --host 127.0.0.1 --port 8000
```

### Available Tools

#### Documentation Tools
- `get_v_documentation([section])` - Get V documentation, optionally for a specific section
- `search_v_docs(query)` - Search through V documentation for specific topics

#### Example Tools
- `list_v_examples()` - List all available V code examples
- `get_v_example(name)` - Get the complete source code for a specific example
- `search_v_examples(query)` - Search through example code for patterns

#### Standard Library Tools
- `list_v_stdlib_modules()` - List all V standard library modules
- `get_v_module_info(module_name)` - Get detailed information about a specific module

#### Language Reference Tools
- `explain_v_syntax(feature)` - Get detailed explanation of V language features
- `get_v_quick_reference()` - Get a quick reference guide for V basics

## MCP Client Integration

### Using with Claude Desktop
Add this to your Claude Desktop configuration:

```json
{
  "mcpServers": {
    "v-language": {
      "command": "python",
      "args": ["/path/to/v-mcp-server/main.py"],
      "env": {
        "V_REPO_PATH": "/path/to/v/repository"
      }
    }
  }
}
```

### Using with Other MCP Clients
The server supports all standard MCP transports:
- **STDIO**: Best for local development and testing
- **HTTP**: Suitable for web applications and remote access
- **SSE**: Good for real-time applications

## Examples

### Getting Started with V
```
User: How do I get started with V programming?
Assistant: Let me help you get started with V programming. First, let me get the basic documentation overview.

[get_v_documentation] -> Returns overview of V documentation sections

Then let me get the quick reference guide to show you the basics.

[get_v_quick_reference] -> Returns quick reference with essential syntax
```

### Finding Code Examples
```
User: Show me how to work with arrays in V
Assistant: I'll show you how arrays work in V by looking at examples and documentation.

[explain_v_syntax("arrays")] -> Returns detailed explanation of V arrays

Let me also find some practical examples.

[search_v_examples("array")] -> Returns examples that use arrays
```

### Standard Library Exploration
```
User: What modules are available in V's standard library?
Assistant: Let me show you the available modules in V's standard library.

[list_v_stdlib_modules] -> Returns list of all stdlib modules

If you're interested in a specific module, I can provide more details.

[get_v_module_info("os")] -> Returns detailed info about the os module
```

### Learning Specific Features
```
User: How do structs work in V?
Assistant: Let me explain V structs with examples.

[explain_v_syntax("structs")] -> Returns comprehensive explanation of structs

Let me also show you a practical example.

[get_v_example("struct")] -> Returns example code using structs
```

## Configuration

### Environment Variables
- `V_REPO_PATH`: Path to the V language repository (defaults to parent directory)

### Server Options
- `--transport`: Transport protocol (`stdio`, `http`, `sse`)
- `--host`: Host address (for HTTP/SSE transports)
- `--port`: Port number (for HTTP/SSE transports)
- `--path`: URL path (for HTTP transport)

## Architecture

The MCP server is built with the following components:

### VDocumentationServer Class
- Handles file system operations within the V repository
- Caches documentation and example data for performance
- Provides search functionality across documentation and code

### MCP Tools
Each tool corresponds to a specific functionality:
- **Documentation tools**: Access and search V language documentation
- **Example tools**: Browse and retrieve code examples
- **Standard library tools**: Explore V's standard library modules
- **Syntax tools**: Explain language features and provide quick reference

### FastMCP Framework
Built on the FastMCP library, providing:
- **Multiple transport support**: STDIO, HTTP, SSE
- **Type safety**: Full type annotations and validation
- **Performance**: Efficient caching and search algorithms
- **Extensibility**: Easy to add new tools and features

## Development

### Adding New Tools
To add a new tool, create a function decorated with `@mcp.tool`:

```python
@mcp.tool
def new_tool_name(param1: str, param2: int) -> str:
    """Tool description."""
    # Implementation
    return result
```

### Extending Functionality
The server can be extended by:
- Adding new tools for specific V features
- Implementing code execution capabilities
- Adding integration with V's compiler
- Creating interactive tutorials

### Testing
Run tests with pytest:
```bash
pytest tests/
```

## License

MIT

## Related Projects

- [V Programming Language](https://github.com/vlang/v) - The main V language repository
- [FastMCP](https://github.com/jlowin/fastmcp) - The MCP framework used by this server
- [v-analyzer](https://github.com/vlang/v-analyzer) - V language server for IDE support
