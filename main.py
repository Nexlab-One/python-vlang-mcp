#!/usr/bin/env python3
"""
V Language MCP Server

A Model Context Protocol server that provides comprehensive information about
the V programming language to help LLMs understand and generate V code.
"""

import os
import re
import json
import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from functools import lru_cache
from fastmcp import FastMCP
import asyncio
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class VServerConfig:
    """Configuration for the V MCP Server."""
    v_repo_path: Path
    v_ui_path: Optional[Path] = None
    cache_ttl_seconds: int = 300  # 5 minutes default
    max_search_results: int = 50
    log_level: str = "INFO"

    @classmethod
    def from_env(cls) -> 'VServerConfig':
        """Create configuration from environment variables."""
        # V repository path
        v_repo_path = os.getenv('V_REPO_PATH')
        if v_repo_path:
            repo_path = Path(v_repo_path)
        else:
            # Default to parent directory
            repo_path = Path(__file__).parent.parent

        # V UI repository path (optional)
        v_ui_path = os.getenv('V_UI_PATH')
        if v_ui_path:
            ui_path = Path(v_ui_path)
        else:
            # Default to v-ui submodule in parent directory
            ui_path = repo_path / "v-ui"
            if not ui_path.exists():
                ui_path = None

        # Cache TTL
        cache_ttl = int(os.getenv('V_CACHE_TTL_SECONDS', '300'))

        # Max search results
        max_results = int(os.getenv('V_MAX_SEARCH_RESULTS', '50'))

        # Log level
        log_level = os.getenv('V_LOG_LEVEL', 'INFO').upper()

        return cls(
            v_repo_path=repo_path,
            v_ui_path=ui_path,
            cache_ttl_seconds=cache_ttl,
            max_search_results=max_results,
            log_level=log_level
        )

# Initialize FastMCP server
mcp = FastMCP("V Language Assistant")

# Load configuration
config = VServerConfig.from_env()

# Configure logging level
logging.getLogger().setLevel(getattr(logging, config.log_level, logging.INFO))

class VDocumentationServer:
    """Server for providing V language documentation and examples."""

    def __init__(self, config: VServerConfig):
        self.config = config
        self.v_repo_path = config.v_repo_path
        self.docs_path = config.v_repo_path / "doc"
        self.examples_path = config.v_repo_path / "examples"
        self.vlib_path = config.v_repo_path / "vlib"
        
        # V UI paths (optional)
        self.v_ui_path = config.v_ui_path
        self.v_ui_examples_path = config.v_ui_path / "examples" if config.v_ui_path else None
        self.v_ui_docs_path = config.v_ui_path / "docs.md" if config.v_ui_path else None

        # Cache with TTL (time-to-live) in seconds
        self._cache = {}
        self._cache_ttl = config.cache_ttl_seconds
        self._cache_timestamps = {}
        self._max_search_results = config.max_search_results

        # Store path validation results for graceful degradation
        self._path_status = self._validate_paths()

    def _validate_paths(self) -> Dict[str, bool]:
        """Validate that required paths exist and return status."""
        path_status = {
            "docs": self.docs_path.exists(),
            "examples": self.examples_path.exists(),
            "stdlib": self.vlib_path.exists(),
            "v_ui": self.v_ui_path.exists() if self.v_ui_path else False,
            "v_ui_examples": self.v_ui_examples_path.exists() if self.v_ui_examples_path else False
        }

        missing_paths = []
        for component, exists in path_status.items():
            if not exists and component not in ["v_ui", "v_ui_examples"]:  # V UI is optional
                path = getattr(self, f"{component}_path", None)
                if path:
                    missing_paths.append(f"{component.title()}: {path}")

        if missing_paths:
            logger.warning(f"Some V repository components are missing: {', '.join(missing_paths)}")
            logger.warning("Server functionality will be limited to available components")
        else:
            logger.info("All V repository components found successfully")
        
        if path_status.get("v_ui"):
            logger.info("V UI repository found and will be indexed")
        elif self.v_ui_path:
            logger.info(f"V UI repository path specified but not found: {self.v_ui_path}")

        return path_status

    def _get_cache(self, key: str) -> Any:
        """Get item from cache if it exists and hasn't expired."""
        if key in self._cache:
            if time.time() - self._cache_timestamps.get(key, 0) < self._cache_ttl:
                return self._cache[key]
            else:
                # Remove expired entry
                del self._cache[key]
                del self._cache_timestamps[key]
        return None

    def _set_cache(self, key: str, value: Any) -> None:
        """Store item in cache with current timestamp."""
        self._cache[key] = value
        self._cache_timestamps[key] = time.time()

    def _clear_expired_cache(self) -> None:
        """Remove all expired cache entries."""
        current_time = time.time()
        expired_keys = [
            key for key, timestamp in self._cache_timestamps.items()
            if current_time - timestamp >= self._cache_ttl
        ]
        for key in expired_keys:
            del self._cache[key]
            del self._cache_timestamps[key]

    def clear_cache(self) -> Dict[str, int]:
        """Clear all cache entries and return statistics."""
        cache_count = len(self._cache)
        timestamp_count = len(self._cache_timestamps)

        self._cache.clear()
        self._cache_timestamps.clear()

        return {
            "cleared_entries": cache_count,
            "cleared_timestamps": timestamp_count,
            "message": f"Cleared {cache_count} cache entries"
        }

    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics without clearing."""
        return {
            "entries": len(self._cache),
            "timestamps": len(self._cache_timestamps),
            "ttl_seconds": self._cache_ttl
        }

    def _validate_query(self, query: str, min_length: int = 2) -> str:
        """Validate and sanitize search query."""
        if not query or len(query.strip()) < min_length:
            raise ValueError(f"Query must be at least {min_length} characters long")
        return query.strip()

    def _read_file_content(self, file_path: Path) -> str:
        """Read file content safely."""
        try:
            if not file_path.exists():
                return f"Error: File not found: {file_path}"

            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                return f.read()
        except PermissionError:
            return f"Error: Permission denied reading file {file_path}"
        except UnicodeDecodeError:
            return f"Error: Unable to decode file {file_path} (encoding issue)"
        except Exception as e:
            logger.error(f"Unexpected error reading file {file_path}: {e}")
            return f"Error reading file {file_path}: {str(e)}"

    def _search_in_file(self, file_path: Path, pattern: str, context_lines: int = 3) -> List[Dict]:
        """Search for pattern in file and return matches with enhanced context."""
        try:
            if not file_path.exists():
                return [{'error': f'File not found: {file_path}'}]

            if not pattern or len(pattern.strip()) < 1:
                return [{'error': 'Search pattern cannot be empty'}]

            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()

            matches = []
            # Create case-insensitive pattern with word boundaries for better matching
            search_pattern = r'\b' + re.escape(pattern) + r'\b'
            compiled_pattern = re.compile(search_pattern, re.IGNORECASE | re.MULTILINE)

            lines = content.split('\n')
            for i, line in enumerate(lines):
                if compiled_pattern.search(line):
                    # Enhanced context: try to get paragraph-level context
                    start_line = max(0, i - context_lines)
                    end_line = min(len(lines), i + context_lines + 1)

                    # Look for paragraph boundaries (empty lines)
                    while start_line > 0 and lines[start_line - 1].strip():
                        start_line -= 1
                    while end_line < len(lines) and lines[end_line].strip():
                        end_line += 1

                    context_lines_list = lines[start_line:end_line]
                    context = '\n'.join(context_lines_list).strip()

                    # Calculate relevance score based on:
                    # - Exact match bonus
                    # - Position in line (earlier = higher score)
                    # - Context richness
                    score = 1.0
                    if pattern.lower() in line.lower():
                        score += 0.5  # Exact match bonus
                    if line.lower().startswith(pattern.lower()):
                        score += 0.3  # Starts with pattern bonus
                    if len(context) > len(line):
                        score += 0.2  # Rich context bonus

                    matches.append({
                        'line': i + 1,
                        'content': line.strip(),
                        'context': context,
                        'file': str(file_path.relative_to(self.v_repo_path)),
                        'score': score,
                        'pattern': pattern
                    })

            # Sort by relevance score (highest first)
            matches.sort(key=lambda x: x['score'], reverse=True)
            return matches

        except re.error as e:
            return [{'error': f'Invalid regex pattern: {pattern} - {str(e)}'}]
        except PermissionError:
            return [{'error': f'Permission denied reading file: {file_path}'}]
        except Exception as e:
            logger.error(f'Unexpected error searching file {file_path}: {e}')
            return [{'error': f'Error searching file {file_path}: {str(e)}'}]

    def get_documentation_sections(self) -> Dict[str, str]:
        """Extract main sections from V documentation."""
        cache_key = "docs_sections"

        # Try cache first
        cached_result = self._get_cache(cache_key)
        if cached_result:
            return cached_result

        docs_file = self.docs_path / "docs.md"
        if not docs_file.exists():
            result = {"error": "Documentation file not found"}
            self._set_cache(cache_key, result)
            return result

        content = self._read_file_content(docs_file)

        # Split by main headers
        sections = {}
        current_section = None
        current_content = []

        for line in content.split('\n'):
            if line.startswith('# '):
                if current_section:
                    sections[current_section] = '\n'.join(current_content)
                current_section = line[2:].strip()
                current_content = [line]
            elif line.startswith('## '):
                if current_section:
                    sections[current_section] = '\n'.join(current_content)
                current_section = line[3:].strip()
                current_content = [line]
            else:
                current_content.append(line)

        if current_section:
            sections[current_section] = '\n'.join(current_content)

        # Cache the result
        self._set_cache(cache_key, sections)
        return sections

    def search_documentation(self, query: str) -> List[Dict]:
        """Search V documentation for relevant information."""
        try:
            query = self._validate_query(query)
            docs_file = self.docs_path / "docs.md"
            if not docs_file.exists():
                return [{"error": f"Documentation file not found at {docs_file}"}]

            return self._search_in_file(docs_file, query)
        except ValueError as e:
            return [{"error": str(e)}]
        except Exception as e:
            logger.error(f"Error searching documentation: {e}")
            return [{"error": f"Failed to search documentation: {str(e)}"}]

    def get_examples_list(self) -> List[Dict]:
        """Get list of available V examples."""
        cache_key = "examples_list"

        # Try cache first
        cached_result = self._get_cache(cache_key)
        if cached_result:
            return cached_result

        if not self.examples_path.exists():
            result = [{"error": "Examples directory not found"}]
            self._set_cache(cache_key, result)
            return result

        examples = []
        for item in self.examples_path.rglob("*.v"):
            if item.is_file():
                examples.append({
                    'name': item.stem,
                    'path': str(item.relative_to(self.v_repo_path)),
                    'description': self._extract_example_description(item)
                })

        # Cache the result
        self._set_cache(cache_key, examples)
        return examples

    def _extract_example_description(self, file_path: Path) -> str:
        """Extract description from example file comments."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()[:10]  # Check first 10 lines

            for line in lines:
                if line.strip().startswith('//') or line.strip().startswith('/*'):
                    desc = line.strip()[2:].strip()
                    if desc and not desc.startswith('Copyright'):
                        return desc
        except:
            pass
        return f"V example: {file_path.stem}"

    def get_example_content(self, example_name: str) -> Dict:
        """Get content of a specific example."""
        cache_key = f"example_content_{example_name}"

        # Try cache first
        cached_result = self._get_cache(cache_key)
        if cached_result:
            return cached_result

        for item in self.examples_path.rglob(f"{example_name}.v"):
            if item.is_file():
                result = {
                    'name': example_name,
                    'path': str(item.relative_to(self.v_repo_path)),
                    'content': self._read_file_content(item)
                }
                # Cache the result
                self._set_cache(cache_key, result)
                return result

        result = {"error": f"Example '{example_name}' not found"}
        # Cache negative results too to avoid repeated filesystem searches
        self._set_cache(cache_key, result)
        return result

    def search_examples(self, query: str) -> List[Dict]:
        """Search through V examples for patterns."""
        try:
            query = self._validate_query(query)

            if not self.examples_path.exists():
                return [{"error": f"Examples directory not found at {self.examples_path}"}]

            results = []
            v_files = list(self.examples_path.rglob("*.v"))

            if not v_files:
                return [{"error": "No V example files found"}]

            for v_file in v_files[:self._max_search_results]:  # Limit based on configuration
                matches = self._search_in_file(v_file, query)
                if matches:
                    for match in matches:
                        if 'error' not in match:  # Only add successful matches
                            match['example_name'] = v_file.stem
                            results.append(match)

            return results
        except ValueError as e:
            return [{"error": str(e)}]
        except Exception as e:
            logger.error(f"Error searching examples: {e}")
            return [{"error": f"Failed to search examples: {str(e)}"}]

    def get_stdlib_modules(self) -> List[Dict]:
        """Get list of V standard library modules."""
        cache_key = "stdlib_modules"

        # Try cache first
        cached_result = self._get_cache(cache_key)
        if cached_result:
            return cached_result

        if not self.vlib_path.exists():
            result = [{"error": "Standard library directory not found"}]
            self._set_cache(cache_key, result)
            return result

        modules = []
        for item in self.vlib_path.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                readme_file = item / "README.md"
                description = "V standard library module"
                if readme_file.exists():
                    content = self._read_file_content(readme_file)
                    # Extract first meaningful line as description
                    for line in content.split('\n')[:5]:
                        line = line.strip()
                        if line and not line.startswith('#') and len(line) > 10:
                            description = line
                            break

                modules.append({
                    'name': item.name,
                    'path': str(item.relative_to(self.v_repo_path)),
                    'description': description
                })

        sorted_modules = sorted(modules, key=lambda x: x['name'])

        # Cache the result
        self._set_cache(cache_key, sorted_modules)
        return sorted_modules

    def get_module_info(self, module_name: str) -> Dict:
        """Get information about a specific standard library module."""
        module_path = self.vlib_path / module_name
        if not module_path.exists():
            return {"error": f"Module '{module_name}' not found"}

        info = {
            'name': module_name,
            'files': [],
            'readme': None
        }

        # Get README if available
        readme_file = module_path / "README.md"
        if readme_file.exists():
            info['readme'] = self._read_file_content(readme_file)

        # List V files in the module
        for v_file in module_path.rglob("*.v"):
            if v_file.is_file():
                info['files'].append({
                    'name': v_file.name,
                    'path': str(v_file.relative_to(self.v_repo_path)),
                    'size': v_file.stat().st_size
                })

        return info

    def get_v_ui_examples_list(self) -> List[Dict]:
        """Get a list of all V UI examples."""
        cache_key = "v_ui_examples_list"
        
        # Try cache first
        cached_result = self._get_cache(cache_key)
        if cached_result:
            return cached_result

        if not self.v_ui_examples_path or not self.v_ui_examples_path.exists():
            result = [{"error": "V UI examples directory not found"}]
            self._set_cache(cache_key, result)
            return result

        examples = []
        for item in self.v_ui_examples_path.rglob("*.v"):
            if item.is_file():
                examples.append({
                    'name': item.stem,
                    'path': str(item.relative_to(self.v_ui_path)),
                    'full_path': str(item)
                })

        sorted_examples = sorted(examples, key=lambda x: x['name'])

        # Cache the result
        self._set_cache(cache_key, sorted_examples)
        return sorted_examples

    def get_v_ui_example_content(self, example_name: str) -> Dict:
        """Get the content of a specific V UI example."""
        cache_key = f"v_ui_example_{example_name}"

        # Try cache first
        cached_result = self._get_cache(cache_key)
        if cached_result:
            return cached_result

        if not self.v_ui_examples_path or not self.v_ui_examples_path.exists():
            result = {"error": "V UI examples directory not found"}
            self._set_cache(cache_key, result)
            return result

        # Search for the example file
        for item in self.v_ui_examples_path.rglob(f"{example_name}.v"):
            if item.is_file():
                content = self._read_file_content(item)
                result = {
                    'name': example_name,
                    'path': str(item.relative_to(self.v_ui_path)),
                    'content': content
                }
                self._set_cache(cache_key, result)
                return result

        result = {"error": f"V UI example '{example_name}' not found"}
        self._set_cache(cache_key, result)
        return result

    def search_v_ui_examples(self, query: str) -> List[Dict]:
        """Search through V UI examples for specific patterns."""
        try:
            query = self._validate_query(query)
            
            if not self.v_ui_examples_path or not self.v_ui_examples_path.exists():
                return [{"error": f"V UI examples directory not found at {self.v_ui_examples_path}"}]

            results = []
            v_files = list(self.v_ui_examples_path.rglob("*.v"))

            if not v_files:
                return [{"error": "No V UI example files found"}]

            for file_path in v_files:
                matches = self._search_in_file(file_path, query)
                for match in matches:
                    match['source'] = 'v_ui'
                    match['file'] = str(file_path.relative_to(self.v_ui_path))
                    results.append(match)

            # Sort by relevance score and limit results
            results.sort(key=lambda x: x.get('score', 0), reverse=True)
            results = results[:self._max_search_results]

            return results if results else [{"message": f"No matches found for '{query}' in V UI examples"}]

        except ValueError as e:
            return [{"error": str(e)}]
        except Exception as e:
            logger.error(f"Error searching V UI examples: {e}")
            return [{"error": f"Error searching V UI examples: {str(e)}"}]

# Initialize the documentation server
v_server = VDocumentationServer(config)

# MCP Tools
@mcp.tool
def get_v_documentation(section: Optional[str] = None) -> str:
    """
    Get V programming language documentation.

    Provides access to the complete V programming language documentation.
    When no section is specified, returns an overview of all available sections.
    When a specific section is requested, returns detailed content for that section.

    Args:
        section: Optional specific section to retrieve (e.g., 'Structs', 'Functions', 'Modules')

    Returns:
        Documentation content for the requested section or overview of available sections
    """
    try:
        # Check if documentation is available
        if not v_server._path_status.get("docs", False):
            return """# V Documentation - Not Available

❌ **V documentation is not available on this system.**

This could be because:
- The V repository is not properly set up
- The documentation files are missing
- The V_REPO_PATH environment variable points to the wrong location

## Solutions:

1. **Verify V repository location:**
   ```bash
   # Check if you're in the correct directory
   ls -la
   # Should see doc/, examples/, vlib/ directories
   ```

2. **Set V_REPO_PATH if needed:**
   ```bash
   export V_REPO_PATH="/path/to/v/repository"
   ```

3. **Check server configuration:**
   Use `get_v_config()` to see current settings

4. **Restart the MCP server** after fixing the path

**Alternative:** Use `explain_v_syntax(feature)` for specific language features or `get_v_quick_reference()` for basic syntax."""

        sections = v_server.get_documentation_sections()

        if "error" in sections:
            return f"""# V Documentation - Error

❌ **Error loading V documentation:** {sections['error']}

This might be a temporary issue. Try:
- `clear_v_cache()` to refresh cached content
- `get_v_config()` to check server status
- Restarting the MCP server

**Alternative resources:**
- Use `explain_v_syntax(feature)` for specific language features
- Use `get_v_quick_reference()` for basic syntax reference"""

        if section:
            if section in sections:
                return f"# {section}\n\n{sections[section]}"
            else:
                available_sections = list(sections.keys())
                return f"""# Section Not Found

❌ **Section '{section}' not found in V documentation.**

**Available sections:**
{chr(10).join(f"- {sec}" for sec in available_sections[:10])}

**Suggestions:**
- Check spelling and capitalization
- Use `get_v_documentation()` to see all sections
- Try `search_v_docs('{section}')` for related content"""
        else:
            # Return overview of main sections
            overview = "# V Programming Language Documentation\n\n"
            overview += f"✅ **Documentation loaded successfully** ({len(sections)} sections available)\n\n"
            overview += "**Available sections:**\n\n"
            for sec in sections.keys():
                overview += f"- {sec}\n"
            overview += "\n**Usage:** `get_v_documentation(section_name)` to get specific sections."
            return overview

    except Exception as e:
        logger.error(f"Unexpected error in get_v_documentation: {e}")
        return f"""# Documentation Error

❌ **Unexpected error loading V documentation:** {str(e)}

Please try:
- `get_v_config()` to check server status
- `clear_v_cache()` to reset cache
- Restarting the MCP server

**Alternative:** Use `explain_v_syntax(feature)` for specific language features."""

@mcp.tool
def search_v_docs(query: str) -> str:
    """
    Search through V documentation for specific topics.

    Performs full-text search across the V programming language documentation.
    Returns relevant sections with context where the search terms are found.

    Args:
        query: Search term to look for in V documentation (minimum 2 characters)

    Returns:
        Search results with relevant documentation sections and context
    """
    results = v_server.search_documentation(query)

    if not results:
        return f"No results found for '{query}' in V documentation."

    output = f"# Search Results for '{query}'\n\n"
    successful_results = [r for r in results if 'error' not in r]

    if successful_results:
        output += f"Found {len(successful_results)} matches (showing top 10):\n\n"

        for result in successful_results[:10]:  # Show top 10 by relevance score
            output += f"**File:** {result['file']}\n"
            output += f"**Line {result['line']}:** {result['content']}\n"
            output += f"**Context:**\n```\n{result['context']}\n```\n\n"
    else:
        # Show any error messages
        for result in results:
            if 'error' in result:
                output += f"Error: {result['error']}\n\n"

    return output

@mcp.tool
def list_v_examples() -> str:
    """
    Get a list of available V programming examples.

    Returns a comprehensive list of all available V code examples from the repository.
    Each example includes its name, file path, and description extracted from comments.

    Returns:
        Formatted list of example programs with descriptions (shows first 20 examples)
    """
    try:
        # Check if examples are available
        if not v_server._path_status.get("examples", False):
            return """# V Examples - Not Available

❌ **V code examples are not available on this system.**

This could be because:
- The V repository is not properly set up
- The examples directory is missing
- The V_REPO_PATH environment variable points to the wrong location

## Solutions:

1. **Verify examples directory exists:**
   ```bash
   # Check if examples directory exists
   ls -la examples/
   ```

2. **Set V_REPO_PATH if needed:**
   ```bash
   export V_REPO_PATH="/path/to/v/repository"
   ```

3. **Check server configuration:**
   Use `get_v_config()` to see current settings

4. **Restart the MCP server** after fixing the path

**Alternative:** Use `explain_v_syntax(feature)` to learn V language features with code examples."""

        examples = v_server.get_examples_list()

        if not examples:
            return """# No Examples Found

❌ **No V examples were found.**

This might be because:
- The examples directory exists but is empty
- File permissions prevent reading
- The V repository structure has changed

Try:
- `get_v_config()` to check server status
- `clear_v_cache()` to refresh cached content
- Restarting the MCP server"""

        # Check if there are any actual examples (not just error entries)
        valid_examples = [ex for ex in examples if 'error' not in ex]
        error_examples = [ex for ex in examples if 'error' in ex]

        output = "# V Programming Examples\n\n"

        if error_examples:
            output += f"⚠️ **Warning:** {len(error_examples)} example(s) could not be loaded.\n\n"

        if valid_examples:
            output += f"✅ **Found {len(valid_examples)} examples**\n\n"

            for example in valid_examples[:20]:  # Limit to first 20 examples
                output += f"**{example['name']}**\n"
                output += f"- Path: {example['path']}\n"
                output += f"- Description: {example['description']}\n\n"

            if len(valid_examples) > 20:
                output += f"\n*Showing first 20 of {len(valid_examples)} examples.*\n"
                output += "*Use `get_v_example(name)` to see the full code for any example.*"
            else:
                output += "*Use `get_v_example(name)` to see the full code for any example.*"
        else:
            output += "❌ **No valid examples could be loaded.**\n\n"
            output += "**Troubleshooting:**\n"
            output += "- Check file permissions in the examples directory\n"
            output += "- Verify the examples contain .v files\n"
            output += "- Try `get_v_config()` for detailed status"

        return output

    except Exception as e:
        logger.error(f"Unexpected error in list_v_examples: {e}")
        return f"""# Examples Error

❌ **Unexpected error loading V examples:** {str(e)}

Please try:
- `get_v_config()` to check server status
- `clear_v_cache()` to reset cache
- Restarting the MCP server

**Alternative:** Use `explain_v_syntax(feature)` for language feature explanations."""

@mcp.tool
def get_v_example(example_name: str) -> str:
    """
    Get the source code of a specific V example.

    Retrieves the complete source code for a named V programming example.
    The example name should match the filename without the .v extension.

    Args:
        example_name: Name of the example to retrieve (e.g., 'fibonacci', 'hello_world')

    Returns:
        Complete source code of the example with syntax highlighting, or error message if not found
    """
    result = v_server.get_example_content(example_name)

    if 'error' in result:
        return f"""# Example Not Found

❌ **Example '{example_name}' not found.**

**Possible reasons:**
- Incorrect spelling or capitalization
- Example doesn't exist in the repository
- Examples directory is not available

**Suggestions:**
- Use `list_v_examples()` to see all available examples
- Try `search_v_examples('{example_name}')` for similar examples
- Check `get_v_config()` for server status

**Error details:** {result['error']}"""

    output = f"# V Example: {result['name']}\n\n"
    output += f"**Path:** {result['path']}\n\n"
    output += "## Source Code\n\n"
    output += f"```v\n{result['content']}\n```\n"

    return output

@mcp.tool
def search_v_examples(query: str) -> str:
    """
    Search through V example code for specific patterns or features.

    Performs full-text search across all V programming examples in the repository.
    Useful for finding code patterns, specific functions, or language features in use.

    Args:
        query: Search term to look for in example code (minimum 2 characters)

    Returns:
        Examples containing the search term with context and file information
    """
    results = v_server.search_examples(query)

    if not results:
        return f"No examples found containing '{query}'."

    output = f"# Examples containing '{query}'\n\n"
    successful_results = [r for r in results if 'error' not in r]

    if successful_results:
        output += f"Found {len(successful_results)} matches across examples:\n\n"

        current_example = None
        for result in successful_results[:15]:  # Show top 15 results
            if current_example != result['example_name']:
                current_example = result['example_name']
                output += f"## Example: {current_example}\n\n"

            output += f"**File:** {result['file']}\n"
            output += f"**Line {result['line']}:** {result['content']}\n"
            output += f"**Context:**\n```v\n{result['context']}\n```\n\n"
    else:
        # Show any error messages
        for result in results:
            if 'error' in result:
                output += f"Error: {result['error']}\n\n"

    return output

@mcp.tool
def list_v_stdlib_modules() -> str:
    """
    Get a list of V standard library modules.

    Returns a comprehensive list of all available modules in V's standard library.
    Each module includes its name, description, and indicates its functionality.

    Returns:
        Formatted list of standard library modules with descriptions
    """
    modules = v_server.get_stdlib_modules()

    if not modules:
        return "No standard library modules found."

    output = "# V Standard Library Modules\n\n"
    for module in modules:
        if 'error' in module:
            output += f"Error: {module['error']}\n"
        else:
            output += f"**{module['name']}**\n"
            output += f"- Description: {module['description']}\n\n"

    output += "\nUse `get_v_module_info(module_name)` to get detailed information about a specific module."
    return output

@mcp.tool
def get_v_module_info(module_name: str) -> str:
    """
    Get detailed information about a V standard library module.

    Provides comprehensive information about a specific V standard library module,
    including its README documentation (if available) and list of source files.

    Args:
        module_name: Name of the module to get information about (e.g., 'os', 'json', 'net')

    Returns:
        Detailed information including files, documentation, and module structure
    """
    result = v_server.get_module_info(module_name)

    if 'error' in result:
        return f"Error: {result['error']}\n\nUse `list_v_stdlib_modules()` to see available modules."

    output = f"# V Standard Library Module: {result['name']}\n\n"

    if result['readme']:
        output += "## Documentation\n\n"
        output += result['readme']
        output += "\n\n"

    if result['files']:
        output += "## Files\n\n"
        for file_info in result['files'][:10]:  # Limit to first 10 files
            output += f"- **{file_info['name']}** ({file_info['size']} bytes)\n"
            output += f"  - Path: {file_info['path']}\n"

        if len(result['files']) > 10:
            output += f"\n*... and {len(result['files']) - 10} more files*\n"

    return output

@mcp.tool
def explain_v_syntax(feature: str) -> str:
    """
    Explain V programming language syntax and features.

    Provides detailed explanations of V programming language concepts and syntax.
    Includes code examples and practical usage patterns for each feature.

    Args:
        feature: The V language feature to explain (e.g., 'arrays', 'structs', 'functions', 'concurrency')

    Returns:
        Comprehensive explanation of the requested V language feature with examples
    """
    # Common V language features and their explanations
    features = {
        'variables': """
# V Variables

V supports several types of variables:

## Declaration and Initialization
```v
name := 'Bob'  // Inferred type string
age := 20       // Inferred type int
is_adult := true // Inferred type bool
```

## Explicit Type Declaration
```v
mut name string = 'Bob'  // Mutable variable
age int = 20            // Immutable variable
```

## Constants
```v
const pi = 3.14159
const (
    rate = 0.05
    days = 365
)
```
""",

        'arrays': """
# V Arrays

## Declaration and Initialization
```v
mut numbers := [1, 2, 3]     // Inferred type []int
names := ['Alice', 'Bob']    // []string
empty := []int{}             // Empty array
```

## Array Operations
```v
numbers << 4                 // Append element
numbers << [5, 6]            // Append array
first := numbers[0]          // Access element
numbers[1] = 10              // Modify element
len := numbers.len           // Get length
```

## Array Methods
```v
numbers.insert(0, 0)         // Insert at index
numbers.delete(1)            // Delete element
numbers.reverse()            // Reverse array
numbers.sort()               // Sort array
```
""",

        'structs': """
# V Structs

## Definition
```v
struct User {
    id   int
    name string
    age  int
mut:
    email string  // Mutable field
pub:
    active bool   // Public field
}
```

## Usage
```v
user := User{
    id: 1
    name: 'Alice'
    age: 30
    email: 'alice@example.com'
    active: true
}

// Access fields
println(user.name)    // Alice
user.email = 'new@example.com'  // Modify mutable field
```

## Methods
```v
fn (u User) full_name() string {
    return '${u.name} (ID: ${u.id})'
}

fn (mut u User) deactivate() {
    u.active = false
}
```
""",

        'functions': """
# V Functions

## Basic Function
```v
fn greet(name string) string {
    return 'Hello, ${name}!'
}

message := greet('World')
```

## Multiple Return Values
```v
fn divide(a int, b int) (int, int) {
    quotient := a / b
    remainder := a % b
    return quotient, remainder
}

q, r := divide(10, 3)
```

## Variadic Functions
```v
fn sum(numbers ...int) int {
    mut total := 0
    for num in numbers {
        total += num
    }
    return total
}

result := sum(1, 2, 3, 4, 5)
```

## Anonymous Functions
```v
fn apply_twice(f fn(int) int, x int) int {
    return f(f(x))
}

double := fn(x int) int { return x * 2 }
result := apply_twice(double, 3)  // 12
```
""",

        'control_flow': """
# V Control Flow

## If Statements
```v
age := 18
if age >= 18 {
    println('Adult')
} else if age >= 13 {
    println('Teenager')
} else {
    println('Child')
}
```

## If as Expression
```v
max := if a > b { a } else { b }
status := if user.active { 'Active' } else { 'Inactive' }
```

## Match Statement
```v
color := 'red'
match color {
    'red' { println('Stop!') }
    'yellow' { println('Caution!') }
    'green' { println('Go!') }
    else { println('Unknown color') }
}
```

## For Loops
```v
// Basic loop
for i in 0..10 {
    println(i)
}

// Loop over array
fruits := ['apple', 'banana', 'cherry']
for fruit in fruits {
    println(fruit)
}

// Loop with index
for i, fruit in fruits {
    println('${i}: ${fruit}')
}

// Infinite loop
for {
    println('Forever...')
    break
}
```
""",

        'modules': """
# V Modules

## Module Declaration
Each V file belongs to a module. The module name is the same as the folder name.

```
project/
├── main.v
└── utils/
    ├── math.v
    └── string.v
```

## Importing Modules
```v
import os
import utils.math
import utils.string as str_utils
```

## Selective Imports
```v
import os { read_file, write_file }
import utils.math { add, multiply as mul }
```

## Module Initialization
```v
// In utils/math.v
module math

pub fn add(a int, b int) int {
    return a + b
}

// In main.v
import utils.math

fn main() {
    result := math.add(5, 3)
    println(result)  // 8
}
```
""",

        'error_handling': """
# V Error Handling

## Option Types
```v
fn find_user(id int) ?User {
    if id < 0 {
        return none
    }
    return User{ id: id, name: 'User ${id}' }
}

// Usage
user := find_user(123) or {
    println('User not found')
    return
}
println(user.name)
```

## If Unwrapping
```v
if user := find_user(456) {
    println(user.name)
} else {
    println('User not found')
}
```

## Error Types
```v
fn risky_operation() !string {
    if rand.intn(2) == 0 {
        return error('Something went wrong')
    }
    return 'Success!'
}

// Usage
result := risky_operation() or {
    println('Error: ${err}')
    exit(1)
}
println(result)
```
""",

        'concurrency': """
# V Concurrency

## Goroutines
```v
fn worker(id int) {
    println('Worker ${id} starting')
    time.sleep(1 * time.second)
    println('Worker ${id} done')
}

fn main() {
    for i in 1..5 {
        go worker(i)
    }
    time.sleep(2 * time.second)  // Wait for goroutines
}
```

## Channels
```v
fn producer(ch chan int) {
    for i in 1..10 {
        ch <- i
        time.sleep(100 * time.millisecond)
    }
    ch.close()
}

fn consumer(ch chan int) {
    for {
        if val := <-ch {
            println('Received: ${val}')
        } else {
            break  // Channel closed
        }
    }
}

fn main() {
    ch := chan int{}
    go producer(ch)
    consumer(ch)
}
```
"""
    }

    try:
        if not feature or len(feature.strip()) < 1:
            available_features = list(features.keys())
            return f"Feature name cannot be empty. Available features: {', '.join(available_features)}"

        feature_key = feature.lower().strip()

        if feature_key in features:
            return features[feature_key]
        else:
            available_features = sorted(features.keys())
            return f"Feature '{feature}' not found. Available features: {', '.join(available_features)}\n\nUse `search_v_docs('{feature}')` to search documentation for this topic."
    except Exception as e:
        logger.error(f"Error explaining V syntax for feature '{feature}': {e}")
        available_features = sorted(features.keys())
        return f"An error occurred while explaining '{feature}'. Available features: {', '.join(available_features)}"

@mcp.tool
def get_v_quick_reference() -> str:
    """
    Get a quick reference guide for V programming language basics.

    Provides a concise reference covering the most essential V programming language
    syntax and concepts. Perfect for getting started or as a quick reminder.

    Returns:
        Quick reference guide with essential V language syntax and examples
    """
    return """
# V Programming Language Quick Reference

## Basic Syntax
```v
// Hello World
fn main() {
    println('Hello, World!')
}

// Variables
name := 'Alice'           // Inferred string
age := 30                 // Inferred int
is_active := true         // Inferred bool
mut counter := 0          // Mutable variable

// Constants
const pi = 3.14159
const days_in_week = 7
```

## Data Types
```v
// Primitives
bool_var := true          // bool
int_var := 42             // int
float_var := 3.14         // f64
string_var := 'hello'     // string
rune_var := `A`           // rune (Unicode code point)

// Arrays
numbers := [1, 2, 3]      // []int
names := ['Alice', 'Bob'] // []string
empty := []int{}          // Empty array

// Maps
ages := {'Alice': 30, 'Bob': 25}  // map[string]int
```

## Control Flow
```v
// If statement
if age >= 18 {
    println('Adult')
} else {
    println('Minor')
}

// For loops
for i in 0..5 {
    println(i)  // 0, 1, 2, 3, 4
}

fruits := ['apple', 'banana']
for fruit in fruits {
    println(fruit)
}

// Match
color := 'red'
match color {
    'red' => println('Stop')
    'green' => println('Go')
    else => println('Unknown')
}
```

## Functions
```v
fn greet(name string) string {
    return 'Hello, ${name}!'
}

fn add(a int, b int) int {
    return a + b
}

// Multiple return values
fn divide(a int, b int) (int, int) {
    return a / b, a % b
}

quotient, remainder := divide(10, 3)
```

## Structs
```v
struct User {
    id   int
    name string
mut:
    email string
}

user := User{
    id: 1
    name: 'Alice'
    email: 'alice@example.com'
}
```

## Modules
```v
// In utils.v
module utils

pub fn helper() string {
    return 'Helper function'
}

// In main.v
import utils

fn main() {
    result := utils.helper()
    println(result)
}
```

## Error Handling
```v
fn risky() ?string {
    if rand.intn(2) == 0 {
        return none
    }
    return 'Success!'
}

if result := risky() {
    println(result)
} else {
    println('Failed!')
}
```

## Arrays and Strings
```v
// Array operations
mut numbers := [1, 2, 3]
numbers << 4              // Append
numbers.insert(0, 0)      // Insert at index
numbers.delete(1)         // Delete at index

// String operations
text := 'Hello, World!'
println(text.len)         // Length
println(text.contains('World'))  // true
parts := text.split(', ')  // Split into array
```

## File I/O
```v
import os

// Read file
content := os.read_file('file.txt') or {
    println('Error reading file')
    return
}

// Write file
text := 'Hello, File!'
os.write_file('output.txt', text) or {
    println('Error writing file')
    return
}
```

## Concurrency
```v
import time

fn worker(id int) {
    println('Worker ${id}')
    time.sleep(1 * time.second)
}

// Start goroutine
go worker(1)
go worker(2)

// Channels
ch := chan int{}
go fn() {
    ch <- 42
}()
value := <-ch
println(value)  // 42
```

For more detailed information, use the other tools available:
- `get_v_documentation()` - Full language documentation
- `search_v_docs(query)` - Search documentation
- `list_v_examples()` - Browse code examples
- `explain_v_syntax(feature)` - Detailed syntax explanations

Available features for explain_v_syntax:
- variables, arrays, structs, functions, control_flow, modules, error_handling, concurrency
"""

@mcp.tool
def get_v_config() -> str:
    """
    Get current V MCP server configuration.

    Shows the current configuration settings including V repository path,
    cache settings, and other server parameters.

    Returns:
        Current server configuration information
    """
    output = f"# V MCP Server Configuration\n\n"
    output += f"**V Repository Path:** `{config.v_repo_path}`\n"
    output += f"**Cache TTL:** {config.cache_ttl_seconds} seconds ({config.cache_ttl_seconds // 60} minutes)\n"
    output += f"**Max Search Results:** {config.max_search_results}\n"
    output += f"**Log Level:** {config.log_level}\n\n"

    # Check if paths exist
    output += f"**Path Status:**\n"
    paths = {
        "Documentation": config.v_repo_path / "doc",
        "Examples": config.v_repo_path / "examples",
        "Standard Library": config.v_repo_path / "vlib"
    }

    for name, path in paths.items():
        status = "✅ Found" if path.exists() else "❌ Missing"
        output += f"- **{name}:** {status} (`{path}`)\n"

    # Cache statistics
    cache_stats = v_server.get_cache_stats()
    output += f"\n**Cache Statistics:**\n"
    output += f"- Current cache entries: {cache_stats['entries']}\n"
    output += f"- Cache TTL: {cache_stats['ttl_seconds']}s ({cache_stats['ttl_seconds'] // 60} minutes)\n"

    return output

@mcp.tool
def clear_v_cache() -> str:
    """
    Clear the V MCP server cache.

    This clears all cached documentation, examples, and search results.
    Useful when the V repository has been updated and you want fresh content.

    Returns:
        Cache clearing statistics and confirmation message
    """
    result = v_server.clear_cache()

    output = f"# Cache Cleared\n\n"
    output += f"✅ {result['message']}\n\n"
    output += f"**Statistics:**\n"
    output += f"- Cache entries cleared: {result['cleared_entries']}\n"
    output += f"- Timestamp entries cleared: {result['cleared_timestamps']}\n\n"
    output += f"Next requests will reload content from the V repository."

    return output

@mcp.tool
def list_v_ui_examples() -> str:
    """
    Get a list of available V UI examples.

    Returns a comprehensive list of all available V UI code examples from the v-ui repository.

    Returns:
        JSON string containing a list of V UI examples with their names and paths
    """
    try:
        if not v_server._path_status.get("v_ui_examples", False):
            return json.dumps({
                "error": "V UI examples are not available",
                "message": "V UI repository not found or examples directory missing. Make sure the v-ui submodule is initialized."
            }, indent=2)

        examples = v_server.get_v_ui_examples_list()
        
        if examples and "error" in examples[0]:
            return json.dumps(examples[0], indent=2)

        return json.dumps({
            "count": len(examples),
            "examples": examples[:50]  # Limit to first 50 for readability
        }, indent=2)

    except Exception as e:
        logger.error(f"Error listing V UI examples: {e}")
        return json.dumps({"error": f"Error listing V UI examples: {str(e)}"}, indent=2)

@mcp.tool
def get_v_ui_example(example_name: str) -> str:
    """
    Get the source code of a specific V UI example.

    Retrieves the complete source code for a named V UI programming example.

    Args:
        example_name: Name of the V UI example to retrieve (without .v extension)

    Returns:
        JSON string containing the example source code and metadata
    """
    try:
        if not v_server._path_status.get("v_ui_examples", False):
            return json.dumps({
                "error": "V UI examples are not available",
                "message": "V UI repository not found or examples directory missing."
            }, indent=2)

        result = v_server.get_v_ui_example_content(example_name)
        return json.dumps(result, indent=2)

    except Exception as e:
        logger.error(f"Error getting V UI example: {e}")
        return json.dumps({"error": f"Error getting V UI example: {str(e)}"}, indent=2)

@mcp.tool
def search_v_ui_examples(query: str) -> str:
    """
    Search through V UI example code for specific patterns or features.

    Performs full-text search across all V UI programming examples in the repository.

    Args:
        query: Search query (supports regex patterns)

    Returns:
        JSON string containing search results with matching code snippets
    """
    try:
        if not v_server._path_status.get("v_ui_examples", False):
            return json.dumps({
                "error": "V UI examples are not available",
                "message": "V UI repository not found or examples directory missing."
            }, indent=2)

        results = v_server.search_v_ui_examples(query)
        return json.dumps({
            "query": query,
            "count": len(results),
            "results": results
        }, indent=2)

    except Exception as e:
        logger.error(f"Error searching V UI examples: {e}")
        return json.dumps({"error": f"Error searching V UI examples: {str(e)}"}, indent=2)

@mcp.tool
def get_v_help() -> str:
    """
    Get help and information about available V MCP server tools.

    Provides an overview of all available tools and how to use them effectively.

    Returns:
        Help information about available tools and usage examples
    """
    return """
# V MCP Server Help

The V MCP Server provides comprehensive access to V programming language resources.

## Available Tools

### 📚 Documentation Tools
- **`get_v_documentation([section])`** - Get V documentation, optionally for a specific section
  - Example: `get_v_documentation()` or `get_v_documentation("Structs")`

- **`search_v_docs(query)`** - Search through V documentation
  - Example: `search_v_docs("arrays")`

### 💡 Code Examples
- **`list_v_examples()`** - List all available V code examples
- **`get_v_example(name)`** - Get complete source code for a specific example
  - Example: `get_v_example("fibonacci")`
- **`search_v_examples(query)`** - Search through example code
  - Example: `search_v_examples("http")`

### 🔧 Standard Library
- **`list_v_stdlib_modules()`** - List all V standard library modules
- **`get_v_module_info(module_name)`** - Get detailed info about a specific module
  - Example: `get_v_module_info("os")`

### 🎨 V UI Examples
- **`list_v_ui_examples()`** - List all available V UI code examples
- **`get_v_ui_example(name)`** - Get complete source code for a specific V UI example
  - Example: `get_v_ui_example("users")`
- **`search_v_ui_examples(query)`** - Search through V UI example code
  - Example: `search_v_ui_examples("button")`

### 🎯 Language Reference
- **`explain_v_syntax(feature)`** - Explain V language features
  - Available features: variables, arrays, structs, functions, control_flow, modules, error_handling, concurrency
  - Example: `explain_v_syntax("structs")`
- **`get_v_quick_reference()`** - Get quick V syntax reference

### ⚙️ Configuration & Cache
- **`get_v_config()`** - Show current server configuration and cache statistics
- **`clear_v_cache()`** - Clear cached content for fresh results

### 🆘 Help & Discovery
- **`get_v_help()`** - Show this help information

## Usage Tips

1. **Start with the basics**: Use `get_v_quick_reference()` to learn essential syntax
2. **Explore examples**: Use `list_v_examples()` to see practical code
3. **Search documentation**: Use `search_v_docs()` for specific topics
4. **Learn features**: Use `explain_v_syntax()` for detailed explanations
5. **Browse stdlib**: Use `list_v_stdlib_modules()` to explore available modules
6. **Check configuration**: Use `get_v_config()` to verify server settings
7. **Clear cache when needed**: Use `clear_v_cache()` after V repository updates

## Common Issues

- **Empty search results**: Try more specific search terms (minimum 2 characters)
- **Example not found**: Use `list_v_examples()` to see available examples
- **Unknown feature**: Use `explain_v_syntax("")` to see available features
"""

if __name__ == "__main__":
    # Run the MCP server
    try:
        logger.info("Starting V MCP Server...")
        mcp.run()
    except Exception as e:
        logger.error(f"Failed to start MCP server: {e}")
        raise
