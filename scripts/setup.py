#!/usr/bin/env python3
"""Setup script for StackOne RAG system.

This script installs dependencies and creates indices for all configurations.
Run with: python scripts/setup.py
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

# Configuration names to load
CONFIGS = ["c0", "c1", "c2", "c3", "c4", "c5"]


def print_header(msg: str) -> None:
    """Print a formatted header."""
    print(f"\n{'=' * 60}")
    print(f"  {msg}")
    print(f"{'=' * 60}\n")


def print_step(msg: str) -> None:
    """Print a step message."""
    print(f"→ {msg}")


def print_error(msg: str) -> None:
    """Print an error message."""
    print(f"✗ ERROR: {msg}", file=sys.stderr)


def print_success(msg: str) -> None:
    """Print a success message."""
    print(f"✓ {msg}")


def check_uv() -> bool:
    """Check if uv is installed."""
    if shutil.which("uv") is None:
        print_error("uv is not installed.")
        print("  Install it from: https://docs.astral.sh/uv/getting-started/installation/")
        return False
    print_success("uv is installed")
    return True


def check_env_file() -> bool:
    """Check if .env file exists with OPENAI_API_KEY."""
    project_root = Path(__file__).parent.parent
    env_file = project_root / ".env"

    if not env_file.exists():
        print_error(".env file not found.")
        print("  Copy .env_example to .env and add your OPENAI_API_KEY:")
        print("    cp .env_example .env")
        return False

    # Check if OPENAI_API_KEY is set
    env_content = env_file.read_text()
    if "OPENAI_API_KEY=" not in env_content:
        print_error("OPENAI_API_KEY not found in .env file.")
        print("  Add your OpenAI API key to .env:")
        print("    OPENAI_API_KEY=sk-...")
        return False

    # Check if the key has a value (not empty)
    for line in env_content.splitlines():
        if line.startswith("OPENAI_API_KEY="):
            key_value = line.split("=", 1)[1].strip()
            if not key_value or key_value == "sk-..." or key_value == '""' or key_value == "''":
                print_error("OPENAI_API_KEY is not set to a valid value.")
                print("  Edit .env and add your actual OpenAI API key.")
                return False

    print_success(".env file configured with OPENAI_API_KEY")
    return True


def install_dependencies() -> bool:
    """Install Python dependencies using uv."""
    print_step("Installing dependencies with uv...")

    project_root = Path(__file__).parent.parent
    result = subprocess.run(
        ["uv", "sync", "--all-extras"],
        cwd=project_root,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print_error("Failed to install dependencies.")
        print(result.stderr)
        return False

    print_success("Dependencies installed")
    return True


def load_config(config: str) -> bool:
    """Load data for a specific configuration."""
    print_step(f"Loading data for configuration: {config}")

    project_root = Path(__file__).parent.parent
    result = subprocess.run(
        ["uv", "run", "python", "-m", "ai_exercise.loading.loader", config],
        cwd=project_root,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print_error(f"Failed to load config {config}")
        print(result.stderr)
        return False

    print_success(f"Config {config} loaded")
    return True


def main() -> int:
    """Main setup function."""
    print_header("StackOne RAG System Setup")

    # Change to project root
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)

    # Step 1: Check prerequisites
    print_header("Step 1: Checking Prerequisites")

    if not check_uv():
        return 1

    if not check_env_file():
        return 1

    # Step 2: Install dependencies
    print_header("Step 2: Installing Dependencies")

    if not install_dependencies():
        return 1

    # Step 3: Load data for all configurations
    print_header("Step 3: Loading Data for All Configurations")

    failed_configs = []
    for config in CONFIGS:
        if not load_config(config):
            failed_configs.append(config)

    if failed_configs:
        print_error(f"Failed to load configs: {', '.join(failed_configs)}")
        return 1

    # Success
    print_header("Setup Complete!")

    print("All configurations have been loaded. You can now:")
    print()
    print("  1. Run the demo app:")
    print("     make start-app")
    print()
    print("  2. Run evaluations:")
    print("     make eval CONFIG=c0")
    print("     make eval CONFIG=c5")
    print()
    print("  3. Compare configurations:")
    print("     make eval-compare CONFIGS=c0,c1,c2,c3,c4,c5")
    print()
    print("  4. View results in Streamlit:")
    print("     make compare-evals")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
