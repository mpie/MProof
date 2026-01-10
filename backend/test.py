#!/usr/bin/env python3
"""
Run tests for the document analysis system.
"""

import subprocess
import sys
import os

def run_tests():
    """Run all tests."""
    print("🧪 Running backend tests...")

    # Change to backend directory
    os.chdir(os.path.dirname(__file__))

    # Run pytest
    result = subprocess.run([
        sys.executable, "-m", "pytest",
        "tests/",
        "-v",
        "--tb=short",
        "--disable-warnings"
    ], capture_output=True, text=True)

    print(result.stdout)
    if result.stderr:
        print("Errors:", result.stderr)

    return result.returncode == 0

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)