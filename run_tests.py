#!/usr/bin/env python3
"""Test runner for comprehensive classification server verification.

This module provides a unified test runner for validating the functionality
of the classification server and related components. It executes a suite of
tests covering system overview, multi-team classification, workflow validation,
error handling, and integration testing.

The test runner provides clear visual feedback and summary statistics to
quickly identify any issues in the classification system.
"""

import asyncio
import subprocess
import sys
from pathlib import Path


def run_test(test_file: str, description: str) -> bool:
    """Run a single test file and return success status.
    
    Executes a test script as a subprocess with timeout protection and
    captures output for error reporting. Provides visual feedback using
    emoji indicators for test status.
    
    Args:
        test_file: Path to the test script to execute.
        description: Human-readable description of the test.
    
    Returns:
        bool: True if test passed, False otherwise.
    """
    print(f"🧪 Running {description}...")
    try:
        result = subprocess.run([sys.executable, test_file], 
                              capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            print(f"✅ {description}: PASSED")
            return True
        else:
            print(f"❌ {description}: FAILED")
            if result.stderr:
                print(f"   Error: {result.stderr.strip()}")
            return False
    except subprocess.TimeoutExpired:
        print(f"⏰ {description}: TIMEOUT")
        return False
    except Exception as e:
        print(f"💥 {description}: ERROR - {e}")
        return False


def main():
    """Run all classification server tests in priority order.
    
    Executes the complete test suite, providing progress feedback and
    a final summary report. Tests are run in order of importance to
    quickly identify critical issues.
    
    The test suite covers:
    - System overview and capability verification
    - Multi-team classification functionality
    - Classification workflow correctness
    - Error handling and edge cases
    - Integration with knowledge server
    """
    print("🚀 Classification Server Test Suite")
    print("=" * 50)
    
    # Define tests in order of importance
    tests = [
        ("test_system_overview.py", "System Overview & Capabilities"),
        ("test_multi_team_classification.py", "Multi-Team Classification"), 
        ("test_classification_workflow.py", "Classification Workflow"),
        ("test_error_handling.py", "Error Handling & Edge Cases"),
        ("test_knowledge.py", "Knowledge Server Integration")
    ]
    
    results = []
    for test_file, description in tests:
        if Path(test_file).exists():
            success = run_test(test_file, description)
            results.append((description, success))
        else:
            print(f"⚠️  {description}: SKIPPED (file not found)")
            results.append((description, None))
    
    print("\n📊 Test Results Summary:")
    print("-" * 30)
    
    passed = sum(1 for _, result in results if result is True)
    failed = sum(1 for _, result in results if result is False)
    skipped = sum(1 for _, result in results if result is None)
    
    for description, result in results:
        if result is True:
            print(f"✅ {description}")
        elif result is False:
            print(f"❌ {description}")
        else:
            print(f"⚠️  {description} (skipped)")
    
    print(f"\nTotal: {passed} passed, {failed} failed, {skipped} skipped")
    
    if failed == 0 and passed > 0:
        print("\n🎉 All tests PASSED! Classification server is ready!")
        return True
    else:
        print("\n💥 Some tests FAILED! Please check the issues above.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)