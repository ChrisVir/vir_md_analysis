"""
Unit and regression test for the vir_md_analysis package.
"""

# Import package, test suite, and other packages as needed
import sys

import pytest

import vir_md_analysis


def test_vir_md_analysis_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "vir_md_analysis" in sys.modules
