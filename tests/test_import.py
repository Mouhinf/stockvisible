"""Trivial smoke test: the package must be importable."""

import stockvisible


def test_package_imports():
    assert stockvisible.__version__ == "0.1.0"
