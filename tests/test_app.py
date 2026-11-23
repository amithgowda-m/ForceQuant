"""Tests for the web application launcher (app.py)."""
import os
import pytest

from app import find_available_port, DASHBOARD_FILE, REPO_ROOT

def test_dashboard_file_exists():
    assert os.path.exists(DASHBOARD_FILE), f"Expected {DASHBOARD_FILE} to exist"
    assert os.path.getsize(DASHBOARD_FILE) > 50_000, "Dashboard HTML file is too small or incomplete"

def test_find_available_port():
    port = find_available_port(8000)
    assert 8000 <= port < 8025
