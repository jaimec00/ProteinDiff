"""
Shared pytest fixtures for tests.
"""
import pytest
import torch


@pytest.fixture
def device():
    """Return cuda if available, else cpu."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
