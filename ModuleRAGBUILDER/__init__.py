# ModuleRAGBUILDER/__init__.py
"""
ModuleRAGBUILDER package entry.

Exposes the main orchestrator (build_rag_dataset) after rag_builder.py is created.
Keep this file minimal so import paths are predictable.
"""
__version__ = "0.1.0"

# Note: rag_builder imports other modules in this package.
# We intentionally do NOT import rag_builder here to avoid heavy imports at package import time.
# Users should import the orchestrator explicitly:
#   from ModuleRAGBUILDER.rag_builder import build_rag_dataset
