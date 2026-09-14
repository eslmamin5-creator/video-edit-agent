"""Assembler agent: multiple existing scene files -> rough cut / finished film.

Works fully offline. Reuses shared core (EDL, MasterTimeline, render,
Brand Profiles, motion router, B-roll, subject compositing, QA primitives)
rather than duplicating them -- see `agents/assembler/pipeline.py`.
"""
