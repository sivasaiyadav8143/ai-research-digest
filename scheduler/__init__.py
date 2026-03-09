# scheduler/__init__.py
# Makes the scheduler/ directory a Python package.
# Exposes DigestScheduler for clean imports from app.py.
# Usage elsewhere: from scheduler import DigestScheduler

from scheduler.job_scheduler import DigestScheduler
