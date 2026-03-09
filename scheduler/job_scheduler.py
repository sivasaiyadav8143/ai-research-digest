"""
================================================================================
PHASE 6 — JOB SCHEDULER
================================================================================
Module  : scheduler/job_scheduler.py
Purpose : Manages scheduled daily email jobs using APScheduler.
          When a user selects "Daily Schedule" in the Gradio UI, their email
          is registered here as a recurring job that runs once every 24 hours.

How APScheduler works in this project:
    - We use BackgroundScheduler — it runs jobs in a background thread
      so the Gradio UI stays responsive while jobs execute
    - Jobs are stored IN MEMORY (not persisted to disk)
    - This means: if the HuggingFace Space restarts, scheduled jobs are lost
    - Users would need to re-subscribe after a Space restart

    NOTE ON HF SPACES FREE TIER:
    HuggingFace free Spaces go to sleep after ~15 minutes of inactivity.
    This means the scheduler will stop running when the Space sleeps.
    For a production newsletter, you'd want either:
        Option A: HF Spaces upgraded tier (always-on)
        Option B: External cron job (GitHub Actions, cron-job.org)
        Option C: Store emails in a DB and trigger from external scheduler
    For a portfolio project on free tier, this is perfectly fine and demonstrates
    the scheduling architecture correctly.

APScheduler docs: https://apscheduler.readthedocs.io/en/3.x/

Author  : AI Research Digest Project
================================================================================
"""

import logging
from datetime import datetime, timezone
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.triggers.cron import CronTrigger


# Configure logging so APScheduler's internal messages are visible
# This helps debug scheduling issues during development
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DigestScheduler:
    """
    Manages scheduled daily digest jobs for subscribed email addresses.

    Each subscriber gets their own APScheduler job identified by their
    email address. This allows individual subscribe/unsubscribe control.

    Usage:
        # In app.py — create ONE shared instance for the whole app
        scheduler = DigestScheduler(pipeline_fn=run_pipeline)
        scheduler.start()

        # When user subscribes via Gradio UI
        scheduler.add_job(email="user@example.com", hour=8, minute=0)

        # When user unsubscribes
        scheduler.remove_job(email="user@example.com")

    Args:
        pipeline_fn : The function to call for each scheduled run.
                      Should accept a single argument: recipient_email (str)
                      This will be the main pipeline function from app.py
    """

    def __init__(self, pipeline_fn: callable):
        """
        Initialises the scheduler with the pipeline function to call.

        Args:
            pipeline_fn: Callable that takes recipient_email (str) as argument.
                         This is the full fetch → filter → summarise → send pipeline.
        """
        self.pipeline_fn = pipeline_fn

        # BackgroundScheduler runs jobs in a separate thread
        # timezone="UTC" ensures consistent scheduling regardless of server location
        self.scheduler = BackgroundScheduler(timezone="UTC")

        # Track active jobs: maps email → job_id
        # Used to check if someone is already subscribed and to remove jobs
        self.active_jobs: dict[str, str] = {}

        logger.info("[Scheduler] DigestScheduler initialised")

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self):
        """
        Starts the background scheduler.
        Must be called once when the app starts (in app.py).
        The scheduler runs in a daemon thread — it stops automatically
        when the main Python process exits.
        """
        if not self.scheduler.running:
            self.scheduler.start()
            logger.info("[Scheduler] ✅ Background scheduler started")

    def shutdown(self):
        """
        Gracefully shuts down the scheduler.
        Called when the app is closing to cleanly stop background threads.
        APScheduler waits for currently running jobs to finish before stopping.
        """
        if self.scheduler.running:
            self.scheduler.shutdown(wait=True)
            logger.info("[Scheduler] Scheduler shut down cleanly")

    # ── Job Management ────────────────────────────────────────────────────────

    def add_job(self, email: str, hour: int = 8, minute: int = 0) -> dict:
        """
        Registers a new daily digest job for the given email address.

        The job will run every day at the specified UTC hour:minute.
        Default is 08:00 UTC which is a reasonable morning time for
        most European/US timezones.

        If the email already has a job, it is replaced with the new schedule
        (idempotent — safe to call multiple times for the same email).

        Args:
            email  : Recipient email address. Used as the unique job identifier.
            hour   : UTC hour to send the digest (0-23). Default: 8
            minute : UTC minute to send the digest (0-59). Default: 0

        Returns:
            dict: {
                "success"    : bool,
                "message"    : str,
                "next_run"   : str  — ISO format datetime of next scheduled run,
                "job_id"     : str  — unique job identifier
            }
        """
        # Create a safe job ID from the email (APScheduler needs alphanumeric IDs)
        # Replace @ and . with underscores: user@example.com → user_example_com
        job_id = f"digest_{email.replace('@', '_').replace('.', '_')}"

        # If already subscribed, remove the old job first
        if email in self.active_jobs:
            self._remove_job_by_id(self.active_jobs[email])
            logger.info(f"[Scheduler] Replaced existing job for {email}")

        try:
            # Schedule the job using CronTrigger
            # CronTrigger fires at a specific time each day (like a Unix cron)
            # interval=1 day at hour:minute UTC
            job = self.scheduler.add_job(
                func     = self._run_pipeline_for_email,  # Function to call
                trigger  = CronTrigger(
                    hour    = hour,
                    minute  = minute,
                    timezone= "UTC",
                ),
                args     = [email],       # Arguments passed to the function
                id       = job_id,        # Unique identifier for this job
                name     = f"Daily digest for {email}",
                replace_existing = True,  # Replace if ID already exists
                misfire_grace_time = 3600,  # Allow up to 1hr late if server was sleeping
            )

            # Track this job
            self.active_jobs[email] = job_id

            # Get next run time for confirmation message
            next_run = job.next_run_time.strftime("%Y-%m-%d %H:%M UTC") if job.next_run_time else "Unknown"

            logger.info(f"[Scheduler] ✅ Job scheduled for {email} — next run: {next_run}")

            return {
                "success" : True,
                "message" : f"Daily digest scheduled for {email} at {hour:02d}:{minute:02d} UTC",
                "next_run": next_run,
                "job_id"  : job_id,
            }

        except Exception as e:
            logger.error(f"[Scheduler] ❌ Failed to add job for {email}: {e}")
            return {
                "success" : False,
                "message" : f"Failed to schedule job: {str(e)}",
                "next_run": None,
                "job_id"  : None,
            }

    def remove_job(self, email: str) -> dict:
        """
        Cancels the scheduled daily digest for an email address.

        Args:
            email: The email whose job should be cancelled.

        Returns:
            dict: { "success": bool, "message": str }
        """
        if email not in self.active_jobs:
            return {
                "success": False,
                "message": f"No scheduled job found for {email}",
            }

        job_id = self.active_jobs[email]
        success = self._remove_job_by_id(job_id)

        if success:
            del self.active_jobs[email]
            return {
                "success": True,
                "message": f"Cancelled daily digest for {email}",
            }
        else:
            return {
                "success": False,
                "message": f"Failed to cancel job for {email}",
            }

    def list_jobs(self) -> list[dict]:
        """
        Returns a list of all currently scheduled jobs.
        Useful for debugging and monitoring from the Gradio UI.

        Returns:
            list[dict]: Each dict has email, job_id, next_run_time
        """
        jobs_info = []
        for email, job_id in self.active_jobs.items():
            job = self.scheduler.get_job(job_id)
            if job:
                jobs_info.append({
                    "email"        : email,
                    "job_id"       : job_id,
                    "next_run_time": job.next_run_time.strftime("%Y-%m-%d %H:%M UTC")
                                     if job.next_run_time else "Unknown",
                })
        return jobs_info

    def is_subscribed(self, email: str) -> bool:
        """
        Checks whether an email address has an active scheduled job.

        Args:
            email: Email address to check.

        Returns:
            bool: True if a daily job exists for this email, False otherwise.
        """
        return email in self.active_jobs

    # ── Internal Helpers ──────────────────────────────────────────────────────

    def _run_pipeline_for_email(self, email: str):
        """
        Called by APScheduler when a job fires.
        Wraps the pipeline function with error handling so a single failure
        doesn't crash the entire scheduler or affect other subscribers.

        Args:
            email: The recipient email for this scheduled run.
        """
        logger.info(f"[Scheduler] 🚀 Running scheduled digest for {email}")
        try:
            self.pipeline_fn(email)
            logger.info(f"[Scheduler] ✅ Completed digest for {email}")
        except Exception as e:
            # Log the error but don't re-raise — scheduler must keep running
            logger.error(f"[Scheduler] ❌ Pipeline failed for {email}: {e}")

    def _remove_job_by_id(self, job_id: str) -> bool:
        """
        Removes a job from APScheduler by its ID.

        Args:
            job_id: The APScheduler job ID to remove.

        Returns:
            bool: True if removed successfully, False if job not found.
        """
        try:
            self.scheduler.remove_job(job_id)
            return True
        except Exception as e:
            logger.warning(f"[Scheduler] Could not remove job {job_id}: {e}")
            return False
