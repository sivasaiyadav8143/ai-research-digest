"""
================================================================================
PHASE 4 — NEWSLETTER AGENT
================================================================================
Module  : agents/newsletter_agent.py
Purpose : Takes the summarised papers from Phase 3, renders them into a
          polished HTML email using our Jinja2 template, and delivers the
          newsletter to the recipient via SendGrid API.

Why SendGrid?
    - Free tier: 100 emails/day — plenty for a portfolio project
    - Works on HuggingFace Spaces (uses HTTPS/port 443, never blocked)
    - No domain verification needed to send to any email address
    - Simple HTTP API — no extra complexity
    - Docs: https://docs.sendgrid.com/api-reference/mail-send

Why not Gmail SMTP?
    - HuggingFace Spaces blocks outbound SMTP (port 587) for security
    - SendGrid uses HTTPS which is always allowed

Why Jinja2 for templating?
    - Industry standard Python templating engine (used in Flask, Django)
    - Keeps HTML completely separate from Python logic
    - Supports loops, conditionals, and filters
    - Easy to update the email design without touching Python code

Pipeline position:
    Phase 1 (Fetch) → Phase 2 (Filter) → Phase 3 (Summarise) → Phase 4 (HERE)

Author  : AI Research Digest Project
================================================================================
"""

import os
import requests
from datetime import datetime, timezone
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape
from dotenv import load_dotenv

# Import our data models from previous phases
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.summariser_agent import SummarisedPaper

# Load .env for local development
# On HuggingFace Spaces, secrets are injected automatically
load_dotenv()


# ── Newsletter Agent ──────────────────────────────────────────────────────────

class NewsletterAgent:
    """
    Renders summarised papers into a beautiful HTML email and delivers
    it to the recipient using the SendGrid email API over HTTPS.

    Usage:
        agent = NewsletterAgent(sender_email="you@example.com")
        result = agent.run(summarised_papers, recipient_email="user@example.com")

    Environment Variables Required:
        SENDGRID_API_KEY : Your SendGrid API key
                           Get free at: https://sendgrid.com
                           → Settings → API Keys → Create API Key → Full Access
        SENDER_EMAIL     : The "From" address (must be verified in SendGrid)
                           → Settings → Sender Authentication → Single Sender Verification

    SendGrid Free Tier:
        - 100 emails/day forever
        - No credit card required
        - No domain needed (single sender verification is enough)
        - Sends to ANY email address
    """

    # SendGrid Mail Send API endpoint — uses HTTPS (port 443), always works on HF
    SENDGRID_API_URL = "https://api.sendgrid.com/v3/mail/send"

    # Path to our Jinja2 HTML template
    TEMPLATE_DIR  = Path(__file__).parent.parent / "templates"
    TEMPLATE_FILE = "email_template.html"

    def __init__(self, sender_email: str = None,
                 sender_name: str = "AI Research Digest"):
        """
        Initialises the newsletter agent and validates SendGrid credentials.

        Args:
            sender_email : Verified sender email address.
                           Falls back to SENDER_EMAIL env var if not provided.
            sender_name  : Display name shown in recipient's inbox.

        Raises:
            ValueError: If SENDGRID_API_KEY or SENDER_EMAIL are not set.
        """
        # Load credentials from environment variables
        self.api_key      = os.getenv("SENDGRID_API_KEY")
        self.sender_email = sender_email or os.getenv("SENDER_EMAIL")
        self.sender_name  = sender_name

        if not self.api_key:
            raise ValueError(
                "SENDGRID_API_KEY not set.\n"
                "  → Sign up free at: https://sendgrid.com\n"
                "  → Settings → API Keys → Create API Key → Full Access\n"
                "  → Add to HF Secrets: SENDGRID_API_KEY = SG.your_key_here"
            )

        if not self.sender_email:
            raise ValueError(
                "SENDER_EMAIL not set.\n"
                "  → Add to HF Secrets: SENDER_EMAIL = you@gmail.com\n"
                "  → Must be verified in SendGrid:\n"
                "    Settings → Sender Authentication → Single Sender Verification"
            )

        # Build auth header — used in every API request
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type" : "application/json",
        }

        # Set up Jinja2 templating environment
        self.jinja_env = Environment(
            loader     = FileSystemLoader(str(self.TEMPLATE_DIR)),
            autoescape = select_autoescape(["html"]),
        )

        print(f"[Newsletter Agent] Initialised — sender: {self.sender_email}")

    # ── Main Entry Point ──────────────────────────────────────────────────────

    def run(self, summarised_papers: list[SummarisedPaper],
            recipient_email: str) -> dict:
        """
        Full newsletter pipeline: render template → send email.

        Args:
            summarised_papers : Output from SummariserAgent — list of
                                SummarisedPaper objects ready to display.
            recipient_email   : Email address to deliver the newsletter to.

        Returns:
            dict: {
                "success"     (bool) : Whether the email was sent successfully
                "email_id"    (str)  : SendGrid message ID for tracking
                "message"     (str)  : Human-readable status message
                "html_preview"(str)  : Rendered HTML (for Gradio preview tab)
            }
        """
        print(f"\n[Newsletter Agent] Starting newsletter pipeline...")
        print(f"[Newsletter Agent] Recipient : {recipient_email}")
        print(f"[Newsletter Agent] Papers    : {len(summarised_papers)}")

        # ── Step 1: Render HTML template ──────────────────────────────────────
        print(f"[Newsletter Agent] Step 1 — Rendering HTML template...")
        html_content = self._render_template(summarised_papers)
        print(f"[Newsletter Agent]   ✅ HTML rendered ({len(html_content):,} chars)")

        # ── Step 2: Send via SendGrid API ─────────────────────────────────────
        print(f"[Newsletter Agent] Step 2 — Sending via SendGrid...")
        result = self._send_email(
            to_email     = recipient_email,
            html_content = html_content,
            paper_count  = len(summarised_papers),
        )

        if result["success"]:
            print(f"[Newsletter Agent] ✅ Email delivered! ID: {result['email_id']}")
        else:
            print(f"[Newsletter Agent] ❌ Failed: {result['message']}")

        # Include rendered HTML so Gradio UI can show a preview
        result["html_preview"] = html_content
        return result

    # ── Step 1: Template Rendering ────────────────────────────────────────────

    def _render_template(self, summarised_papers: list[SummarisedPaper]) -> str:
        """
        Renders the Jinja2 HTML email template with real paper data.

        Args:
            summarised_papers: List of SummarisedPaper objects from Phase 3

        Returns:
            str: Fully rendered HTML string ready to be sent as email body
        """
        now = datetime.now(timezone.utc)

        # Estimate reading time — ~200 words per paper at 238 words/minute
        estimated_words    = len(summarised_papers) * 200 + 100
        estimated_read_min = max(1, round(estimated_words / 238))

        context = {
            "papers"        : summarised_papers,
            "date"          : now.strftime("%A, %B %-d %Y"),
            "paper_count"   : len(summarised_papers),
            "read_time"     : estimated_read_min,
            "time_of_day"   : self._get_time_of_day(now.hour),
            "recipient_name": "Reader",
        }

        template     = self.jinja_env.get_template(self.TEMPLATE_FILE)
        html_content = template.render(**context)
        return html_content

    def _get_time_of_day(self, hour: int) -> str:
        """
        Returns a greeting word based on UTC hour.

        Args:
            hour: UTC hour (0-23)

        Returns:
            str: "morning", "afternoon", or "evening"
        """
        if 5 <= hour < 12:
            return "morning"
        elif 12 <= hour < 17:
            return "afternoon"
        else:
            return "evening"

    # ── Step 2: SendGrid API Delivery ─────────────────────────────────────────

    def _send_email(self, to_email: str, html_content: str,
                    paper_count: int) -> dict:
        """
        Sends the rendered HTML email via SendGrid REST API.

        SendGrid API call structure (v3 Mail Send):
            POST https://api.sendgrid.com/v3/mail/send
            Headers: Authorization: Bearer SG.xxx
            Body: {
                "personalizations": [{"to": [{"email": "..."}]}],
                "from": {"email": "...", "name": "..."},
                "subject": "...",
                "content": [{"type": "text/html", "value": "..."}]
            }

        SendGrid returns HTTP 202 Accepted on success (not 200).
        The Message-ID is in the response headers as X-Message-Id.

        Args:
            to_email     : Recipient email address
            html_content : Fully rendered HTML from _render_template()
            paper_count  : Number of papers (used in subject line)

        Returns:
            dict: { "success": bool, "email_id": str, "message": str }
        """
        # Build subject line with today's date
        today_str = datetime.now(timezone.utc).strftime("%b %-d")
        subject   = f"🧠 AI Research Digest — {today_str} ({paper_count} papers)"

        # Plain text fallback for email clients that can't render HTML
        plain_text = (
            f"AI Research Digest — {today_str}\n\n"
            f"Your digest contains {paper_count} AI research paper summaries.\n"
            f"Please view this email in an HTML-capable client for the full newsletter.\n\n"
            f"Powered by Mistral 7B and open-source tools."
        )

        # Build the SendGrid API request payload
        payload = {
            "personalizations": [
                {
                    "to": [{"email": to_email}],
                }
            ],
            "from": {
                "email": self.sender_email,
                "name" : self.sender_name,
            },
            "subject": subject,
            "content": [
                # Plain text first (fallback for basic clients)
                {"type": "text/plain", "value": plain_text},
                # HTML second (preferred by modern email clients)
                {"type": "text/html",  "value": html_content},
            ],
        }

        try:
            response = requests.post(
                self.SENDGRID_API_URL,
                headers = self.headers,
                json    = payload,
                timeout = 30,
            )

            # SendGrid returns 202 Accepted on success (not 200 OK)
            if response.status_code == 202:
                # Extract message ID from response headers for tracking
                email_id = response.headers.get("X-Message-Id", "unknown")
                return {
                    "success" : True,
                    "email_id": email_id,
                    "message" : f"Email successfully sent to {to_email}",
                }

            # ── Handle specific SendGrid error codes ──────────────────────────

            elif response.status_code == 401:
                # Invalid API key
                return {
                    "success" : False,
                    "email_id": None,
                    "message" : (
                        "SendGrid authentication failed — invalid API key.\n"
                        "Check SENDGRID_API_KEY in your HF Space Secrets."
                    ),
                }

            elif response.status_code == 403:
                # Sender not verified
                return {
                    "success" : False,
                    "email_id": None,
                    "message" : (
                        "SendGrid sender not verified.\n"
                        "Go to: SendGrid → Settings → Sender Authentication "
                        "→ Single Sender Verification → verify your SENDER_EMAIL."
                    ),
                }

            else:
                # Other API errors — include response body for debugging
                error_body = response.json() if response.content else {}
                errors     = error_body.get("errors", [{}])
                error_msg  = errors[0].get("message", response.text[:200])
                return {
                    "success" : False,
                    "email_id": None,
                    "message" : f"SendGrid error {response.status_code}: {error_msg}",
                }

        except requests.Timeout:
            return {
                "success" : False,
                "email_id": None,
                "message" : "Request to SendGrid timed out. Please try again.",
            }

        except Exception as e:
            return {
                "success" : False,
                "email_id": None,
                "message" : f"Unexpected error sending email: {str(e)}",
            }

    # ── Utility: HTML Preview ─────────────────────────────────────────────────

    def save_preview(self, summarised_papers: list[SummarisedPaper],
                     output_path: str = "preview_email.html") -> str:
        """
        Saves the rendered HTML to a local file for visual preview.
        Open the file in your browser to check design without sending.

        Args:
            summarised_papers : Papers to render into the preview
            output_path       : Where to save the HTML file

        Returns:
            str: Absolute path to the saved preview file
        """
        html = self._render_template(summarised_papers)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)
        abs_path = os.path.abspath(output_path)
        print(f"[Newsletter Agent] 👁️  Preview saved: {abs_path}")
        return abs_path
