---
title: Ai Research Digest
emoji: 🏆
colorFrom: yellow
colorTo: blue
sdk: gradio
sdk_version: 6.9.0
app_file: app.py
pinned: false
license: mit
short_description: AI research papers summarisation using Mistral 7B.
---


# 🧠 AI Research Digest

> Stay ahead of AI research — without reading the papers.

An open-source pipeline that fetches the latest AI research papers from **arXiv** and **HuggingFace Papers**, summarises them in plain English using **Mistral 7B**, and delivers a beautiful HTML newsletter to your inbox — daily or on demand.

Built entirely with open-source tools and deployed on **HuggingFace Spaces**.

---

## ✨ What It Does

1. **Fetches** the latest AI papers from arXiv (cs.AI, cs.LG, cs.CL, cs.CV) and HuggingFace Papers
2. **Filters & ranks** papers by recency, keyword relevance, and community interest — selects top 5
3. **Summarises** each paper using Mistral 7B Instruct with a structured plain-English format:
   - 📰 Headline — one punchy sentence
   - 📖 What it does — 2-3 plain sentences, no jargon
   - 💡 Why it matters — real-world impact
   - 🔍 Analogy — "Think of it like..."
4. **Delivers** a styled HTML newsletter via Resend email API

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────┐
│              GRADIO UI (HuggingFace Space)           │
│   [Email Input]  [Send Now / Daily Schedule]         │
└──────────────────────┬──────────────────────────────┘
                       │
           ┌───────────▼───────────┐
           │    run_pipeline()     │  ← app.py orchestrates all phases
           └───────────┬───────────┘
                       │
         ┌─────────────┼─────────────┐
         ▼             ▼             
  ┌─────────────┐ ┌─────────────┐   
  │ArXiv Fetcher│ │  HF Fetcher │   Phase 1
  └──────┬──────┘ └──────┬──────┘   
         └───────┬────────┘         
          ┌──────▼──────┐           
          │Filter & Rank│           Phase 2
          └──────┬──────┘           
          ┌──────▼──────┐           
          │  Summariser │           Phase 3
          │ Mistral 7B  │           
          └──────┬──────┘           
          ┌──────▼──────┐           
          │  Newsletter │           Phase 4
          │    Agent    │           
          └─────────────┘           
```

---

## 📁 Project Structure

```
ai-research-digest/
│
├── app.py                      # Main entry point — Gradio UI + pipeline orchestrator
├── requirements.txt            # Python dependencies
├── .env.example                # Environment variables template (copy to .env)
│
├── agents/
│   ├── __init__.py
│   ├── fetcher_arxiv.py        # Phase 1 — arXiv API fetcher
│   ├── fetcher_hf.py           # Phase 1 — HuggingFace Papers scraper
│   ├── filter_agent.py         # Phase 2 — Dedup, filter, rank papers
│   ├── summariser_agent.py     # Phase 3 — Mistral 7B plain-English summaries
│   └── newsletter_agent.py     # Phase 4 — HTML render + Resend email delivery
│
├── scheduler/
│   ├── __init__.py
│   └── job_scheduler.py        # APScheduler — manages daily digest jobs
│
└── templates/
    └── email_template.html     # Jinja2 HTML email template
```

---

## 🚀 Quick Start (Local Development)

### 1. Clone the repo
```bash
git clone https://huggingface.co/spaces/yourusername/ai-research-digest
cd ai-research-digest
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set up environment variables
```bash
cp .env.example .env
# Edit .env and fill in your API keys
```

You need two API keys:
| Key | Where to get it | Free tier |
|-----|----------------|-----------|
| `HF_TOKEN` | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) | ✅ Free |
| `RESEND_API_KEY` | [resend.com](https://resend.com) | ✅ 3,000 emails/month |

### 4. Run locally
```bash
python app.py
# Opens at http://localhost:7860
```

---

## ☁️ Deploy to HuggingFace Spaces

### 1. Create a new Space
- Go to [huggingface.co/new-space](https://huggingface.co/new-space)
- SDK: **Gradio**
- Visibility: Public (for portfolio) or Private

### 2. Add your secrets
In your Space → **Settings → Repository Secrets**:
```
HF_TOKEN       = hf_your_token_here
RESEND_API_KEY = re_your_key_here
SENDER_EMAIL   = onboarding@resend.dev   (or your verified domain)
```

### 3. Push your code
```bash
git remote add space https://huggingface.co/spaces/yourusername/ai-research-digest
git push space main
```

HuggingFace Spaces will automatically install `requirements.txt` and run `app.py`.

---

## 🔑 API Keys Setup

### HuggingFace Token (for Mistral 7B)
1. Go to [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. Click **New token** → Role: **Read** → Create
3. Copy the token (starts with `hf_`)

### Resend API Key (for email delivery)
1. Sign up at [resend.com](https://resend.com) — free account
2. Go to **API Keys** → **Create API Key**
3. Copy the key (starts with `re_`)
4. For production: verify your domain at **Domains** → **Add Domain**
5. For testing: use `onboarding@resend.dev` as sender (limited to your verified email)

---

## 🛠️ Tech Stack

| Component | Technology | Why |
|-----------|-----------|-----|
| UI | [Gradio](https://gradio.app) | Native HF Spaces support, zero config |
| LLM | [Mistral 7B Instruct](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3) | Best open-source instruction model |
| LLM Hosting | [HF Inference API](https://huggingface.co/docs/api-inference) | Free tier, no GPU required |
| arXiv papers | [arXiv API](https://arxiv.org/help/api) | Free, no auth, Atom XML |
| HF papers | Scraper (BeautifulSoup) | Community-curated, high signal |
| Email | [Resend](https://resend.com) | Modern API, 3000 free/month |
| Scheduling | [APScheduler](https://apscheduler.readthedocs.io) | Lightweight background jobs |
| Templating | [Jinja2](https://jinja.palletsprojects.com) | Industry-standard HTML templates |
| Deployment | [HuggingFace Spaces](https://huggingface.co/spaces) | Free hosting, public URL |

---

## ⚠️ Known Limitations (Free Tier)

| Limitation | Cause | Workaround |
|-----------|-------|------------|
| Scheduled jobs lost on restart | APScheduler stores jobs in memory | Use HF persistent Spaces or external cron |
| Space goes to sleep after 15min inactivity | HF free tier | Upgrade Space or use a keep-alive ping service |
| ~60-90 sec pipeline time | Mistral cold starts on free HF Inference API | Expected on free tier |

---

## 📄 License

MIT — free to use, modify, and deploy.

---
