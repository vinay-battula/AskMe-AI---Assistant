
import os
import re
from datetime import datetime
from pathlib import Path
from xml.sax.saxutils import escape

from dotenv import load_dotenv
import ollama
from ddgs import DDGS
from supabase import create_client

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    HRFlowable,
)

# Load environment variables from .env
load_dotenv()


# ==================================================
# CONFIGURATION
# ==================================================

MODEL = "llama3.2:1b"
CONTEXT_SIZE = 4096
CHAT_TOKENS = 1200
AGENT_TOKENS = 450

PROJECT_DIR = Path(__file__).resolve().parent
REPORTS_DIR = PROJECT_DIR / "reports"
DATABASE_PATH = PROJECT_DIR / "askme_history.db"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)




# ==================================================
# DATABASE: SUPABASE CHAT HISTORY
# ==================================================

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError(
        "Missing SUPABASE_URL or SUPABASE_KEY environment variable."
    )

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)


def save_chat_message(conversation_id, role, content):
    if role not in ("user", "assistant"):
        raise ValueError("Role must be user or assistant.")

    supabase.table("chat_messages").insert({
        "conversation_id": str(conversation_id),
        "role": role,
        "message": content,
    }).execute()


def get_chat_history(conversation_id, limit=100):
    safe_limit = max(1, min(int(limit), 500))

    response = (
        supabase.table("chat_messages")
        .select("role, message, created_at")
        .eq("conversation_id", str(conversation_id))
        .order("created_at", desc=True)
        .order("id", desc=True)
        .limit(safe_limit)
        .execute()
    )

    rows = list(reversed(response.data or []))

    return [
        {
            "role": row["role"],
            "content": row["message"],
            "created_at": row["created_at"],
        }
        for row in rows
    ]


def list_conversations():
    response = (
        supabase.table("chat_messages")
        .select("conversation_id, created_at, id")
        .order("id", desc=True)
        .limit(1000)
        .execute()
    )

    conversations = []
    seen = set()

    for row in response.data or []:
        conversation_id = row["conversation_id"]

        if conversation_id not in seen:
            seen.add(conversation_id)
            conversations.append({
                "conversation_id": conversation_id,
                "last_activity": row["created_at"],
            })

    return conversations

# ==================================================
# OLLAMA HELPER
# ==================================================

def ask_ollama(agent_name, instructions, prompt, max_tokens=AGENT_TOKENS):
    print(f"\n[{agent_name}] AI is working...")

    response = ollama.chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": instructions},
            {"role": "user", "content": prompt},
        ],
        options={
            "num_predict": max_tokens,
            "num_ctx": CONTEXT_SIZE,
            "temperature": 0.2,
        },
        keep_alive=-1,
    )

    answer = response.message.content.strip()

    if not answer:
        raise RuntimeError(f"{agent_name} returned an empty answer.")

    return answer


# ==================================================
# SMART CHAT
# ==================================================

CHAT_INSTRUCTIONS = """
You are AskMe AI, a helpful general-purpose assistant.

Understand the user's actual request and answer that request directly.

RESPONSE RULES:
1. If the user asks for information, explain the information clearly.
2. If the user asks for code, provide code for the requested language.
3. If the user asks for a complete program, include all required imports,
   classes, functions, entry points, and closing braces.
4. Do not answer a different programming problem from the one requested.
5. Put runnable code in Markdown code fences.
6. Explain the code briefly after the code when useful.
7. If the user asks to fix code, identify and fix the actual problem.
8. Never deliberately replace missing code with ellipses.
9. Do not invent facts, sources, or test results.
10. If something is unclear, ask a short clarification question.
11. Be clear about uncertainty and limitations.
"""


def quick_answer(question, history=None):
    question = str(question).strip()

    if not question:
        raise ValueError("Please enter a question.")

    messages = [
        {"role": "system", "content": CHAT_INSTRUCTIONS}
    ]

    # Include recent messages from the current conversation.
    if isinstance(history, list):
        for item in history[-8:]:
            if not isinstance(item, dict):
                continue

            role = item.get("role")
            content = item.get("content")

            if role in ("user", "assistant") and isinstance(content, str):
                if content.strip():
                    messages.append({
                        "role": role,
                        "content": content[:3000],
                    })

    messages.append({"role": "user", "content": question})

    response = ollama.chat(
        model=MODEL,
        messages=messages,
        options={
            "num_predict": CHAT_TOKENS,
            "num_ctx": CONTEXT_SIZE,
            "temperature": 0.2,
        },
        keep_alive=-1,
    )

    answer = response.message.content.strip()

    if not answer:
        raise RuntimeError("Ollama returned an empty answer.")

    return answer


# ==================================================
# AGENT 1: RESEARCHER
# ==================================================

def researcher_agent(topic):
    print("\n[1/3] RESEARCHER AGENT")
    print("Searching the web...")

    with DDGS() as search_engine:
        results = list(search_engine.text(topic, max_results=5))

    if not results:
        raise RuntimeError(
            "No web results found. Check your internet connection "
            "or try a different research topic."
        )

    sources = []
    notes = []

    for number, result in enumerate(results, start=1):
        title = result.get("title") or "Untitled source"
        url = result.get("href") or ""
        snippet = (result.get("body") or "")[:700]

        sources.append({
            "title": title,
            "url": url,
        })

        notes.append(
            f"Source {number}\n"
            f"Title: {title}\n"
            f"URL: {url}\n"
            f"Information: {snippet}"
        )

    research = ask_ollama(
        "Researcher",
        """
Summarize the supplied web search results.
Identify important facts and applications.
Do not invent facts, sources, or references.
Mention uncertainty where appropriate.
""",
        f"Topic: {topic}\n\n" + "\n\n".join(notes),
        max_tokens=AGENT_TOKENS,
    )

    return research, sources


# ==================================================
# AGENT 2: ANALYST
# ==================================================

def analyst_agent(topic, research):
    print("\n[2/3] ANALYST AGENT")

    return ask_ollama(
        "Analyst",
        """
Analyze the supplied research.
Explain the main findings, benefits, applications, limitations,
risks, and uncertainty in beginner-friendly English.
Do not invent evidence.
""",
        f"Topic: {topic}\n\nResearch:\n{research}",
        max_tokens=AGENT_TOKENS,
    )


# ==================================================
# AGENT 3: WRITER
# ==================================================

def writer_agent(topic, research, analysis, sources):
    print("\n[3/3] WRITER AGENT")

    references = "\n".join(
        f"- {source['title']}: {source['url']}"
        for source in sources
    )

    return ask_ollama(
        "Writer",
        """
Write a useful Markdown research report with:
title, summary, introduction, main findings, applications,
benefits, limitations, conclusion, and references.
Use the supplied research only. Do not invent facts or URLs.
Write a complete report rather than stopping halfway.
""",
        f"""
Topic: {topic}

Research:
{research}

Analysis:
{analysis}

Available sources:
{references}

Write the complete report.
""",
        max_tokens=AGENT_TOKENS,
    )


# ==================================================
# PDF REPORT CREATION
# ==================================================

def create_pdf_report(title, content, output_path):
    output_path = Path(output_path)

    styles = getSampleStyleSheet()

    styles.add(ParagraphStyle(
        name="ReportTitle",
        parent=styles["Title"],
        fontSize=20,
        leading=26,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#243B53"),
        spaceAfter=18,
    ))

    styles.add(ParagraphStyle(
        name="ReportHeading",
        parent=styles["Heading2"],
        fontSize=13,
        leading=17,
        textColor=colors.HexColor("#243B53"),
        spaceBefore=12,
        spaceAfter=6,
    ))

    styles.add(ParagraphStyle(
        name="ReportBody",
        parent=styles["BodyText"],
        fontSize=9.5,
        leading=14,
        spaceAfter=7,
        wordWrap="CJK",
    ))

    document = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        rightMargin=0.7 * inch,
        leftMargin=0.7 * inch,
        topMargin=0.7 * inch,
        bottomMargin=0.7 * inch,
        title=title,
        author="AskMe AI",
    )

    story = [
        Paragraph(escape(title), styles["ReportTitle"]),
        Paragraph(
            "Generated by AskMe AI on "
            + datetime.now().strftime("%Y-%m-%d %H:%M"),
            styles["Italic"],
        ),
        Spacer(1, 12),
        HRFlowable(
            width="100%",
            thickness=0.8,
            color=colors.lightgrey,
        ),
        Spacer(1, 12),
    ]

    for raw_line in content.splitlines():
        line = raw_line.strip()

        if not line:
            story.append(Spacer(1, 5))
            continue

        if line.startswith("#"):
            heading = line.lstrip("#").strip()
            story.append(
                Paragraph(escape(heading), styles["ReportHeading"])
            )
        elif line.startswith("- ") or line.startswith("* "):
            bullet = escape(line[2:])
            story.append(
                Paragraph(
                    "&bull; " + bullet,
                    styles["ReportBody"],
                )
            )
        else:
            # Basic Markdown cleanup for readable PDF text.
            cleaned = re.sub(r"\*\*(.*?)\*\*", r"\1", line)
            cleaned = re.sub(r"\*(.*?)\*", r"\1", cleaned)
            cleaned = re.sub(r"`([^`]*)`", r"\1", cleaned)

            story.append(
                Paragraph(
                    escape(cleaned),
                    styles["ReportBody"],
                )
            )

    document.build(story)
    return output_path


# ==================================================
# SAVE RESEARCH REPORT AS MARKDOWN AND PDF
# ==================================================

def save_report(topic, research, analysis, report, sources):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    safe_topic = re.sub(
        r"[^a-zA-Z0-9_-]+", "_", topic
    ).strip("_")[:60] or "research"

    filename = f"{safe_topic}_{timestamp}"

    references = "\n".join(
        f"- [{source['title']}]({source['url']})"
        if source["url"]
        else f"- {source['title']}"
        for source in sources
    )

    content = f"""# {topic}

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

**Model:** {MODEL}

---

{report}

## Research Notes

{research}

## Analysis Notes

{analysis}

## Web Sources

{references}
"""

    markdown_path = REPORTS_DIR / f"{filename}.md"
    pdf_path = REPORTS_DIR / f"{filename}.pdf"

    markdown_path.write_text(content, encoding="utf-8")

    create_pdf_report(
        title=f"AskMe AI Research Report: {topic}",
        content=content,
        output_path=pdf_path,
    )

    return {
        "markdown": markdown_path,
        "pdf": pdf_path,
    }


# ==================================================
# RUN COMPLETE RESEARCH WORKFLOW
# ==================================================

def run_research(topic):
    topic = str(topic).strip()

    if not topic:
        raise ValueError("Please enter a research topic.")

    # Fail early with a clear error if the model is unavailable.
    ollama.show(MODEL)

    research, sources = researcher_agent(topic)
    analysis = analyst_agent(topic, research)
    report = writer_agent(topic, research, analysis, sources)

    paths = save_report(
        topic,
        research,
        analysis,
        report,
        sources,
    )

    return {
        "topic": topic,
        "research": research,
        "analysis": analysis,
        "report": report,
        "sources": sources,
        "markdown_path": str(paths["markdown"]),
        "pdf_path": str(paths["pdf"]),
    }


# ==================================================
# TERMINAL ENTRY POINT
# ==================================================


def main():
    print("\n========== ASKME AI RESEARCH ASSISTANT ==========")

    topic = input("Enter your research topic: ").strip()

    if not topic:
        print("Please enter a topic.")
        return

    # Use one conversation ID for the question and its answer.
    conversation_id = datetime.now().strftime("research_%Y%m%d_%H%M%S")

    try:
        # Save the user's research question.
        save_chat_message(
            conversation_id,
            "user",
            topic
        )
        print("\nQuestion saved to Supabase.")

        # Run the existing research workflow.
        result = run_research(topic)

        # Save the completed AI response.
        assistant_message = (
            f"# {result['topic']}\n\n"
            f"{result['report']}\n\n"
            f"Markdown report: {result['markdown_path']}\n\n"
            f"PDF report: {result['pdf_path']}"
        )

        save_chat_message(
            conversation_id,
            "assistant",
            assistant_message
        )

        print("AI response saved to Supabase.")

        print("\n========== RESEARCH REPORT ==========")
        print(result["report"])

        print("\nMarkdown report:")
        print(result["markdown_path"])

        print("\nPDF report:")
        print(result["pdf_path"])

        print("\nConversation ID:")
        print(conversation_id)

    except Exception as error:
        print(
            f"\nResearch failed: "
            f"{type(error).__name__}: {error}"
        )


if __name__ == "__main__":
    main()
