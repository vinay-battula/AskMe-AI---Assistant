
import re
from datetime import datetime
from pathlib import Path

import ollama
from ddgs import DDGS


# ==================================================
# CONFIGURATION
# ==================================================

MODEL = "llama3.2:1b"  # Use this model after pulling it with Ollama
CONTEXT_SIZE = 2048
CHAT_TOKENS = 100
AGENT_TOKENS = 180

PROJECT_DIR = Path(__file__).resolve().parent
REPORTS_DIR = PROJECT_DIR / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


# ==================================================
# OLLAMA AI FUNCTION
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
            "temperature": 0.3,
        },
        keep_alive=-1,
    )

    return response.message.content.strip()


# ==================================================
# FAST CHAT WITH CONVERSATION HISTORY
# ==================================================


import ollama

MODEL = "llama3.2:latest"  # Use the exact name from: ollama list


def quick_answer(question, history=None):
    messages = [
        {
            "role": "system",
            "content": """
You are AskMe AI, a helpful programming and general-purpose assistant.

IMPORTANT RULES:
1. Answer the exact question the user asks.
2. Always provide complete answers. Never intentionally stop halfway.
3. When asked for code, provide a complete, runnable program.
4. Include all required imports, the class, the main method,
   all closing braces, and any required statements.
5. Put the complete code first, then a short explanation.
6. For Java programs, include the complete class and main method.
7. Do not replace the end of a program with "..." or omit code.
8. Use Markdown code fences with the correct programming language.
9. If an answer needs to be long, finish the code before explaining it.
10. For follow-up questions, use the conversation history.
"""
        }
    ]

    if isinstance(history, list):
        for item in history[-6:]:
            if (
                isinstance(item, dict)
                and item.get("role") in ("user", "assistant")
                and isinstance(item.get("content"), str)
            ):
                messages.append({
                    "role": item["role"],
                    "content": item["content"][:2000]
                })

    messages.append({
        "role": "user",
        "content": question
    })

    response = ollama.chat(
        model=MODEL,
        messages=messages,
        options={
            "num_predict": 1200,
            "num_ctx": 4096,
            "temperature": 0.2
        },
        keep_alive=-1
    )

    answer = response.message.content.strip()

    if not answer:
        raise RuntimeError("The AI returned an empty answer.")

    return answer

    # The current question is sent separately.
    messages.append({"role": "user", "content": question})

    response = ollama.chat(
        model=MODEL,
        messages=messages,
        options={
            "num_predict": CHAT_TOKENS,
            "num_ctx": CONTEXT_SIZE,
            "temperature": 0.3,
        },
        keep_alive=-1,
    )

    return response.message.content.strip()


# ==================================================
# AGENT 1: RESEARCHER
# ==================================================

def researcher_agent(topic):
    print("\n[1/3] RESEARCHER AGENT")
    print("Searching the web...")

    with DDGS() as search_engine:
        results = list(search_engine.text(topic, max_results=3))

    if not results:
        raise RuntimeError("No web results found. Check your internet connection.")

    sources = []
    notes = []

    for number, result in enumerate(results, start=1):
        title = result.get("title") or "Untitled source"
        url = result.get("href") or ""
        snippet = (result.get("body") or "")[:500]

        sources.append({"title": title, "url": url})
        notes.append(
            f"Source {number}\nTitle: {title}\nURL: {url}\n"
            f"Information: {snippet}"
        )

    research = ask_ollama(
        "Researcher",
        (
            "Summarize the supplied web search results. Identify important "
            "facts and applications. Do not invent facts or references. "
            "Mention uncertainty and keep the summary concise."
        ),
        f"Topic: {topic}\n\nSearch results:\n" + "\n\n".join(notes),
    )

    return research, sources


# ==================================================
# AGENT 2: ANALYST
# ==================================================

def analyst_agent(topic, research):
    print("\n[2/3] ANALYST AGENT")

    return ask_ollama(
        "Analyst",
        (
            "Analyze the supplied research. Explain main findings, benefits, "
            "applications, limitations, risks, and uncertainty. Do not invent "
            "evidence. Use beginner-friendly English and be concise."
        ),
        f"Topic: {topic}\n\nResearch:\n{research}",
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
        (
            "Write a clear, concise Markdown research report with a title, "
            "summary, introduction, main findings, applications, benefits, "
            "limitations, conclusion, and references. Use only the supplied "
            "research and analysis. Do not invent facts or URLs."
        ),
        f"""
Topic: {topic}

Research:
{research}

Analysis:
{analysis}

Sources:
{references}

Write the report now.
""",
        max_tokens=450,
    )


# ==================================================
# SAVE REPORT
# ==================================================

def save_report(topic, research, analysis, report, sources):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    safe_topic = re.sub(
        r"[^a-zA-Z0-9_-]+", "_", topic
    ).strip("_")[:60] or "research"

    report_path = REPORTS_DIR / f"{safe_topic}_{timestamp}.md"

    references = "\n".join(
        f"- [{source['title']}]({source['url']})"
        if source["url"]
        else f"- {source['title']}"
        for source in sources
    )

    content = f"""# Research Report

**Topic:** {topic}

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

    report_path.write_text(content, encoding="utf-8")
    return report_path


# ==================================================
# OPTIONAL TERMINAL ENTRY POINT
# ==================================================

def main():
    topic = input("Enter your research topic: ").strip()

    if not topic:
        print("Please enter a topic.")
        return

    try:
        ollama.show(MODEL)

        research, sources = researcher_agent(topic)
        analysis = analyst_agent(topic, research)
        report = writer_agent(topic, research, analysis, sources)

        path = save_report(topic, research, analysis, report, sources)

        print("\n", report)
        print(f"\nReport saved to: {path.resolve()}")

    except Exception as error:
        print(f"Research failed: {type(error).__name__}: {error}")


if __name__ == "__main__":
    main()
