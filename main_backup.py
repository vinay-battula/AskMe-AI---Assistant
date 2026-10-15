
import re
from pathlib import Path

from openai import OpenAI

# Connect to Ollama running locally on your computer.
# This uses Ollama's OpenAI-compatible API, not OpenAI's cloud API.
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",
)

MODEL = "llama3.2"

# Create the reports folder next to this Python file.
BASE_DIR = Path(__file__).resolve().parent
REPORT_DIR = BASE_DIR / "reports"
REPORT_DIR.mkdir(exist_ok=True)


def research_topic(topic: str) -> str:
    """Generate a research report using a local Ollama model."""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": """
You are an AI research assistant.

Produce clear, structured and informative research reports.
Explain important concepts, applications, benefits and limitations.
Distinguish established facts from uncertainty.
Never invent citations, statistics or sources.
You are running locally and do not have live web search
in this version. Do not claim to have searched the internet.
Recommend reliable sources for further reading when appropriate.
""",
            },
            {
                "role": "user",
                "content": f"""
Prepare a research report about:

{topic}

Include these sections:
1. Introduction
2. Explanation of the topic
3. Key applications
4. Benefits
5. Limitations and challenges
6. Conclusion
7. Suggested sources for further reading
8. References
9. Additional resources
10. Answers
Use Markdown headings and clear language.
""",
            },
        ],
    )

    report = response.choices[0].message.content

    if not report:
        raise RuntimeError("The local model returned an empty report.")

    return report


def main():
    print("=" * 50)
    print("AGENTIC AI RESEARCH ASSISTANT")
    print("Powered by Ollama (Local AI)")
    print("=" * 50)

    topic = input("\nEnter a research topic: ").strip()

    if not topic:
        print("Please enter a valid topic.")
        return

    print("\nResearching your topic...")
    print("Please wait.\n")

    try:
        report = research_topic(topic)

        print("=" * 50)
        print("RESEARCH REPORT")
        print("=" * 50)
        print(report)

        # Create a safe filename for the report.
        safe_name = re.sub(r'[<>:"/\\|?*]', "_", topic)
        safe_name = safe_name.strip(" .")[:80] or "research_report"

        report_path = REPORT_DIR / f"{safe_name}.md"
        report_path.write_text(report, encoding="utf-8")

        print(f"\nReport saved to: {report_path}")

    except Exception as error:
        print("\nThe research assistant could not complete the task.")
        print(f"Error: {error}")
        print("\nCheck that Ollama is running and the model is installed.")
        print("To install the model, run: ollama pull llama3.2")


if __name__ == "__main__":
    main()
