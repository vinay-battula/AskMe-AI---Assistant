
from flask import Flask, render_template, request, jsonify

from main import (
    quick_answer,
    researcher_agent,
    analyst_agent,
    writer_agent,
    save_report,
)

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


# ==================================================
# FAST CHAT ENDPOINT
# ==================================================

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}

    message = data.get("message", "")
    history = data.get("history", [])

    if not isinstance(message, str):
        return jsonify({"error": "Message must be text."}), 400

    message = message.strip()

    if not message:
        return jsonify({"error": "Please enter a question."}), 400

    if len(message) > 2000:
        return jsonify({"error": "Keep your question under 2000 characters."}), 400

    if not isinstance(history, list):
        history = []

    # Limit history size and validate message objects.
    history = [
        item for item in history[-8:]
        if isinstance(item, dict)
        and item.get("role") in ("user", "assistant")
        and isinstance(item.get("content"), str)
    ]

    try:
        answer = quick_answer(message, history)
        return jsonify({"reply": answer}), 200

    except Exception:
        app.logger.exception("Fast chat failed")
        return jsonify({
            "error": (
                "Chat failed. Check that Ollama is running and "
                "the configured model is installed."
            )
        }), 500


# ==================================================
# THREE-AGENT RESEARCH ENDPOINT
# ==================================================

@app.route("/research", methods=["POST"])
def research():
    data = request.get_json(silent=True) or {}
    message = data.get("message", "")

    if not isinstance(message, str):
        return jsonify({"error": "Message must be text."}), 400

    message = message.strip()

    if not message:
        return jsonify({"error": "Please enter a research topic."}), 400

    if len(message) > 2000:
        return jsonify({"error": "Keep your topic under 2000 characters."}), 400

    try:
        research_text, sources = researcher_agent(message)
        analysis = analyst_agent(message, research_text)
        report = writer_agent(
            message, research_text, analysis, sources
        )

        report_path = save_report(
            message, research_text, analysis, report, sources
        )

        return jsonify({
            "reply": report,
            "report_file": report_path.name,
        }), 200

    except Exception:
        app.logger.exception("Research workflow failed")
        return jsonify({
            "error": (
                "Research failed. Check Ollama, the installed model, "
                "your internet connection, and the Flask terminal."
            )
        }), 500


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
        use_reloader=False,
    )
