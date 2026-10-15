import re
from pathlib import Path

from flask import (
    Flask,
    render_template,
    request,
    jsonify,
    send_from_directory,
)
from werkzeug.utils import secure_filename

from main import (
    quick_answer,
    researcher_agent,
    analyst_agent,
    writer_agent,
    save_report,
    save_chat_message,
    get_chat_history,
    list_conversations,
    REPORTS_DIR,
)

app = Flask(__name__)

# Your published GitHub Pages frontend.
FRONTEND_ORIGIN = "https://vinay-battula.github.io"


# ==================================================
# CORS: ALLOW YOUR GITHUB PAGES FRONTEND
# ==================================================

@app.after_request
def add_cors_headers(response):
    origin = request.headers.get("Origin", "")

    if (
        origin == FRONTEND_ORIGIN
        or origin == FRONTEND_ORIGIN + "/"
    ):
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Vary"] = "Origin"
        response.headers["Access-Control-Allow-Headers"] = (
            "Content-Type, Authorization"
        )
        response.headers["Access-Control-Allow-Methods"] = (
            "GET, POST, OPTIONS"
        )

    return response


# ==================================================
# HOME PAGE
# ==================================================

@app.route("/")
def home():
    return render_template("index.html")


# ==================================================
# HEALTH CHECK
# ==================================================

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "application": "AskMe AI",
        "message": "The Flask backend is running.",
    })


# ==================================================
# FAST CHAT ENDPOINT
# ==================================================

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}

    message = data.get("message", "")
    history = data.get("history", [])
    conversation_id = data.get("conversation_id", "default")

    if not isinstance(message, str):
        return jsonify({"error": "Message must be text."}), 400

    message = message.strip()

    if not message:
        return jsonify({"error": "Please enter a question."}), 400

    if len(message) > 2000:
        return jsonify({
            "error": "Keep your question under 2000 characters."
        }), 400

    if not isinstance(conversation_id, str):
        return jsonify({
            "error": "Invalid conversation ID."
        }), 400

    conversation_id = conversation_id.strip()

    if not conversation_id or len(conversation_id) > 100:
        return jsonify({
            "error": "Conversation ID must contain 1–100 characters."
        }), 400

    if not isinstance(history, list):
        history = []

    history = [
        item for item in history[-8:]
        if isinstance(item, dict)
        and item.get("role") in ("user", "assistant")
        and isinstance(item.get("content"), str)
    ]

    try:
        # Save the user's question.
        save_chat_message(conversation_id, "user", message)

        answer = quick_answer(message, history)

        # Save the AI's response.
        save_chat_message(conversation_id, "assistant", answer)

        return jsonify({
            "reply": answer,
            "conversation_id": conversation_id,
        }), 200

    except Exception:
        app.logger.exception("Chat failed")
        return jsonify({
            "error": (
                "Chat failed. Check that Ollama is running, "
                "the model is installed, and the Flask logs."
            )
        }), 500


# ==================================================
# LIST SAVED CONVERSATIONS
# ==================================================

@app.route("/api/conversations", methods=["GET"])
def conversations():
    try:
        return jsonify({
            "conversations": list_conversations()
        }), 200

    except Exception:
        app.logger.exception("Could not load conversations")
        return jsonify({
            "error": "Could not load saved conversations."
        }), 500


# ==================================================
# LOAD A CONVERSATION
# ==================================================

@app.route("/api/history/<conversation_id>", methods=["GET"])
def conversation_history(conversation_id):
    if not conversation_id or len(conversation_id) > 100:
        return jsonify({
            "error": "Invalid conversation ID."
        }), 400

    try:
        history = get_chat_history(conversation_id)

        return jsonify({
            "conversation_id": conversation_id,
            "history": history,
        }), 200

    except Exception:
        app.logger.exception("Could not load chat history")
        return jsonify({
            "error": "Could not load chat history."
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
        return jsonify({
            "error": "Please enter a research topic."
        }), 400

    if len(message) > 2000:
        return jsonify({
            "error": "Keep your topic under 2000 characters."
        }), 400

    try:
        research_text, sources = researcher_agent(message)
        analysis = analyst_agent(message, research_text)

        report = writer_agent(
            message,
            research_text,
            analysis,
            sources,
        )

        # The updated main.py returns a dictionary of file paths.
        saved = save_report(
            message,
            research_text,
            analysis,
            report,
            sources,
        )

        pdf_path = Path(saved["pdf"])
        markdown_path = Path(saved["markdown"])

        return jsonify({
            "reply": report,
            "report_file": pdf_path.name,
            "report_url": f"/download/report/{pdf_path.name}",
            "markdown_file": markdown_path.name,
        }), 200

    except Exception:
        app.logger.exception("Research workflow failed")
        return jsonify({
            "error": (
                "Research failed. Check Ollama, the installed model, "
                "your internet connection, and the Flask logs."
            )
        }), 500


# ==================================================
# DOWNLOAD GENERATED REPORTS
# ==================================================

@app.route("/download/report/<filename>", methods=["GET"])
def download_report(filename):
    safe_name = secure_filename(filename)

    if (
        not safe_name
        or safe_name != filename
        or not safe_name.lower().endswith((".pdf", ".md"))
    ):
        return jsonify({
            "error": "Invalid report filename."
        }), 400

    report_path = REPORTS_DIR / safe_name

    if not report_path.is_file():
        return jsonify({
            "error": "Report not found. Generate the report again."
        }), 404

    return send_from_directory(
        str(REPORTS_DIR),
        safe_name,
        as_attachment=True,
    )


# ==================================================
# START SERVER
# ==================================================

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(__import__("os").environ.get("PORT", "5000")),
        debug=False,
        use_reloader=False,
    )
