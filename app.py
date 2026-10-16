
import os
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


# ==================================================
# APP CONFIGURATION
# ==================================================

app = Flask(__name__)

FRONTEND_ORIGIN = "https://vinay-battula.github.io"
MAX_MESSAGE_LENGTH = 2000
MAX_CONVERSATION_ID_LENGTH = 100


# ==================================================
# CORS: GITHUB PAGES FRONTEND
# ==================================================

@app.after_request
def add_cors_headers(response):
    origin = request.headers.get("Origin", "")

    if origin == FRONTEND_ORIGIN:
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

@app.route("/", methods=["GET"])
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
    }), 200


# ==================================================
# VALIDATE REQUEST MESSAGE
# ==================================================

def validate_message(data, field_name="message"):
    if not isinstance(data, dict):
        return None, "Invalid request data."

    message = data.get(field_name, "")

    if not isinstance(message, str):
        return None, "Message must be text."

    message = message.strip()

    if not message:
        return None, "Please enter a message."

    if len(message) > MAX_MESSAGE_LENGTH:
        return None, (
            f"Message must be under "
            f"{MAX_MESSAGE_LENGTH} characters."
        )

    return message, None


# ==================================================
# CHAT ENDPOINT
# ==================================================

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}

    message, error = validate_message(data)

    if error:
        return jsonify({"error": error}), 400

    conversation_id = data.get("conversation_id", "default")

    if not isinstance(conversation_id, str):
        return jsonify({
            "error": "Invalid conversation ID."
        }), 400

    conversation_id = conversation_id.strip()

    if (
        not conversation_id
        or len(conversation_id) > MAX_CONVERSATION_ID_LENGTH
    ):
        return jsonify({
            "error": "Conversation ID must contain 1–100 characters."
        }), 400

    history = data.get("history", [])

    if not isinstance(history, list):
        history = []

    # Accept only recent, valid chat messages.
    history = [
        {
            "role": item["role"],
            "content": item["content"],
        }
        for item in history[-8:]
        if isinstance(item, dict)
        and item.get("role") in ("user", "assistant")
        and isinstance(item.get("content"), str)
        and len(item["content"]) <= MAX_MESSAGE_LENGTH
    ]

    try:
        # Store the user's message.
        save_chat_message(
            conversation_id,
            "user",
            message,
        )

        # Generate the response using Ollama.
        answer = quick_answer(message, history)

        if not isinstance(answer, str):
            raise ValueError(
                "The AI response must be a string."
            )

        # Store the AI response.
        save_chat_message(
            conversation_id,
            "assistant",
            answer,
        )

        return jsonify({
            "reply": answer,
            "conversation_id": conversation_id,
        }), 200

    except Exception:
        app.logger.exception("Chat request failed")

        return jsonify({
            "error": (
                "Chat failed. Check that Ollama is running, "
                "your model is installed, and the Flask terminal "
                "for error details."
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
        app.logger.exception(
            "Could not load conversations"
        )

        return jsonify({
            "error": "Could not load saved conversations."
        }), 500


# ==================================================
# LOAD CONVERSATION HISTORY
# ==================================================

@app.route(
    "/api/history/<conversation_id>",
    methods=["GET"],
)
def conversation_history(conversation_id):
    if (
        not conversation_id
        or len(conversation_id) > MAX_CONVERSATION_ID_LENGTH
    ):
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
        app.logger.exception(
            "Could not load conversation history"
        )

        return jsonify({
            "error": "Could not load chat history."
        }), 500


# ==================================================
# MULTI-AGENT RESEARCH
# ==================================================

@app.route("/research", methods=["POST"])
def research():
    data = request.get_json(silent=True) or {}

    topic, error = validate_message(data)

    if error:
        return jsonify({
            "error": error.replace("message", "research topic")
        }), 400

    try:
        # Agent 1: Researcher gathers information and sources.
        research_text, sources = researcher_agent(topic)

        # Agent 2: Analyst examines the research.
        analysis = analyst_agent(
            topic,
            research_text,
        )

        # Agent 3: Writer creates the final report.
        report = writer_agent(
            topic,
            research_text,
            analysis,
            sources,
        )

        # Save PDF and Markdown reports.
        saved = save_report(
            topic,
            research_text,
            analysis,
            report,
            sources,
        )

        pdf_path = Path(saved["pdf"]).resolve()
        markdown_path = Path(saved["markdown"]).resolve()
        reports_dir = Path(REPORTS_DIR).resolve()

        # Ensure the generated files are inside reports/.
        if (
            pdf_path.parent != reports_dir
            or markdown_path.parent != reports_dir
        ):
            raise ValueError(
                "Report files must be saved inside the reports directory."
            )

        if not pdf_path.is_file() or not markdown_path.is_file():
            raise FileNotFoundError(
                "The report files were not created successfully."
            )

        return jsonify({
            "reply": report,
            "report_file": pdf_path.name,
            "report_url": (
                f"/download/report/{pdf_path.name}"
            ),
            "markdown_file": markdown_path.name,
            "markdown_url": (
                f"/download/report/{markdown_path.name}"
            ),
        }), 200

    except Exception:
        app.logger.exception(
            "Multi-agent research workflow failed"
        )

        return jsonify({
            "error": (
                "Research failed. Check Ollama, the installed model, "
                "your internet connection, and the Flask terminal."
            )
        }), 500


# ==================================================
# DOWNLOAD PDF OR MARKDOWN REPORT
# ==================================================

@app.route(
    "/download/report/<filename>",
    methods=["GET"],
)
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

    reports_dir = Path(REPORTS_DIR).resolve()
    report_path = (reports_dir / safe_name).resolve()

    # Prevent access to files outside the reports directory.
    if report_path.parent != reports_dir:
        return jsonify({
            "error": "Invalid report path."
        }), 400

    if not report_path.is_file():
        return jsonify({
            "error": (
                "Report not found. Please generate the report again."
            )
        }), 404

    return send_from_directory(
        str(reports_dir),
        safe_name,
        as_attachment=True,
    )


# ==================================================
# START FLASK SERVER
# ==================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False,
        use_reloader=False,
    )
