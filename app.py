
from flask import Flask, render_template, request, redirect, session
from services.llm_service import analyze_statement
from services.database_service import (
    init_database,
    register_user,
    login_user,
    save_analysis,
    get_history
)

app = Flask(__name__)

app.secret_key = "selfchkgpt-secret-key"

# Create database tables
init_database()


# ==============================
# HOME
# ==============================

@app.route("/")
def home():
    return render_template("index.html")


# ==============================
# LOGIN
# ==============================

@app.route("/login", methods=["GET", "POST"])
def login():

    error = None

    if request.method == "POST":

        email = request.form["email"]
        password = request.form["password"]

        user = login_user(email, password)

        if user:

            session["email"] = email
            session["name"] = user[1]

            return redirect("/dashboard")

        else:

            error = "Invalid email or password"

    return render_template(
        "login.html",
        error=error
    )


# ==============================
# REGISTER
# ==============================

@app.route("/register", methods=["GET", "POST"])
def register():

    error = None

    if request.method == "POST":

        name = request.form["name"]
        email = request.form["email"]
        password = request.form["password"]

        success = register_user(
            name,
            email,
            password
        )

        if success:

            return redirect("/login")

        else:

            error = "Email already registered"

    return render_template(
        "register.html",
        error=error
    )


# ==============================
# DASHBOARD / ANALYZE
# ==============================

@app.route("/dashboard", methods=["GET", "POST"])
def dashboard():

    # Force login
    if "email" not in session:
        return redirect("/login")

    result = None

    if request.method == "POST":

        prompt = request.form["prompt"]

        # Gemini analysis
        analysis = analyze_statement(prompt)

        score = analysis["score"]
        label = analysis["label"]
        explanation = analysis["explanation"]

        # Save to database
        save_analysis(
            session["email"],
            prompt,
            score,
            label,
            explanation
        )

        result = {
            "score": score,

            "response": [
                {
                    "text": explanation,

                    "class": (
                        "green"
                        if label == "TRUE"
                        else "red"
                        if label == "FALSE"
                        else "yellow"
                    )
                }
            ]
        }

    return render_template(
        "dashboard.html",
        result=result
    )


# ==============================
# HISTORY
# ==============================

@app.route("/history")
def history():

    # Force login
    if "email" not in session:
        return redirect("/login")

    records = get_history(
        session["email"]
    )

    history_data = []

    for record in records:

        history_data.append({

            "prompt": record[0],

            "score": record[1],

            "label": record[2],

            "explanation": record[3]

        })

    return render_template(
        "history.html",
        history=history_data
    )


# ==============================
# LOGOUT
# ==============================

@app.route("/logout")
def logout():

    session.clear()

    return redirect("/")


# ==============================
# HELP
# ==============================

@app.route("/help")
def help_page():

    return render_template(
        "help.html"
    )


# ==============================
# RUN
# ==============================

if __name__ == "__main__":

    app.run(debug=True)
