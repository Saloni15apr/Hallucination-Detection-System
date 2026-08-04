from flask import Flask, render_template, request

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/login")
def login():
    return render_template("login.html")


@app.route("/register")
def register():
    return render_template("register.html")


@app.route("/dashboard", methods=["GET", "POST"])
def dashboard():

    result = None

    if request.method == "POST":

        prompt = request.form["prompt"]

        # Dummy Hallucination Detection Result
        result = {
            "score": 0.82,
            "response": [
                {
                    "text": "Artificial Intelligence is transforming healthcare.",
                    "class": "green"
                },
                {
                    "text": "Humans have three hearts.",
                    "class": "red"
                },
                {
                    "text": "Machine Learning is a subset of Artificial Intelligence.",
                    "class": "green"
                },
                {
                    "text": "The Moon is made of cheese.",
                    "class": "red"
                }
            ]
        }

    return render_template("dashboard.html", result=result)


@app.route("/history")
def history():

    history = [
        {"prompt": "What is Artificial Intelligence?", "score": 0.94},
        {"prompt": "Explain Quantum Computing", "score": 0.81},
        {"prompt": "Tell me about Mars", "score": 0.72}
    ]

    return render_template("history.html", history=history)


@app.route("/help")
def help_page():
    return render_template("help.html")


if __name__ == "__main__":
    app.run(debug=True)