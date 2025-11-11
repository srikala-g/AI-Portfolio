from flask import Flask
from datetime import datetime

app = Flask(__name__)

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route("/time")
def current_time():
    now = datetime.now()
    return f"<p>Current server time: {now.strftime('%Y-%m-%d %H:%M:%S')}</p>"

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)

