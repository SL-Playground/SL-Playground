from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('isl_new.html')

@app.route('/runcode')
def run_code():
    # Add code to execute your Python script here
    # For example:
    import subprocess
    result = subprocess.run(['python', 'ASL_Words_modal.py'], capture_output=True)
    return result.stdout.decode('utf-8')

if __name__ == '__main__':
    app.run(debug=True)
