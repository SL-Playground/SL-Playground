from flask import Flask

app = Flask(__name__)

@app.route('/')
def index():
    return '''
    <html>
    <body>
        <h1>Hello, World!</h1>
        <button onclick="runPython()">Run Python</button>
        <script>
            function runPython() {
                fetch('/runcode')
                .then(response => response.text())
                .then(data => console.log(data))
                .catch(error => console.error('Error:', error));
            }
        </script>
    </body>
    </html>
    '''

@app.route('/runcode')
def run_code():
    # Add code to execute your Python script here
    # For example:
    import subprocess
    result = subprocess.run(['python', 'helppython.py'], capture_output=True)
    return result.stdout.decode('utf-8')

if __name__ == '__main__':
    app.run(debug=True)
