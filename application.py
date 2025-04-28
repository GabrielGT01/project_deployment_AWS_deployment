
import subprocess
import os

def application(environ, start_response):
    start_response('200 OK', [('Content-type', 'text/html')])
    
    # Start Streamlit process
    port = environ.get('PORT', '8080')
    process = subprocess.Popen(['streamlit', 'run', 'house_predictor.py', '--server.port', port])
    
    return [b"Streamlit app is running. Please navigate to the correct URL."]
