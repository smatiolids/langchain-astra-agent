# app.py
from flask import Flask, jsonify, request
from dotenv import load_dotenv
load_dotenv(override=True)


from service_orders_assistant import generate_response

app = Flask(__name__)

@app.route('/api/hello', methods=['GET'])
def hello():
    return jsonify(message='Hello, World!')

@app.route('/api/service-orders-assistant', methods=['POST'])
def api_flight_assistant_generate_response():
    question = request.json.get('question', '')
    response = generate_response(question)
    return jsonify(response=response)

if __name__ == '__main__':
    app.run(debug=True)
