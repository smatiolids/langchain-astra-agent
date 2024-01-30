# app.py
from flask import Flask, jsonify, request
from dotenv import load_dotenv
load_dotenv(override=True)


from service_orders_assistant import generate_response as service_order_agent
from flight_assistant import generate_response as flight_agent

app = Flask(__name__)

@app.route('/api/service-orders-assistant', methods=['POST'])
def api_flight_assistant_generate_response():
    question = request.json.get('question', '')
    response = service_order_agent(question)
    return jsonify(response=response)


@app.route('/api/flight-assistant', methods=['POST'])
def api_flight_assistant_generate_response():
    question = request.json.get('question', '')
    response = flight_agent(question)
    return jsonify(response=response)

if __name__ == '__main__':
    app.run(debug=True)
