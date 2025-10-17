from flask import Flask, render_template, jsonify, request
import random
from datetime import datetime, timedelta
from dcd_pricer import DCDPricer

app = Flask(__name__)
dcd_pricer = DCDPricer()

# Generate dummy price data
def generate_dummy_data():
    prices = []
    dates = []
    base_price = 1.05  # Example: EUR/USD price
    current_date = datetime.now()
    
    for i in range(30):
        price = base_price + random.uniform(-0.02, 0.02)
        prices.append(round(price, 4))
        date = current_date - timedelta(days=i)
        dates.append(date.strftime('%Y-%m-%d'))
    
    return list(reversed(dates)), list(reversed(prices))

@app.route('/')
def index():
    dates, prices = generate_dummy_data()
    return render_template('index.html', dates=dates, prices=prices)

@app.route('/calculate_dcd', methods=['POST'])
def calculate_dcd():
    data = request.json
    currency_pair = data.get('currency_pair')
    spot_price = float(data.get('spot_price'))
    notional = float(data.get('notional'))
    tenor_days = int(data.get('tenor_days'))
    strike_direction = data.get('strike_direction', 'PUT')
    
    try:
        result = dcd_pricer.calculate_dcd(
            currency_pair=currency_pair,
            spot_price=spot_price,
            notional=notional,
            tenor_days=tenor_days,
            strike_direction=strike_direction
        )
        return jsonify({"success": True, "result": result})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400

@app.route('/get_rates')
def get_rates():
    # Get actual rates from DCD pricer
    rates = {}
    currency_pairs = ['AUDUSD', 'GBPUSD', 'EURUSD', 'USDJPY', 'USDMYR']
    tenors = {
        '1_week': 7,
        '2_weeks': 14,
        '1_month': 30,
        '3_months': 90
    }
    
    for pair in currency_pairs:
        rates[pair] = {}
        for tenor_name, days in tenors.items():
            result = dcd_pricer.calculate_dcd(pair, 1.0, 100000, days)
            rates[pair][tenor_name] = result['enhanced_yield']
    
    return jsonify(rates)

if __name__ == '__main__':
    app.run(debug=True, port=8080)
