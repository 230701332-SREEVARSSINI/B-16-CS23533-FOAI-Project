from flask import Flask, jsonify, request, render_template, Response
from test4 import VehicleMonitor, DataStreamSimulator, ISSUE_DETAILS
import pandas as pd
import random
import subprocess
import json
import os
import math

app = Flask(__name__, static_folder='./static', template_folder='./templates')

def clean_value(value, default=0):
    """Convert NaN and None values to a default value for JSON serialization"""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return default
    return value

# Initialize your main monitor and data stream
monitor = VehicleMonitor()
stream = DataStreamSimulator("project/exp1_14drivers_14cars_dailyRoutes.csv")


# ---------- ROUTE: Home UI ----------
@app.route('/')
def home():
    return render_template('./index.html')


# ---------- ROUTE: Live Vehicle Data ----------
@app.route('/api/live-data', methods=['GET'])
def get_live_data():
    sample = stream.get_next_sample()
    result = monitor.predict_and_diagnose(sample)

    issue_data = []
    for issue in result["issues"][:1]:  # only one issue shown
        detail = ISSUE_DETAILS.get(issue["name"], {})
        issue_data.append({
            "name": issue["name"],
            "description": issue["description"],
            "immediate_cost": detail.get("immediate_cost", 0),
            "future_costs": detail.get("consequences", [])
        })

    return jsonify({
        "temperature": round(clean_value(result["temperature"], 90.0), 1),
        "rpm": clean_value(sample.get("RPM"), random.randint(1000, 1600)),
        "engine_load": clean_value(sample.get("ENGINE_LOAD"), 0),
        "o2_sensor": clean_value(sample.get("O2_SENSOR"), 0),
        "fuel_efficiency": clean_value(sample.get("FUEL_EFFICIENCY"), 25.0),
        "system_health": 100,
        "emissions": "Normal" if not result["issues"] else "Issue Detected",
        "maintenance_due": "2.5k km",
        "issues": issue_data,
        "alerts": result["system_status"]
    })


# ---------- ROUTE: Manually Run test4.py via Subprocess ----------
@app.route('/run-test4', methods=['GET'])
def run_test4():
    try:
        result = subprocess.run(
            ['python', 'project/test4.py'],  # Adjust path if needed
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',
            errors='replace')
        output_lines = result.stdout.strip().split('\n')
        return Response(json.dumps(
            {
                "status": "success",
                "output_lines": output_lines
            }, indent=4),
                        mimetype='application/json')
    except Exception as e:
        return Response(json.dumps({
            "status": "error",
            "message": str(e)
        },
                                   indent=4),
                        mimetype='application/json'), 500


# ---------- Flask Entry Point ----------
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
