import pandas as pd
import joblib
import csv
import random
import time
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import numpy as np

# ========================================
# ISSUE CONSEQUENCES & COSTS (INR)
# ========================================

ISSUE_DETAILS = {
    "Critical O2 Fault": {
        "immediate_cost":
        16500,
        "consequences": [{
            "description": "Catalytic converter may overheat and fail.",
            "future_cost": 123750,
            "future_failures": ["Catalytic Converter"]
        }]
    },
    "O2 Sensor Degradation": {
        "immediate_cost":
        8250,
        "consequences": [{
            "description":
            "Reduced fuel efficiency and higher emissions.",
            "future_cost":
            33000,
            "future_failures": ["Catalytic Converter", "Fuel Economy"]
        }]
    },
    "Extreme Rich Condition": {
        "immediate_cost":
        24750,
        "consequences": [{
            "description": "Spark plugs may foul, leading to misfires.",
            "future_cost": 41250,
            "future_failures": ["Spark Plugs"]
        }]
    },
    "Rich Mixture": {
        "immediate_cost":
        16500,
        "consequences": [{
            "description": "Carbon buildup in engine, reduced performance.",
            "future_cost": 33000,
            "future_failures": ["Engine Valves"]
        }]
    },
    "Extreme Lean Condition": {
        "immediate_cost":
        33000,
        "consequences": [{
            "description": "Possible piston damage and overheating.",
            "future_cost": 123750,
            "future_failures": ["Pistons", "Head Gasket"]
        }]
    },
    "Lean Mixture": {
        "immediate_cost":
        16500,
        "consequences": [{
            "description": "Potential misfires and increased NOx emissions.",
            "future_cost": 41250,
            "future_failures": ["Spark Plugs", "Exhaust Valves"]
        }]
    },
    "Combustion Imbalance": {
        "immediate_cost":
        "8,250",
        "consequences": [{
            "description": "May cause rough idling and poor power delivery.",
            "future_cost": 24750,
            "future_failures": ["Engine Smoothness"]
        }]
    }
}

# ========================
# 1. DATA LOADING & CLEANING
# ========================


def load_and_clean_dataset(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        sample = f.read(2048)
        dialect = csv.Sniffer().sniff(sample)
        delimiter = dialect.delimiter

    df = pd.read_csv(file_path, delimiter=delimiter, low_memory=False)
    df.columns = df.columns.str.upper().str.strip()

    for col in df.select_dtypes(include='object').columns:
        try:
            df[col] = pd.to_numeric(df[col].str.replace(',', '.', regex=False),
                                    errors='coerce')
        except:
            continue

    if 'TIMESTAMP' in df.columns:
        df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'],
                                         unit='ms',
                                         errors='coerce')

    df.dropna(axis=1, how='all', inplace=True)
    df.dropna(thresh=len(df) * 0.2, axis=1, inplace=True)
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    df.drop_duplicates(inplace=True)

    return df


# ========================
# 2. MODEL TRAINING
# ========================


def train_model(df, target_column="ENGINE_COOLANT_TEMP"):
    df = df.select_dtypes(include=['number'])
    df.dropna(inplace=True)

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found.")

    X = df.drop(columns=[target_column])
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.2,
                                                        random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    print(f" Model trained | RMSE: {rmse:.2f} | R2: {r2:.2f}")

    joblib.dump(model, 'rf_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    with open('feature_names.txt', 'w') as f:
        f.write('\n'.join(X.columns))

    return model, scaler, X.columns


# ========================
# 3. VEHICLE MONITORING
# ========================


class VehicleMonitor:

    def __init__(self):
        self.model = joblib.load('rf_model.pkl')
        self.scaler = joblib.load('scaler.pkl')
        with open('feature_names.txt', 'r') as f:
            self.feature_names = [line.strip() for line in f.readlines()]

    def predict_and_diagnose(self, live_data):
        # Clean and convert live data
        cleaned_data = {}
        for k, v in live_data.items():
            try:
                cleaned_data[k.upper().strip()] = float(
                    str(v).replace(',', '.'))
            except:
                cleaned_data[k.upper().strip()] = 0.0

        live_df = pd.DataFrame([cleaned_data])
        for f in set(self.feature_names) - set(live_df.columns):
            live_df[f] = 0.0
        live_df = live_df[self.feature_names]

        live_scaled = self.scaler.transform(live_df)
        pred_temp = self.model.predict(live_scaled)[0]

        issues = self.diagnose_emissions(cleaned_data)
        system_status = self.check_system_health(cleaned_data, pred_temp)

        return {
            'temperature': pred_temp,
            'issues': issues,
            'system_status': system_status
        }

    def diagnose_emissions(self, live_data):
        issues = []
        try:
            o2 = float(live_data.get('O2_SENSOR', 1.0))
            if o2 < 0.2:
                issues.append({
                    "name":
                    "Critical O2 Fault",
                    "description":
                    "Sensor voltage < 0.2V - Immediate attention needed"
                })
            elif o2 < 0.45:
                issues.append({
                    "name": "O2 Sensor Degradation",
                    "description": f"Low voltage ({o2:.2f}V)"
                })

            fuel_trim = float(live_data.get('FUEL_TRIM', 0))
            if fuel_trim > 25:
                issues.append({
                    "name":
                    "Extreme Rich Condition",
                    "description":
                    "Potential fuel leak or injector fault"
                })
            elif fuel_trim > 15:
                issues.append({
                    "name": "Rich Mixture",
                    "description": f"+{fuel_trim}% trim"
                })
            elif fuel_trim < -25:
                issues.append({
                    "name": "Extreme Lean Condition",
                    "description": "Possible vacuum leak"
                })
            elif fuel_trim < -15:
                issues.append({
                    "name": "Lean Mixture",
                    "description": f"{fuel_trim}% trim"
                })

            equiv_ratio = float(live_data.get('EQUIV_RATIO', 1.0))
            if not 0.9 <= equiv_ratio <= 1.1:
                issues.append({
                    "name": "Combustion Imbalance",
                    "description": f"Lamda ratio {equiv_ratio:.2f}"
                })
        except:
            issues.append({
                "name":
                "Sensor Error",
                "description":
                "Could not read one or more sensor values."
            })

        return issues

    def check_system_health(self, live_data, pred_temp):
        status = []
        try:
            if pred_temp > 115:
                status.append(("CRITICAL", "Engine overheating!"))
            elif pred_temp > 105:
                status.append(("WARNING", "High engine temperature"))

            rpm = float(live_data.get('RPM', 0))
            if rpm > 5000:
                status.append(("WARNING", "High RPM"))

            if random.random() > 0.95:
                status.append(("WARNING", "Predicted future low oil pressure"))

            if rpm > 4800 and pred_temp > 110:
                status.append(
                    ("CRITICAL", "Engine strain - may overheat soon"))

        except:
            status.append(
                ("Sensor Fault", "Unable to interpret some sensor values."))

        return status


# ========================
# 4. DATA SIMULATOR
# ========================


class DataStreamSimulator:

    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file, low_memory=False)
        self.current_index = 0

    def get_next_sample(self):
        if self.current_index >= len(self.df):
            self.current_index = 0

        sample = self.df.iloc[self.current_index].to_dict()
        self.current_index += 1

        for key in sample:
            val = sample[key]
            if isinstance(val, str) and ',' in val:
                try:
                    sample[key] = float(val.replace(',', '.'))
                except:
                    pass

        if random.random() > 0.9:
            if 'O2_SENSOR' in sample:
                sample['O2_SENSOR'] *= 0.5
            if 'FUEL_TRIM' in sample:
                sample['FUEL_TRIM'] += random.choice([15, -15])

        return sample


# ========================
# 5. MAIN EXECUTION
# ========================

if __name__ == "__main__":
    file_path = "project/exp1_14drivers_14cars_dailyRoutes.csv"
    try:
        print(" Loading and cleaning data...")
        df = load_and_clean_dataset(file_path)
        print(" Data loaded successfully")

        print("\n Training model...")
        train_model(df, target_column="ENGINE_COOLANT_TEMP")

        print("\n Starting Vehicle Monitoring System...")
        monitor = VehicleMonitor()
        stream = DataStreamSimulator(file_path)

        for _ in range(30):
            live_data = stream.get_next_sample()
            results = monitor.predict_and_diagnose(live_data)

            print(
                f"\n Live Analysis [{pd.Timestamp.now().strftime('%H:%M:%S')}]"
            )
            print(f"  Predicted Temp: {results['temperature']:.1f}°C")
            print(
                f"  RPM: {live_data.get('RPM', 'N/A')} | Load: {live_data.get('ENGINE_LOAD', 'N/A')}%"
            )

            if results['issues']:
                print("\n Emission Issues & Cost Analysis:")
                for issue in results['issues'][:3]:
                    name = issue['name']
                    desc = issue['description']
                    print(f"\n Issue: {name}")
                    print(f"    Description: {desc}")

                    detail = ISSUE_DETAILS.get(name)
                    if detail:
                        print(
                            f"    Immediate Repair Cost: INR {detail['immediate_cost']:,}"
                        )
                        for c in detail['consequences']:
                            print(f"    If delayed: {c['description']}")
                            print(
                                f"      Future failures: {', '.join(c['future_failures'])}"
                            )
                            print(
                                f"      Future repair cost: INR {c['future_cost']:,}"
                            )
                    else:
                        print("    No cost data available.")

            if results['system_status']:
                print("\n System Alerts:")
                for alert in results['system_status']:
                    print(f"  - {alert[0]}: {alert[1]}")

            time.sleep(1)

    except FileNotFoundError:
        print(f" Error: File not found at {file_path}")
    except Exception as e:
        print(f" An error occurred: {str(e)}")
