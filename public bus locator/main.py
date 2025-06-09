import tkinter as tk
from tkinter import ttk, messagebox
import folium
import threading
import requests
import openrouteservice
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import joblib
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import numpy as np
import webbrowser

# --------------------
# Configurations
# --------------------

ORS_API_KEY = '5b3ce3597851110001cf62489a4438fc736b4c7d8c1e62684610924e'  # Replace with your OpenRouteService API key

MODEL_FILE = "delivery_model.pkl"
LE_VEHICLE_FILE = "le_vehicle.pkl"
LE_TIME_FILE = "le_time.pkl"

DEFAULT_SPEED_KMH = 30

# --------------------
# ML Model Train / Load
# --------------------

def train_ml_model():
    import pandas as pd
    data = pd.DataFrame({
        'distance_km': [15, 120, 10, 45, 5, 300],
        'vehicle': ['car', 'truck', 'bike', 'car', 'bike', 'truck'],
        'weight_kg': [20, 500, 5, 80, 3, 1000],
        'time_of_day': ['morning', 'afternoon', 'evening', 'night', 'morning', 'afternoon'],
        'delivery_time_min': [35, 140, 25, 60, 20, 280]
    })

    le_vehicle = LabelEncoder()
    le_time = LabelEncoder()
    data['vehicle_enc'] = le_vehicle.fit_transform(data['vehicle'])
    data['time_enc'] = le_time.fit_transform(data['time_of_day'])

    X = data[['distance_km', 'vehicle_enc', 'weight_kg', 'time_enc']]
    y = data['delivery_time_min']

    model = RandomForestRegressor()
    model.fit(X, y)

    joblib.dump(model, MODEL_FILE)
    joblib.dump(le_vehicle, LE_VEHICLE_FILE)
    joblib.dump(le_time, LE_TIME_FILE)

    return model, le_vehicle, le_time

def load_ml_model():
    if os.path.exists(MODEL_FILE) and os.path.exists(LE_VEHICLE_FILE) and os.path.exists(LE_TIME_FILE):
        model = joblib.load(MODEL_FILE)
        le_vehicle = joblib.load(LE_VEHICLE_FILE)
        le_time = joblib.load(LE_TIME_FILE)
        return model, le_vehicle, le_time
    else:
        return train_ml_model()

# --------------------
# Main Application
# --------------------

class TransportTrackerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Smart Transport Tracker with Route-Based ETA")

        self.client = openrouteservice.Client(key=ORS_API_KEY)
        self.model, self.le_vehicle, self.le_time = load_ml_model()
        self.geolocator = Nominatim(user_agent="transport_tracker_app")

        # User input frame
        frm = ttk.Frame(root)
        frm.pack(pady=10, padx=10, fill=tk.X)

        ttk.Label(frm, text="Your Location (city or address):").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.location_name = tk.StringVar(value="New York, NY")
        ttk.Entry(frm, textvariable=self.location_name, width=30).grid(row=0, column=1, pady=2)

        ttk.Label(frm, text="Refresh Interval (sec):").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.interval = tk.IntVar(value=20)
        ttk.Entry(frm, textvariable=self.interval).grid(row=1, column=1, pady=2)

        ttk.Label(frm, text="Vehicle Type:").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.vehicle_type = tk.StringVar(value="car")
        ttk.Combobox(frm, textvariable=self.vehicle_type, state="readonly", values=["car", "truck", "bike"]).grid(row=2, column=1, pady=2)

        ttk.Label(frm, text="Weight (kg):").grid(row=3, column=0, sticky=tk.W, pady=2)
        self.weight = tk.DoubleVar(value=50)
        ttk.Entry(frm, textvariable=self.weight).grid(row=3, column=1, pady=2)

        ttk.Label(frm, text="Time of Day:").grid(row=4, column=0, sticky=tk.W, pady=2)
        self.time_of_day = tk.StringVar(value="morning")
        ttk.Combobox(frm, textvariable=self.time_of_day, state="readonly", values=["morning", "afternoon", "evening", "night"]).grid(row=4, column=1, pady=2)

        self.start_btn = ttk.Button(frm, text="Start Tracking", command=self.start_tracking)
        self.start_btn.grid(row=5, column=0, columnspan=2, pady=10)

        self.tracking = False

    def geocode_location(self, location_name):
        try:
            loc = self.geolocator.geocode(location_name)
            if loc:
                return (loc.latitude, loc.longitude)
            else:
                messagebox.showerror("Error", f"Could not geocode location: {location_name}")
                return None
        except Exception as e:
            messagebox.showerror("Error", f"Geocoding error: {e}")
            return None

    def fetch_vehicle_positions(self):
        # Simulated vehicle positions for demo
        vehicles = [
            {"id": "bus1", "lat": 40.7138, "lon": -74.0050, "vehicle": "bus"},
            {"id": "truck1", "lat": 40.7090, "lon": -74.0100, "vehicle": "truck"},
            {"id": "bike1", "lat": 40.7150, "lon": -74.0070, "vehicle": "bike"},
        ]
        return vehicles

    def get_route_eta(self, start, end):
        try:
            coords = (start[::-1], end[::-1])  # ORS expects (lon, lat)
            routes = self.client.directions(coords)
            summary = routes['routes'][0]['summary']
            distance_km = summary['distance'] / 1000.0
            duration_min = summary['duration'] / 60.0
            return distance_km, duration_min
        except Exception as e:
            print(f"ORS API error: {e}")
            return None, None

    def predict_ml_eta(self, distance_km):
        try:
            v_enc = self.le_vehicle.transform([self.vehicle_type.get()])[0]
            t_enc = self.le_time.transform([self.time_of_day.get()])[0]
            features = np.array([[distance_km, v_enc, self.weight.get(), t_enc]])
            pred = self.model.predict(features)[0]
            return pred
        except Exception as e:
            print(f"ML prediction error: {e}")
            return None

    def update_map(self):
        user_loc = self.geocode_location(self.location_name.get())
        if not user_loc:
            return

        vehicles = self.fetch_vehicle_positions()
        m = folium.Map(location=user_loc, zoom_start=14)

        folium.Marker(user_loc, popup="You", icon=folium.Icon(color="blue")).add_to(m)

        for v in vehicles:
            vehicle_loc = (v['lat'], v['lon'])
            dist_km, route_min = self.get_route_eta(vehicle_loc, user_loc)

            if dist_km is None:
                dist_km = geodesic(vehicle_loc, user_loc).km
                route_min = dist_km / DEFAULT_SPEED_KMH * 60

            ml_eta = self.predict_ml_eta(dist_km)
            eta_str = f"{ml_eta:.1f} min (ML)" if ml_eta else f"{route_min:.1f} min (route)"

            popup_text = (
                f"ID: {v['id']}<br>"
                f"Vehicle: {v['vehicle']}<br>"
                f"Distance: {dist_km:.2f} km<br>"
                f"ETA: {eta_str}"
            )

            folium.Marker(
                vehicle_loc,
                popup=popup_text,
                icon=folium.Icon(
                    color="red",
                    icon="bus" if v['vehicle'] == "bus" else "truck" if v['vehicle'] == "truck" else "bicycle"
                )
            ).add_to(m)

        m.save("live_map.html")
        webbrowser.open("live_map.html")

    def track_loop(self):
        while self.tracking:
            self.update_map()
            interval = self.interval.get()
            for _ in range(interval*10):
                if not self.tracking:
                    break
                self.root.after(100)
                self.root.update_idletasks()

    def start_tracking(self):
        if self.tracking:
            self.tracking = False
            self.start_btn.config(text="Start Tracking")
        else:
            loc = self.geocode_location(self.location_name.get())
            if not loc:
                return
            self.tracking = True
            self.start_btn.config(text="Stop Tracking")
            threading.Thread(target=self.track_loop, daemon=True).start()

if __name__ == "__main__":
    root = tk.Tk()
    app = TransportTrackerApp(root)
    root.mainloop()
