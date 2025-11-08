import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from pathlib import Path

print("Generating synthetic data for demo...")

# Create directories
Path('data/raw/traffic').mkdir(parents=True, exist_ok=True)
Path('data/raw/weather').mkdir(parents=True, exist_ok=True)

# Routes
routes = [
    {'route_id': 'R001', 'origin': 'Okponglo Junction', 'destination': 'Legon Campus', 
     'distance_m': 3200, 'base_speed': 35, 'start_lat': 5.6515, 'start_lng': -0.1851, 
     'end_lat': 5.6691, 'end_lng': -0.1778},
    {'route_id': 'R002', 'origin': 'Atomic Junction', 'destination': 'Madina',
     'distance_m': 4500, 'base_speed': 32, 'start_lat': 5.6691, 'start_lng': -0.1778, 
     'end_lat': 5.6850, 'end_lng': -0.1650},
    {'route_id': 'R003', 'origin': '37 Military Hospital', 'destination': 'Ako Adjei',
     'distance_m': 5800, 'base_speed': 40, 'start_lat': 5.5767, 'start_lng': -0.1863, 
     'end_lat': 5.6037, 'end_lng': -0.1870},
    {'route_id': 'R004', 'origin': 'Airport Junction', 'destination': 'Tetteh Quarshie',
     'distance_m': 3800, 'base_speed': 45, 'start_lat': 5.6037, 'start_lng': -0.1870, 
     'end_lat': 5.6422, 'end_lng': -0.1700},
    {'route_id': 'R005', 'origin': 'Shiashie', 'destination': 'East Legon',
     'distance_m': 4200, 'base_speed': 38, 'start_lat': 5.6422, 'start_lng': -0.1700, 
     'end_lat': 5.6550, 'end_lng': -0.1600}
]

# Generate 60 days of traffic data
start_date = datetime.now() - timedelta(days=60)
data = []

for route in routes:
    current = start_date.replace(hour=5, minute=0, second=0, microsecond=0)
    end = datetime.now()
    
    while current < end:
        if 5 <= current.hour < 22:
            hour = current.hour
            is_weekend = current.weekday() >= 5
            
            # Realistic traffic patterns
            if 6 <= hour < 9 and not is_weekend:
                speed_factor = 0.55
            elif 17 <= hour < 20 and not is_weekend:
                speed_factor = 0.50
            else:
                speed_factor = 0.85
            
            speed = route['base_speed'] * speed_factor * random.uniform(0.9, 1.1)
            speed = max(5, min(speed, 80))
            
            data.append({
                'timestamp': current,
                'route_id': route['route_id'],
                'origin': route['origin'],
                'destination': route['destination'],
                'distance_m': route['distance_m'],
                'speed_kmh': speed,
                'start_lat': route['start_lat'],
                'start_lng': route['start_lng'],
                'end_lat': route['end_lat'],
                'end_lng': route['end_lng'],
                'hour': hour,
                'day_of_week': current.weekday(),
                'is_weekend': is_weekend
            })
        
        current += timedelta(minutes=5)

df = pd.DataFrame(data)
df.to_csv('data/raw/traffic/traffic_data_demo.csv', index=False)
print(f"✓ Generated {len(df):,} traffic records")

# Generate weather data
weather_data = []
current = start_date
while current < datetime.now():
    temp = 28 + random.uniform(-3, 6)
    rain = random.random() < 0.1
    
    weather_data.append({
        'timestamp': current,
        'temperature_c': round(temp, 1),
        'humidity_pct': round(random.uniform(65, 85), 0),
        'rain_1h_mm': round(random.uniform(0.5, 20), 1) if rain else 0,
        'is_raining': 1 if rain else 0
    })
    current += timedelta(minutes=30)

df_weather = pd.DataFrame(weather_data)
df_weather.to_csv('data/raw/weather/weather_data_demo.csv', index=False)
print(f"✓ Generated {len(df_weather):,} weather records")
print("\n✅ Data generation complete!")