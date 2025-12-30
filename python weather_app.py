# -*- coding: utf-8 -*-
"""Weather Prediction Application"""

import requests
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
from datetime import datetime, timedelta
import pytz
import os

class WeatherApp:
    def __init__(self):
        self.API_key = '40793432b6cf008a566a7b4de14e3212'
        self.BASE_URL = 'https://api.openweathermap.org/data/2.5/'
        self.historical_data_path = 'weather.csv'
        self.timezone = pytz.timezone('Asia/Kolkata')
        
        # Check if data file exists
        if not os.path.exists(self.historical_data_path):
            print(f"Error: Data file '{self.historical_data_path}' not found.")
            print("Please ensure the weather.csv file is in the same directory as this script.")
            exit()
    
    def clear_screen(self):
        """Clear the console screen"""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def display_header(self):
        """Display application header"""
        self.clear_screen()
        print("=" * 50)
        print("WEATHER PREDICTION APPLICATION".center(50))
        print("=" * 50)
        print("\n")
    
    def current_weather(self, city):
        """Get current weather data from API"""
        try:
            url = f"{self.BASE_URL}weather?q={city}&appid={self.API_key}&units=metric"
            response = requests.get(url)
            response.raise_for_status()  # Raise exception for HTTP errors
            data = response.json()
            
            wind_data = data.get('wind', {})
            return {
                'city': data['name'],
                'current_temp': round(data['main']['temp']),
                'current_humidity': round(data['main']['humidity']),
                'feels_like': round(data['main']['feels_like']),
                'temp_min': round(data['main']['temp_min']),
                'temp_max': round(data['main']['temp_max']),
                'humidity': round(data['main']['humidity']),
                'description': (data['weather'][0]['description']),
                'country': (data['sys']['country']),
                'wind_deg': wind_data.get('deg', 0),
                'Wind_Gust_Speed': wind_data.get('gust', 0),
                'pressure': data['main']['pressure']
            }
        except Exception as e:
            print(f"\nError fetching weather data: {e}")
            return None
    
    def read_historical_data(self, file_path):
        """Read and clean historical weather data"""
        try:
            df = pd.read_csv(file_path)
            df = df.dropna()
            df = df.drop_duplicates()
            return df
        except Exception as e:
            print(f"\nError reading historical data: {e}")
            return None
    
    def prepare_data(self, data):
        """Prepare data for machine learning models"""
        try:
            le = LabelEncoder()
            if 'WindGustDir' in data.columns:
                data['WindGustDir'] = data['WindGustDir'].astype(str)
                data['WindGustDir'] = le.fit_transform(data['WindGustDir'])
            if 'RainTomorrow' in data.columns:
                data['RainTomorrow'] = data['RainTomorrow'].astype(str)
                data['RainTomorrow'] = le.fit_transform(data['RainTomorrow'])

            features = ['MinTemp','MaxTemp','WindGustDir','WindGustSpeed','Humidity','Pressure','Temp','RainTomorrow']
            existing_features = [f for f in features if f in data.columns]
            X = data[existing_features]
            y = data['RainTomorrow'] if 'RainTomorrow' in data.columns else pd.Series()
            
            return X, y, le
        except Exception as e:
            print(f"\nError preparing data: {e}")
            return None, None, None
    
    def train_rain_model(self, X, y):
        """Train rain prediction model"""
        try:
            if not y.empty and len(y.unique()) > 1:
                if len(X) == len(y):
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                    model = RandomForestClassifier(n_estimators=100, random_state=42)
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    print("\nRain Model Training Complete")
                    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
                    return model
                else:
                    print("\nMismatch in number of samples between features and target.")
                    return None
            else:
                print("\nNot enough data or variety in 'RainTomorrow' to train rain model.")
                return None
        except Exception as e:
            print(f"\nError training rain model: {e}")
            return None
    
    def create_features(self, data, feature):
        """Create features for regression models"""
        try:
            X = []
            y = []
            if feature in data.columns and len(data) > 1:
                for i in range(len(data) - 1):
                    X.append(data[feature].iloc[i])
                    y.append(data[feature].iloc[i + 1])

                X = np.array(X).reshape(-1, 1)
                y = np.array(y)
            else:
                print(f"\nFeature '{feature}' not found or insufficient data.")
                X = np.array([])
                y = np.array([])
            return X, y
        except Exception as e:
            print(f"\nError creating features: {e}")
            return np.array([]), np.array([])
    
    def train_regression_model(self, X, y):
        """Train regression model"""
        try:
            if X.size > 0 and y.size > 0:
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model.fit(X, y)
                return model
            else:
                print("\nNot enough data to train regression model.")
                return None
        except Exception as e:
            print(f"\nError training regression model: {e}")
            return None
    
    def predict_feature(self, model, current_value):
        """Make predictions using regression model"""
        try:
            predictions = [current_value]
            if model is not None:
                for i in range(5):
                    next_value = model.predict(np.array([predictions[-1]]).reshape(-1, 1))
                    predictions.append(next_value[0])
                return predictions[1:]
            else:
                print("\nRegression model not trained, cannot predict.")
                return []
        except Exception as e:
            print(f"\nError making predictions: {e}")
            return []
    
    def get_compass_direction(self, wind_deg):
        """Convert wind degree to compass direction"""
        compass_points = [
            ('N', 348.75, 360), ('N', 0, 11.25),
            ('NNE', 11.25, 33.75), ('NE', 33.75, 56.25),
            ('ENE', 56.25, 78.75), ('E', 78.75, 101.25),
            ('ESE', 101.25, 123.75), ('SE', 123.75, 146.25),
            ('SSE', 146.25, 168.75), ('S', 168.75, 191.25),
            ('SSW', 191.25, 213.75), ('SW', 213.75, 236.25),
            ('WSW', 236.25, 258.75), ('W', 258.75, 281.25),
            ('WNW', 281.25, 303.75), ('NW', 303.75, 326.25),
            ('NNW', 326.25, 348.75)
        ]
        
        compass_direction = 'N'  # Default value
        for point, start, end in compass_points:
            if (start <= wind_deg < end) or (start > end and (wind_deg >= start or wind_deg < end)):
                compass_direction = point
                break
        return compass_direction
    
    def display_weather(self, city):
        """Display weather information for a city"""
        try:
            self.display_header()
            print(f"Fetching weather data for {city}...\n")
            
            # Get current weather
            current_weather_data = self.current_weather(city)
            if current_weather_data is None:
                print("Failed to fetch current weather data.")
                return
            
            # Load and prepare historical data
            historical_data = self.read_historical_data(self.historical_data_path)
            if historical_data is None:
                print("Failed to load historical data.")
                return
            
            X, y, le = self.prepare_data(historical_data)
            rain_model = self.train_rain_model(X, y)
            
            # Process wind direction
            wind_deg = current_weather_data.get('wind_deg', 0) % 360
            compass_direction = self.get_compass_direction(wind_deg)
            compass_labels = [point for point, _, _ in [
                ('N', 348.75, 360), ('N', 0, 11.25),
                ('NNE', 11.25, 33.75), ('NE', 33.75, 56.25),
                ('ENE', 56.25, 78.75), ('E', 78.75, 101.25),
                ('ESE', 101.25, 123.75), ('SE', 123.75, 146.25),
                ('SSE', 146.25, 168.75), ('S', 168.75, 191.25),
                ('SSW', 191.25, 213.75), ('SW', 213.75, 236.25),
                ('WSW', 236.25, 258.75), ('W', 258.75, 281.25),
                ('WNW', 281.25, 303.75), ('NW', 303.75, 326.25),
                ('NNW', 326.25, 348.75)
            ]]
            
            le = LabelEncoder()
            le.fit(compass_labels)
            compass_direction_encoded = le.transform([compass_direction])[0]
            
            # Prepare current data for prediction
            current_data = {
                'MinTemp': current_weather_data.get('temp_min'),
                'MaxTemp': current_weather_data.get('temp_max'),
                'WindGustDir': compass_direction_encoded,
                'WindGustSpeed': current_weather_data.get('Wind_Gust_Speed', 0),
                'Humidity': current_weather_data.get('humidity'),
                'Pressure': current_weather_data.get('pressure'),
                'Temp': current_weather_data.get('current_temp'),
                'RainTomorrow': 0
            }
            
            current_df = pd.DataFrame([current_data])
            
            # Predict rain
            rain_prediction = 'N/A'
            if rain_model is not None and all(col in current_df.columns for col in X.columns):
                try:
                    rain_prediction = rain_model.predict(current_df[X.columns])[0]
                    rain_prediction = 'Yes' if rain_prediction == 1 else 'No'
                except Exception as e:
                    print(f"Error during rain prediction: {e}")
                    rain_prediction = "Prediction failed"
            else:
                rain_prediction = "Not available"
            
            # Train temperature and humidity models
            X_temp, y_temp = self.create_features(historical_data, 'Temp')
            X_hum, y_hum = self.create_features(historical_data, 'Humidity')
            temp_model = self.train_regression_model(X_temp, y_temp)
            hum_model = self.train_regression_model(X_hum, y_hum)
            
            # Make predictions
            future_temp = self.predict_feature(temp_model, current_weather_data.get('temp_min')) if temp_model else []
            future_humidity = self.predict_feature(hum_model, current_weather_data.get('humidity')) if hum_model else []
            
            # Generate future times
            current_time = datetime.now(self.timezone)
            next_hour = current_time + timedelta(hours=1)
            future_times = [(next_hour + timedelta(hours=i)).strftime("%H:00") for i in range(5)]
            
            # Display results
            print("\n" + "=" * 50)
            print(f"WEATHER REPORT FOR {current_weather_data['city'].upper()}, {current_weather_data['country']}")
            print("=" * 50)
            
            print("\nCURRENT CONDITIONS:")
            print(f"• Temperature: {current_weather_data['current_temp']}°C (Feels like {current_weather_data['feels_like']}°C)")
            print(f"• Min/Max: {current_weather_data['temp_min']}°C / {current_weather_data['temp_max']}°C")
            print(f"• Humidity: {current_weather_data['humidity']}%")
            print(f"• Pressure: {current_weather_data['pressure']} hPa")
            print(f"• Wind: {compass_direction} at {current_weather_data.get('Wind_Gust_Speed', 0)} km/h")
            print(f"• Conditions: {current_weather_data['description'].capitalize()}")
            print(f"• Rain Prediction for Tomorrow: {rain_prediction}")
            
            print("\n" + "-" * 50)
            print("HOURLY FORECAST")
            print("-" * 50)
            
            if future_temp:
                print("\nTEMPERATURE:")
                for time, temp in zip(future_times, future_temp):
                    print(f"• {time}: {round(temp, 1)}°C")
            else:
                print("\nTemperature forecast not available.")
            
            if future_humidity:
                print("\nHUMIDITY:")
                for time, humidity in zip(future_times, future_humidity):
                    print(f"• {time}: {round(humidity, 1)}%")
            else:
                print("\nHumidity forecast not available.")
            
            print("\n" + "=" * 50)
            
        except Exception as e:
            print(f"\nAn error occurred: {e}")
    
    def run(self):
        """Main application loop"""
        while True:
            self.display_header()
            print("1. Check Weather for a City")
            print("2. Exit")
            
            choice = input("\nEnter your choice (1-2): ")
            
            if choice == '1':
                city = input("\nEnter city name: ").strip()
                if city:
                    self.display_weather(city)
                    input("\nPress Enter to continue...")
                else:
                    print("\nPlease enter a valid city name.")
            elif choice == '2':
                print("\nThank you for using the Weather Prediction Application!")
                break
            else:
                print("\nInvalid choice. Please try again.")
                input("Press Enter to continue...")

if __name__ == "__main__":
    app = WeatherApp()
    app.run()