"""
Daily Traffic Prediction Update Script
Automatically runs predictions daily and updates the HTML dashboard
"""

import sys
import os
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
import hopsworks
import holidays
from pathlib import Path

# Add project path
root_dir = Path(__file__).parent
sys.path.append(str(root_dir))

# Import dashboard generator
from generate_dashboard import create_prediction_charts, generate_html_dashboard

class TrafficPredictor:
    """Traffic Flow Predictor - Complete two-stage prediction"""

    def __init__(self, api_key):
        """Initialize and connect to Hopsworks"""
        print("="*60)
        print("Toronto Traffic Flow Prediction System")
        print("="*60)

        self.api_key = api_key
        self.project = None
        self.fs = None
        self.mr = None

        # Model objects
        self.model_visitors = None
        self.model_vehicles = None
        self.model_traffic = None

        # Connect to Hopsworks
        self._connect_hopsworks()

        # Load models
        self._load_models()

    def _connect_hopsworks(self):
        """Connect to Hopsworks"""
        print("\n📡 Connecting to Hopsworks...")
        self.project = hopsworks.login(api_key_value=self.api_key)
        self.fs = self.project.get_feature_store()
        self.mr = self.project.get_model_registry()
        print("✓ Connected successfully")

    def _load_models(self):
        """Load three models from Model Registry"""
        print("\n📦 Loading models from Model Registry...")

        # Model 1: Visitors
        model_visitors_meta = self.mr.get_model(
            name="traffic_flow_visitors_xgboost_model",
            version=4,
        )
        saved_dir = model_visitors_meta.download()
        self.model_visitors = XGBRegressor()
        self.model_visitors.load_model(saved_dir + "/model.json")
        print("✓ Visitors model loaded")

        # Model 2: Vehicles
        model_vehicles_meta = self.mr.get_model(
            name="traffic_flow_vehicles_xgboost_model",
            version=4,
        )
        saved_dir = model_vehicles_meta.download()
        self.model_vehicles = XGBRegressor()
        self.model_vehicles.load_model(saved_dir + "/model.json")
        print("✓ Vehicles model loaded")

        # Model 3: Traffic Flow
        model_traffic_meta = self.mr.get_model(
            name="traffic_flow_xgboost_model",
            version=7,
        )
        saved_dir = model_traffic_meta.download()
        self.model_traffic = XGBRegressor()
        self.model_traffic.load_model(saved_dir + "/model.json")
        print("✓ Traffic flow model loaded")

    def get_weather_forecast(self, days=7):
        """Get weather forecast"""
        print(f"\n🌤️  Fetching {days}-day weather forecast...")

        try:
            from mlfs.airquality import util

            # Toronto coordinates
            city = "Toronto"
            latitude = 43.6532
            longitude = -79.3832

            hourly_df = util.get_hourly_weather_forecast(city, latitude, longitude)

            # Convert to daily data
            hourly_df = hourly_df.set_index('date')
            daily_weather = hourly_df.between_time('11:59', '12:01')
            daily_weather = daily_weather.reset_index()

            print(f"✓ Retrieved weather data for {len(daily_weather)} days")
            return daily_weather

        except Exception as e:
            print(f"❌ Error fetching weather: {e}")
            return None

    def add_holiday_info(self, df):
        """Add holiday information (including weekends)"""
        print("\n📅 Adding holiday information...")

        # Keep Timestamp type
        df['date'] = pd.to_datetime(df['date'])

        # Use holidays library
        ca_holidays = holidays.Canada(prov='ON')

        # Special dates
        special_dates = {
            datetime(2026, 11, 27).date(): 'Black Friday',
            datetime(2026, 12, 24).date(): 'Christmas Eve',
            datetime(2026, 12, 31).date(): 'New Year\'s Eve',
            datetime(2025, 11, 28).date(): 'Black Friday',
            datetime(2025, 12, 24).date(): 'Christmas Eve',
            datetime(2025, 12, 31).date(): 'New Year\'s Eve',
        }

        def is_holiday(date_obj):
            # Check weekends
            if date_obj.dayofweek >= 5:
                return 1
            # Check official holidays
            if date_obj.date() in ca_holidays:
                return 1
            # Check special dates
            if date_obj.date() in special_dates:
                return 1
            return 0

        def get_holiday_name(date_obj):
            # Weekend names
            if date_obj.dayofweek == 5:
                return 'Saturday'
            elif date_obj.dayofweek == 6:
                return 'Sunday'
            # Official holidays
            if date_obj.date() in ca_holidays:
                return ca_holidays.get(date_obj.date())
            if date_obj.date() in special_dates:
                return special_dates[date_obj.date()]
            return ''

        df['holidays'] = df['date'].apply(is_holiday)
        df['holiday_name'] = df['date'].apply(get_holiday_name)

        # Count holidays
        num_holidays = df['holidays'].sum()
        weekends = df[df['holiday_name'].isin(['Saturday', 'Sunday'])].shape[0]
        official = num_holidays - weekends

        if num_holidays > 0:
            print(f"✓ Found {num_holidays} holiday(s)/weekend(s)")
            print(f"  - Weekends: {weekends}")
            print(f"  - Official holidays: {official}")
            for idx, row in df[df['holidays'] == 1].iterrows():
                date_str = row['date'].strftime('%Y-%m-%d')
                print(f"  - {date_str}: {row['holiday_name']}")
        else:
            print("✓ No holidays in forecast period")

        return df

    def predict_stage1(self, weather_data):
        """Stage 1: Predict Visitors and Vehicles"""
        print("\n" + "="*60)
        print("STAGE 1: Predicting Visitors and Vehicles")
        print("="*60)

        # Prepare features
        stage1_features = [
            'holidays',
            'temperature_2m_mean',
            'precipitation_sum',
            'wind_speed_10m_max',
            'wind_direction_10m_dominant'
        ]

        X_stage1 = weather_data[stage1_features]

        # Predict Visitors
        print("\n👥 Predicting visitors...")
        predicted_visitors = self.model_visitors.predict(X_stage1)
        weather_data['predicted_visitors'] = predicted_visitors
        print(f"✓ Predicted: {predicted_visitors.mean():,.0f} avg visitors/day")

        # Predict Vehicles
        print("\n🚗 Predicting vehicles...")
        predicted_vehicles = self.model_vehicles.predict(X_stage1)
        weather_data['predicted_vehicles'] = predicted_vehicles
        print(f"✓ Predicted: {predicted_vehicles.mean():,.0f} avg vehicles/day")

        return weather_data

    def predict_stage2(self, data_with_stage1):
        """Stage 2: Predict Traffic Flow"""
        print("\n" + "="*60)
        print("STAGE 2: Predicting Traffic Flow")
        print("="*60)

        # Prepare data - rename columns to match training feature names
        batch_data_stage2 = data_with_stage1.copy()
        batch_data_stage2['visitors'] = batch_data_stage2['predicted_visitors']
        batch_data_stage2['vehicles'] = batch_data_stage2['predicted_vehicles']

        # Prepare features
        stage2_features = [
            'visitors',
            'holidays',
            'vehicles',
            'temperature_2m_mean',
            'precipitation_sum',
            'wind_speed_10m_max',
            'wind_direction_10m_dominant'
        ]

        X_stage2 = batch_data_stage2[stage2_features]

        # Predict Traffic Flow
        print("\n🚦 Predicting traffic flow...")
        predicted_traffic = self.model_traffic.predict(X_stage2)
        data_with_stage1['predicted_traffic_count'] = predicted_traffic
        print(f"✓ Predicted: {predicted_traffic.mean():,.0f} avg traffic/day")

        return data_with_stage1

    def run_full_prediction(self):
        """Run complete prediction pipeline"""
        print("\n" + "="*60)
        print("Starting Full Prediction Pipeline")
        print("="*60)

        # 1. Get weather forecast
        weather_data = self.get_weather_forecast()
        if weather_data is None:
            return None

        # 2. Add holiday information
        weather_data = self.add_holiday_info(weather_data)

        # 3. Stage 1 prediction
        data_with_stage1 = self.predict_stage1(weather_data)

        # 4. Stage 2 prediction
        final_predictions = self.predict_stage2(data_with_stage1)

        print("\n" + "="*60)
        print("Prediction Pipeline Complete!")
        print("="*60)

        return final_predictions

def main():
    """Main function - runs daily"""

    print("\n" + "="*70)
    print(f"Daily Update Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)

    # ========== Configuration ==========
    # Read API key from environment variable, or set directly
    API_KEY = os.environ.get('HOPSWORKS_API_KEY', 'YOUR_API_KEY_HERE')
    OUTPUT_DIR = './dashboard'
    HTML_FILE = os.path.join(OUTPUT_DIR, 'traffic_dashboard.html')

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    try:
        # ========== 1. Run prediction ==========
        predictor = TrafficPredictor(API_KEY)
        batch_data = predictor.run_full_prediction()

        if batch_data is None:
            print("❌ Prediction failed!")
            return

        # ========== 2. Generate charts ==========
        print("\n📊 Generating charts...")
        charts = create_prediction_charts(batch_data, output_dir=OUTPUT_DIR)
        print("✓ Charts generated")

        # ========== 3. Generate HTML ==========
        print("\n📄 Generating HTML dashboard...")
        html_file = generate_html_dashboard(batch_data, charts, output_file=HTML_FILE)
        print(f"✓ HTML dashboard saved: {html_file}")

        # ========== 4. Save prediction data ==========
        csv_file = os.path.join(OUTPUT_DIR, f'predictions_{datetime.now().strftime("%Y%m%d")}.csv')
        batch_data.to_csv(csv_file, index=False)
        print(f"✓ Predictions saved to CSV: {csv_file}")

        # ========== 5. Display statistics ==========
        print("\n" + "="*70)
        print("Prediction Statistics")
        print("="*70)
        print(f"Average Visitors:  {batch_data['predicted_visitors'].mean():>10,.0f}")
        print(f"Average Vehicles:  {batch_data['predicted_vehicles'].mean():>10,.0f}")
        print(f"Average Traffic:   {batch_data['predicted_traffic_count'].mean():>10,.0f}")
        print(f"Peak Traffic Day:  {batch_data.loc[batch_data['predicted_traffic_count'].idxmax(), 'date']}")
        print("="*70)

        print(f"\n✅ Daily update completed successfully!")
        print(f"📁 Dashboard location: {os.path.abspath(HTML_FILE)}")

    except Exception as e:
        print(f"\n❌ Error during daily update: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()