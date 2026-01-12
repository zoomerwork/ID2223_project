"""
Daily Traffic Prediction Update Script
每天自动运行预测并更新HTML仪表板
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

# 添加项目路径
root_dir = Path(__file__).parent
sys.path.append(str(root_dir))

# 导入仪表板生成器
from generate_dashboard import create_prediction_charts, generate_html_dashboard

class TrafficPredictor:
    """交通流量预测器 - 完整的两阶段预测"""
    
    def __init__(self, api_key):
        """初始化并连接Hopsworks"""
        print("="*60)
        print("Toronto Traffic Flow Prediction System")
        print("="*60)
        
        self.api_key = api_key
        self.project = None
        self.fs = None
        self.mr = None
        
        # 模型对象
        self.model_visitors = None
        self.model_vehicles = None
        self.model_traffic = None
        
        # 连接Hopsworks
        self._connect_hopsworks()
        
        # 加载模型
        self._load_models()
    
    def _connect_hopsworks(self):
        """连接到Hopsworks"""
        print("\n📡 Connecting to Hopsworks...")
        self.project = hopsworks.login(api_key_value=self.api_key)
        self.fs = self.project.get_feature_store()
        self.mr = self.project.get_model_registry()
        print("✓ Connected successfully")
    
    def _load_models(self):
        """从Model Registry加载三个模型"""
        print("\n📦 Loading models from Model Registry...")
        
        # 模型1: Visitors
        model_visitors_meta = self.mr.get_model(
            name="visitor_prediction_xgboost_model",
            version=1,
        )
        saved_dir = model_visitors_meta.download()
        self.model_visitors = XGBRegressor()
        self.model_visitors.load_model(saved_dir + "/model.json")
        print("✓ Visitors model loaded")
        
        # 模型2: Vehicles
        model_vehicles_meta = self.mr.get_model(
            name="vehicle_prediction_xgboost_model",
            version=1,
        )
        saved_dir = model_vehicles_meta.download()
        self.model_vehicles = XGBRegressor()
        self.model_vehicles.load_model(saved_dir + "/model.json")
        print("✓ Vehicles model loaded")
        
        # 模型3: Traffic Flow
        model_traffic_meta = self.mr.get_model(
            name="traffic_flow_xgboost_model",
            version=1,
        )
        saved_dir = model_traffic_meta.download()
        self.model_traffic = XGBRegressor()
        self.model_traffic.load_model(saved_dir + "/model.json")
        print("✓ Traffic flow model loaded")
    
    def get_weather_forecast(self, days=7):
        """获取天气预报"""
        print(f"\n🌤️  Fetching {days}-day weather forecast...")
        
        try:
            from mlfs.airquality import util
            
            # Toronto坐标
            city = "Toronto"
            latitude = 43.6532
            longitude = -79.3832
            
            hourly_df = util.get_hourly_weather_forecast(city, latitude, longitude)
            
            # 转换为每日数据
            hourly_df = hourly_df.set_index('date')
            daily_weather = hourly_df.between_time('11:59', '12:01')
            daily_weather = daily_weather.reset_index()
            
            print(f"✓ Retrieved weather data for {len(daily_weather)} days")
            return daily_weather
            
        except Exception as e:
            print(f"❌ Error fetching weather: {e}")
            return None

    def add_holiday_info(self, df):
        """添加节假日信息（包含周末）"""
        print("\n📅 Adding holiday information...")

        # 保持Timestamp类型
        df['date'] = pd.to_datetime(df['date'])

        # 使用holidays库
        ca_holidays = holidays.Canada(prov='ON')

        # 特殊日期
        special_dates = {
            datetime(2026, 11, 27).date(): 'Black Friday',
            datetime(2026, 12, 24).date(): 'Christmas Eve',
            datetime(2026, 12, 31).date(): 'New Year\'s Eve',
            datetime(2025, 11, 28).date(): 'Black Friday',
            datetime(2025, 12, 24).date(): 'Christmas Eve',
            datetime(2025, 12, 31).date(): 'New Year\'s Eve',
        }

        def is_holiday(date_obj):
            # 检查周末
            if date_obj.dayofweek >= 5:
                return 1
            # 检查官方节假日
            if date_obj.date() in ca_holidays:
                return 1
            # 检查特殊日期
            if date_obj.date() in special_dates:
                return 1
            return 0

        def get_holiday_name(date_obj):
            # 周末名称
            if date_obj.dayofweek == 5:
                return 'Saturday'
            elif date_obj.dayofweek == 6:
                return 'Sunday'
            # 官方节假日
            if date_obj.date() in ca_holidays:
                return ca_holidays.get(date_obj.date())
            if date_obj.date() in special_dates:
                return special_dates[date_obj.date()]
            return ''

        df['holidays'] = df['date'].apply(is_holiday)
        df['holiday_name'] = df['date'].apply(get_holiday_name)

        # 统计节假日
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
        """阶段1: 预测Visitors和Vehicles"""
        print("\n" + "="*60)
        print("STAGE 1: Predicting Visitors and Vehicles")
        print("="*60)
        
        # 准备特征
        stage1_features = [
            'holidays',
            'temperature_2m_mean',
            'precipitation_sum',
            'wind_speed_10m_max',
            'wind_direction_10m_dominant'
        ]
        
        X_stage1 = weather_data[stage1_features]
        
        # 预测Visitors
        print("\n👥 Predicting visitors...")
        predicted_visitors = self.model_visitors.predict(X_stage1)
        weather_data['predicted_visitors'] = predicted_visitors
        print(f"✓ Predicted: {predicted_visitors.mean():,.0f} avg visitors/day")
        
        # 预测Vehicles
        print("\n🚗 Predicting vehicles...")
        predicted_vehicles = self.model_vehicles.predict(X_stage1)
        weather_data['predicted_vehicles'] = predicted_vehicles
        print(f"✓ Predicted: {predicted_vehicles.mean():,.0f} avg vehicles/day")
        
        return weather_data
    
    def predict_stage2(self, data_with_stage1):
        """阶段2: 预测Traffic Flow"""
        print("\n" + "="*60)
        print("STAGE 2: Predicting Traffic Flow")
        print("="*60)
        
        # 准备数据 - 重命名列以匹配训练时的特征名
        batch_data_stage2 = data_with_stage1.copy()
        batch_data_stage2['visitors'] = batch_data_stage2['predicted_visitors']
        batch_data_stage2['vehicles'] = batch_data_stage2['predicted_vehicles']
        
        # 准备特征
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
        
        # 预测Traffic Flow
        print("\n🚦 Predicting traffic flow...")
        predicted_traffic = self.model_traffic.predict(X_stage2)
        data_with_stage1['predicted_traffic_count'] = predicted_traffic
        print(f"✓ Predicted: {predicted_traffic.mean():,.0f} avg traffic/day")
        
        return data_with_stage1
    
    def run_full_prediction(self):
        """运行完整的预测流程"""
        print("\n" + "="*60)
        print("Starting Full Prediction Pipeline")
        print("="*60)
        
        # 1. 获取天气预报
        weather_data = self.get_weather_forecast()
        if weather_data is None:
            return None
        
        # 2. 添加节假日信息
        weather_data = self.add_holiday_info(weather_data)
        
        # 3. 阶段1预测
        data_with_stage1 = self.predict_stage1(weather_data)
        
        # 4. 阶段2预测
        final_predictions = self.predict_stage2(data_with_stage1)
        
        print("\n" + "="*60)
        print("Prediction Pipeline Complete!")
        print("="*60)
        
        return final_predictions

def main():
    """主函数 - 每日运行"""
    
    print("\n" + "="*70)
    print(f"Daily Update Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    # ========== 配置 ==========
    # 从环境变量读取API key，或直接设置
    API_KEY = os.environ.get('HOPSWORKS_API_KEY', 'YOUR_API_KEY_HERE')
    OUTPUT_DIR = './dashboard'
    HTML_FILE = os.path.join(OUTPUT_DIR, 'traffic_dashboard.html')
    
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    try:
        # ========== 1. 运行预测 ==========
        predictor = TrafficPredictor(API_KEY)
        batch_data = predictor.run_full_prediction()
        
        if batch_data is None:
            print("❌ Prediction failed!")
            return
        
        # ========== 2. 生成图表 ==========
        print("\n📊 Generating charts...")
        charts = create_prediction_charts(batch_data, output_dir=OUTPUT_DIR)
        print("✓ Charts generated")
        
        # ========== 3. 生成HTML ==========
        print("\n📄 Generating HTML dashboard...")
        html_file = generate_html_dashboard(batch_data, charts, output_file=HTML_FILE)
        print(f"✓ HTML dashboard saved: {html_file}")
        
        # ========== 4. 保存预测数据 ==========
        csv_file = os.path.join(OUTPUT_DIR, f'predictions_{datetime.now().strftime("%Y%m%d")}.csv')
        batch_data.to_csv(csv_file, index=False)
        print(f"✓ Predictions saved to CSV: {csv_file}")
        
        # ========== 5. 显示统计 ==========
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
