# Toronto Traffic Flow Prediction System

A scalable machine learning pipeline for predicting daily traffic flow at Toronto parks using hierarchical XGBoost models, weather forecasts, and holiday information.


## 📊 Project Overview

This project implements an end-to-end machine learning system that predicts traffic flow at Toronto parks up to 7 days in advance. The system uses a two-stage hierarchical prediction approach:

1. **Stage 1**: Predict visitor counts and vehicle counts based on weather and holidays
2. **Stage 2**: Predict overall traffic flow using Stage 1 predictions plus additional features

### Key Features

- **Hierarchical Prediction Architecture**: Two-stage modeling for improved accuracy
- **Automated Daily Updates**: Scheduled predictions with dashboard generation
- **Feature Engineering Pipeline**: Weather data integration, holiday detection, temporal features
- **Interactive Dashboard**: HTML visualization with trend charts and statistics
- **Scalable Infrastructure**: Built on Hopsworks Feature Store and Model Registry

## 🏗️ System Architecture

```
┌─────────────────┐
│  Weather API    │
│  (Open-Meteo)   │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│     Feature Engineering Pipeline        │
│  - Holiday detection (Canada holidays)  │
│  - Temporal features (day, month, etc.) │
│  - Weather aggregation                  │
└────────┬────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────┐
│         Stage 1 Prediction               │
│  ┌──────────────┐  ┌─────────────────┐  │
│  │   Visitors   │  │    Vehicles     │  │
│  │  XGBoost     │  │    XGBoost      │  │
│  └──────────────┘  └─────────────────┘  │
└────────┬─────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────┐
│         Stage 2 Prediction               │
│  ┌──────────────────────────────────┐   │
│  │      Traffic Flow                │   │
│  │      XGBoost Model               │   │
│  │  (uses Stage 1 predictions)      │   │
│  └──────────────────────────────────┘   │
└────────┬─────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────┐
│       Dashboard Generation               │
│  - Interactive HTML                      │
│  - Trend visualizations                  │
│  - Statistical summaries                 │
└──────────────────────────────────────────┘
```





## 📈 Dashboard Features

The generated HTML dashboard includes:

- **7-Day Forecast Table**: Detailed predictions with weather conditions
- **Traffic Trend Chart**: Visual representation of predicted traffic flow
- **Visitors vs Vehicles Chart**: Comparison of the two Stage 1 predictions
- **Statistics Summary**: Average, peak, and holiday information
- **Automatic Updates**: Timestamp showing last update time

## 🔄 Update Schedule

The dashboard updates automatically:
- **Frequency**: Daily
- **Time**: 8:00 AM UTC (via GitHub Actions)
- **Data**: 7-day rolling forecast

### Data Sources

1. **Historical Traffic Data**: Toronto Open Data Portal

   - Daily visitor(entering Canada) counts
   - Vehicle(entering Canada) counts
   - Traffic flow statistics

Data source:https://www150.statcan.gc.ca/n1/pub/71-607-x/71-607-x2022018-eng.htm

2. **Weather Data**: Open-Meteo API
   - 7-day hourly forecasts
   - Historical weather data
   - No API key required
   
   


3. **Holiday Data**: Python `holidays` library
   - Canadian federal holidays
   - Ontario provincial holidays
   - Custom special dates (Black Friday, etc.)

### Technology Stack

- **ML Framework**: XGBoost 2.0+
- **Feature Store**: Hopsworks Feature Store
- **Model Registry**: Hopsworks Model Registry
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib
- **Scheduling**: Cron (Linux/Mac), Task Scheduler (Windows), Docker







## 🙏 Acknowledgments

- Toronto Open Data Portal for traffic datasets
- Open-Meteo for free weather API
- Hopsworks for Feature Store and Model Registry
- XGBoost development team

## 📈 Dashboard
After running the ./notebooks/traffic_flow/traffic_flow_batch_inference.ipynb, you can get the html in ./notebooks/traffic_flow/dashboard.


## 🔗 Links

- [Hopsworks Documentation](https://docs.hopsworks.ai/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [Open-Meteo API](https://open-meteo.com/)
- [Toronto Open Data](https://open.toronto.ca/)

Open the dashboard:
```bash
# On Windows
start dashboard/traffic_dashboard.html

# On macOS
open dashboard/traffic_dashboard.html

# On Linux
xdg-open dashboard/traffic_dashboard.html
```


**Last Updated**: January 2026  
**Project Status**: Active Development
