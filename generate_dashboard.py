"""
Traffic Flow Prediction Dashboard Generator
每天自动更新预测并生成HTML报告
"""

import os
import json
import base64
from datetime import datetime, date
from io import BytesIO
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端

def save_figure_to_base64(fig):
    """将matplotlib图表转换为base64编码的字符串"""
    buffer = BytesIO()
    fig.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode()
    buffer.close()
    plt.close(fig)
    return image_base64

def create_prediction_charts(batch_data, output_dir='./dashboard'):
    """
    创建三个预测图表并保存为base64
    
    返回:
        dict: 包含三个图表的base64编码
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert date for plotting
    plot_dates = pd.to_datetime(batch_data['date'])
    
    # Define colors
    colors = {
        'visitors': '#2E86AB',
        'vehicles': '#A23B72', 
        'traffic': '#F18F01'
    }
    
    charts = {}
    
    # ========== Chart 1: Predicted Visitors ==========
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    visitors_data = batch_data['predicted_visitors']
    
    ax1.plot(plot_dates, visitors_data, 
             marker='o', linewidth=2.5, markersize=8, 
             color=colors['visitors'], label='Predicted Visitors')
    ax1.set_title('Predicted Visitors - Next 7 Days', 
                  fontsize=16, fontweight='bold', pad=20)
    ax1.set_ylabel('Visitor Count', fontsize=12)
    ax1.set_xlabel('Date', fontsize=12)
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # Fill area from min value
    y_min_visitors = visitors_data.min() * 0.95
    ax1.fill_between(plot_dates, y_min_visitors, visitors_data, 
                     alpha=0.3, color=colors['visitors'])
    ax1.set_ylim(bottom=y_min_visitors)
    
    ax1.legend(loc='upper right', framealpha=0.9)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    charts['visitors'] = save_figure_to_base64(fig1)
    
    # ========== Chart 2: Predicted Vehicles ==========
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    vehicles_data = batch_data['predicted_vehicles']
    
    ax2.plot(plot_dates, vehicles_data, 
             marker='s', linewidth=2.5, markersize=8, 
             color=colors['vehicles'], label='Predicted Vehicles')
    ax2.set_title('Predicted Vehicles - Next 7 Days', 
                  fontsize=16, fontweight='bold', pad=20)
    ax2.set_ylabel('Vehicle Count', fontsize=12)
    ax2.set_xlabel('Date', fontsize=12)
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    y_min_vehicles = vehicles_data.min() * 0.95
    ax2.fill_between(plot_dates, y_min_vehicles, vehicles_data, 
                     alpha=0.3, color=colors['vehicles'])
    ax2.set_ylim(bottom=y_min_vehicles)
    
    ax2.legend(loc='upper right', framealpha=0.9)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    charts['vehicles'] = save_figure_to_base64(fig2)
    
    # ========== Chart 3: Predicted Traffic Flow ==========
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    traffic_data = batch_data['predicted_traffic_count']
    
    ax3.plot(plot_dates, traffic_data, 
             marker='D', linewidth=2.5, markersize=8, 
             color=colors['traffic'], label='Predicted Traffic')
    ax3.set_title('Predicted Traffic Flow - Next 7 Days', 
                  fontsize=16, fontweight='bold', pad=20)
    ax3.set_ylabel('Traffic Count', fontsize=12)
    ax3.set_xlabel('Date', fontsize=12)
    ax3.grid(True, alpha=0.3, linestyle='--')
    
    y_min_traffic = traffic_data.min() * 0.95
    ax3.fill_between(plot_dates, y_min_traffic, traffic_data, 
                     alpha=0.3, color=colors['traffic'])
    ax3.set_ylim(bottom=y_min_traffic)
    
    ax3.legend(loc='upper right', framealpha=0.9)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    charts['traffic'] = save_figure_to_base64(fig3)
    
    return charts

def generate_html_dashboard(batch_data, charts, output_file='traffic_dashboard.html'):
    """
    生成完整的HTML仪表板
    
    参数:
        batch_data: DataFrame 包含预测数据
        charts: dict 包含base64编码的图表
        output_file: str 输出HTML文件名
    """
    
    # 计算统计数据
    stats = {
        'avg_visitors': batch_data['predicted_visitors'].mean(),
        'avg_vehicles': batch_data['predicted_vehicles'].mean(),
        'avg_traffic': batch_data['predicted_traffic_count'].mean(),
        'max_traffic_date': batch_data.loc[batch_data['predicted_traffic_count'].idxmax(), 'date'],
        'max_traffic_value': batch_data['predicted_traffic_count'].max(),
        'min_traffic_date': batch_data.loc[batch_data['predicted_traffic_count'].idxmin(), 'date'],
        'min_traffic_value': batch_data['predicted_traffic_count'].min(),
    }
    
    # 生成每日预测表格的HTML
    table_rows = ""
    for idx, row in batch_data.iterrows():
        holiday_badge = '🎉 Holiday' if row.get('holidays', 0) == 1 else ''
        holiday_class = 'holiday-row' if row.get('holidays', 0) == 1 else ''
        
        table_rows += f"""
        <tr class="{holiday_class}">
            <td>{row['date']}</td>
            <td>{holiday_badge}</td>
            <td>{row['predicted_visitors']:,.0f}</td>
            <td>{row['predicted_vehicles']:,.0f}</td>
            <td>{row['predicted_traffic_count']:,.0f}</td>
            <td>{row.get('temperature_2m_mean', 0):.1f}°C</td>
        </tr>
        """
    
    # HTML模板
    html_template = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Toronto Traffic Flow Predictions Dashboard</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }}
        
        .header {{
            background: linear-gradient(135deg, #2E86AB 0%, #1a4d6b 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }}
        
        .header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
        }}
        
        .header .subtitle {{
            font-size: 1.2em;
            opacity: 0.9;
        }}
        
        .header .update-time {{
            margin-top: 15px;
            font-size: 0.9em;
            opacity: 0.8;
        }}
        
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            padding: 40px;
            background: #f8f9fa;
        }}
        
        .stat-card {{
            background: white;
            padding: 25px;
            border-radius: 15px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            text-align: center;
            transition: transform 0.3s, box-shadow 0.3s;
        }}
        
        .stat-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 8px 12px rgba(0,0,0,0.15);
        }}
        
        .stat-card .icon {{
            font-size: 3em;
            margin-bottom: 10px;
        }}
        
        .stat-card .value {{
            font-size: 2em;
            font-weight: bold;
            margin: 10px 0;
        }}
        
        .stat-card .label {{
            color: #666;
            font-size: 1em;
        }}
        
        .stat-card.visitors .value {{ color: #2E86AB; }}
        .stat-card.vehicles .value {{ color: #A23B72; }}
        .stat-card.traffic .value {{ color: #F18F01; }}
        
        .charts-section {{
            padding: 40px;
        }}
        
        .chart-container {{
            margin-bottom: 40px;
            background: white;
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        
        .chart-container h2 {{
            color: #333;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 3px solid #2E86AB;
        }}
        
        .chart-container img {{
            width: 100%;
            height: auto;
            border-radius: 10px;
        }}
        
        .table-section {{
            padding: 40px;
            background: #f8f9fa;
        }}
        
        .table-section h2 {{
            color: #333;
            margin-bottom: 20px;
        }}
        
        table {{
            width: 100%;
            background: white;
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        
        thead {{
            background: linear-gradient(135deg, #2E86AB 0%, #1a4d6b 100%);
            color: white;
        }}
        
        th, td {{
            padding: 15px;
            text-align: center;
        }}
        
        tbody tr:nth-child(even) {{
            background: #f8f9fa;
        }}
        
        tbody tr:hover {{
            background: #e9ecef;
            transition: background 0.3s;
        }}
        
        .holiday-row {{
            background: #fff3cd !important;
            font-weight: bold;
        }}
        
        .footer {{
            background: #333;
            color: white;
            text-align: center;
            padding: 20px;
            font-size: 0.9em;
        }}
        
        .highlight {{
            background: linear-gradient(120deg, #84fab0 0%, #8fd3f4 100%);
            padding: 20px;
            border-radius: 10px;
            margin: 20px 0;
        }}
        
        .highlight h3 {{
            margin-bottom: 10px;
        }}
        
        @media (max-width: 768px) {{
            .stats-grid {{
                grid-template-columns: 1fr;
            }}
            
            .header h1 {{
                font-size: 1.8em;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <div class="header">
            <h1>🚦 Toronto Traffic Flow Predictions</h1>
            <div class="subtitle">7-Day Forecast Dashboard</div>
            <div class="update-time">
                Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            </div>
        </div>
        
        <!-- Statistics Cards -->
        <div class="stats-grid">
            <div class="stat-card visitors">
                <div class="icon">👥</div>
                <div class="value">{stats['avg_visitors']:,.0f}</div>
                <div class="label">Avg Daily Visitors</div>
            </div>
            
            <div class="stat-card vehicles">
                <div class="icon">🚗</div>
                <div class="value">{stats['avg_vehicles']:,.0f}</div>
                <div class="label">Avg Daily Vehicles</div>
            </div>
            
            <div class="stat-card traffic">
                <div class="icon">🚦</div>
                <div class="value">{stats['avg_traffic']:,.0f}</div>
                <div class="label">Avg Traffic Flow</div>
            </div>
        </div>
        
        <!-- Highlights -->
        <div style="padding: 0 40px;">
            <div class="highlight">
                <h3>📈 Peak Traffic Day</h3>
                <p><strong>{stats['max_traffic_date']}</strong> - Expected traffic: <strong>{stats['max_traffic_value']:,.0f}</strong></p>
            </div>
            
            <div class="highlight">
                <h3>📉 Lowest Traffic Day</h3>
                <p><strong>{stats['min_traffic_date']}</strong> - Expected traffic: <strong>{stats['min_traffic_value']:,.0f}</strong></p>
            </div>
        </div>
        
        <!-- Charts -->
        <div class="charts-section">
            <div class="chart-container">
                <h2>👥 Visitor Predictions</h2>
                <img src="data:image/png;base64,{charts['visitors']}" alt="Visitors Prediction Chart">
            </div>
            
            <div class="chart-container">
                <h2>🚗 Vehicle Predictions</h2>
                <img src="data:image/png;base64,{charts['vehicles']}" alt="Vehicles Prediction Chart">
            </div>
            
            <div class="chart-container">
                <h2>🚦 Traffic Flow Predictions</h2>
                <img src="data:image/png;base64,{charts['traffic']}" alt="Traffic Flow Prediction Chart">
            </div>
        </div>
        
        <!-- Detailed Table -->
        <div class="table-section">
            <h2>📅 Daily Predictions Breakdown</h2>
            <table>
                <thead>
                    <tr>
                        <th>Date</th>
                        <th>Holiday</th>
                        <th>Visitors</th>
                        <th>Vehicles</th>
                        <th>Traffic Flow</th>
                        <th>Temperature</th>
                    </tr>
                </thead>
                <tbody>
                    {table_rows}
                </tbody>
            </table>
        </div>
        
        <!-- Footer -->
        <div class="footer">
            <p>Toronto Traffic Flow Prediction System | Powered by Machine Learning</p>
            <p>Two-Stage Prediction Model: Weather → Visitors/Vehicles → Traffic Flow</p>
        </div>
    </div>
</body>
</html>
    """
    
    # 写入文件
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_template)
    
    print(f"✅ HTML dashboard generated: {output_file}")
    return output_file

# ============================================================================
# 使用示例
# ============================================================================

if __name__ == "__main__":
    # 这是一个示例，展示如何使用这些函数
    
    # 假设你已经有了 batch_data（从推理notebook获得）
    # batch_data 应该包含以下列:
    # - date
    # - predicted_visitors
    # - predicted_vehicles
    # - predicted_traffic_count
    # - holidays (可选)
    # - temperature_2m_mean (可选)
    
    print("示例：如何使用HTML生成器")
    print("="*60)
    print("""
    # 在你的推理notebook最后添加:
    
    from generate_dashboard import create_prediction_charts, generate_html_dashboard
    
    # 1. 创建图表
    charts = create_prediction_charts(batch_data)
    
    # 2. 生成HTML
    html_file = generate_html_dashboard(
        batch_data, 
        charts, 
        output_file='traffic_dashboard.html'
    )
    
    # 3. 打开HTML查看
    import webbrowser
    webbrowser.open(html_file)
    """)
