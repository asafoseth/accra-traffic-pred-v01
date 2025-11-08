@echo off
echo ==========================================
echo ACCRA TRAFFIC PREDICTION - QUICK SETUP
echo ==========================================

mkdir data\raw\traffic data\raw\weather data\processed outputs\models outputs\figures logs

python generate_data.py
pip install -q streamlit pandas numpy plotly scikit-learn xgboost folium streamlit-folium

echo Setup complete!
echo Run: streamlit run app.py