#!/usr/bin/env python
# coding: utf-8

# In[11]:


# pak_air_quality_model_with_plots.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
get_ipython().system('pip install streamlit')
import streamlit as st
import pandas as pd



# 1. Load dataset
df = pd.read_csv("/content/pakistan_air_quality_final_clean (1).csv")

# 2. Take a random sample of 2000 rows
df_sample = df.sample(n=2000, random_state=42)

# Display columns to identify the correct target column name
print("Columns in df_sample:", df_sample.columns)

# 3. Define features and target
# Remove 'timestamp' and 'date' as they are strings and cannot be directly used by the model
X = df_sample.drop(columns=["aqi_category", "timestamp", "date", "day_of_week", "month_name", "season"])
y = df_sample["aqi_category"]

# Apply one-hot encoding to the 'city' column
X = pd.get_dummies(X, columns=['city'], drop_first=True)

# 4. Train-test split (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5. Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 6. Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)
print("Classification Report:\n", classification_report(y_test, y_pred))

# 7. Save model for deployment
joblib.dump(model, "air_quality_model.pkl")

# -------------------------------
# Visualization Section
# -------------------------------

# Plot distribution of each feature
for col in X.columns:
    # Check if column is numeric before plotting distribution
    if pd.api.types.is_numeric_dtype(X[col]):
        plt.figure(figsize=(6,4))
        sns.histplot(X[col], kde=True, bins=30)
        plt.title(f"Distribution of {col}")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.show()

# Correlation heatmap
plt.figure(figsize=(10,8))
corr_matrix = X.corr()
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.show()

# Feature importance from model
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10,6))
sns.barplot(x=importances[indices], y=X.columns[indices])
plt.title("Feature Importance (Random Forest)")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()


# In[ ]:


get_ipython().system('streamlit run streamlit_app.py')


# In[12]:


get_ipython().run_cell_magic('writefile', 'streamlit_app.py', '\nimport streamlit as st\nimport pandas as pd\nimport joblib\nimport numpy as np\n\n# Load the trained model\nmodel = joblib.load(\'air_quality_model.pkl\')\n\n# Define the feature names that the model was trained on\n# This list must exactly match the columns and their order used during training\nfeature_names = [\n    \'latitude\', \'longitude\', \'pm10\', \'pm2_5\', \'carbon_monoxide\',\n    \'nitrogen_dioxide\', \'sulphur_dioxide\', \'ozone\', \'dust\', \'temperature\',\n    \'humidity\', \'precipitation\', \'wind_speed\', \'wind_direction\', \'pressure\',\n    \'hour\', \'month\', \'year\', \'is_weekend\',\n    \'city_Faisalabad\', \'city_Islamabad\', \'city_Karachi\', \'city_Lahore\',\n    \'city_Multan\', \'city_Peshawar\', \'city_Quetta\', \'city_Rawalpindi\', \'city_Sialkot\'\n]\n\n# Define the list of cities for the selectbox\n# These are the cities that generated the one-hot encoded columns, plus the base city if drop_first=True was used\n# In this case, we\'ll assume the cities that generated the dummy columns are the only ones we\'ll offer.\npossible_cities = [\n    \'Faisalabad\', \'Islamabad\', \'Karachi\', \'Lahore\', \'Multan\', \'Peshawar\',\n    \'Quetta\', \'Rawalpindi\', \'Sialkot\'\n]\n\nst.set_page_config(page_title="Air Quality Prediction", layout="centered")\nst.title(\'Pakistan Air Quality Prediction App\')\nst.write(\'Enter the details below to predict the Air Quality Index (AQI) Category.\')\n\n# Create input fields for numerical features\nwith st.sidebar:\n    st.header(\'Input Features\')\n    latitude = st.number_input(\'Latitude\', value=30.3753, format="%.4f")\n    longitude = st.number_input(\'Longitude\', value=69.3451, format="%.4f")\n    pm10 = st.number_input(\'PM10 (Particulate Matter)\', value=50.0)\n    pm2_5 = st.number_input(\'PM2.5 (Fine Particulate Matter)\', value=25.0)\n    carbon_monoxide = st.number_input(\'Carbon Monoxide (ppb)\', value=500.0)\n    nitrogen_dioxide = st.number_input(\'Nitrogen Dioxide (ppb)\', value=20.0)\n    sulphur_dioxide = st.number_input(\'Sulphur Dioxide (ppb)\', value=10.0)\n    ozone = st.number_input(\'Ozone (ppb)\', value=40.0)\n    dust = st.number_input(\'Dust (ug/m3)\', value=30.0)\n    temperature = st.number_input(\'Temperature (°C)\', value=25.0)\n    humidity = st.number_input(\'Humidity (%)\', value=60.0)\n    precipitation = st.number_input(\'Precipitation (mm)\', value=0.0)\n    wind_speed = st.number_input(\'Wind Speed (m/s)\', value=5.0)\n    wind_direction = st.number_input(\'Wind Direction (°)\', value=180.0)\n    pressure = st.number_input(\'Pressure (hPa)\', value=1012.0)\n    hour = st.slider(\'Hour of Day\', 0, 23, 12)\n    month = st.slider(\'Month\', 1, 12, 6)\n    year = st.slider(\'Year\', 2023, 2026, 2025)\n    is_weekend = st.selectbox(\'Is Weekend?\', [0, 1], format_func=lambda x: \'Yes\' if x == 1 else \'No\')\n    selected_city = st.selectbox(\'City\', possible_cities)\n\n# Create a dictionary for user input\nuser_input = {\n    \'latitude\': latitude,\n    \'longitude\': longitude,\n    \'pm10\': pm10,\n    \'pm2_5\': pm2_5,\n    \'carbon_monoxide\': carbon_monoxide,\n    \'nitrogen_dioxide\': nitrogen_dioxide,\n    \'sulphur_dioxide\': sulphur_dioxide,\n    \'ozone\': ozone,\n    \'dust\': dust,\n    \'temperature\': temperature,\n    \'humidity\': humidity,\n    \'precipitation\': precipitation,\n    \'wind_speed\': wind_speed,\n    \'wind_direction\': wind_direction,\n    \'pressure\': pressure,\n    \'hour\': hour,\n    \'month\': month,\n    \'year\': year,\n    \'is_weekend\': is_weekend\n}\n\n# Convert user input to DataFrame, ensuring all feature_names are present\ninput_df = pd.DataFrame([user_input])\n\n# Add one-hot encoded city columns\nfor city in possible_cities:\n    input_df[f\'city_{city}\'] = 0\n# Set the selected city to 1\nif f\'city_{selected_city}\' in input_df.columns:\n    input_df[f\'city_{selected_city}\'] = 1\n\n# Ensure the order of columns matches the training data\n# Handle the case where a city might have been dropped during training due to drop_first=True\n# For simplicity, we assume \'city_Islamabad\' was dropped if present in original, so we don\'t have it here.\n# But the `feature_names` list should reflect exactly what the model expects.\n# If `drop_first=True` was used, one city\'s dummy variable would be missing.\n# The current feature_names list already reflects the dropped column (if any) since it came from X.columns.\n\n# Reindex input_df to match the order of feature_names\n# Any columns in feature_names not in input_df will be added as 0 (e.g., the \'dropped\' city)\n# Any columns in input_df not in feature_names will be dropped (shouldn\'t happen if logic is correct)\nfinal_input_df = input_df.reindex(columns=feature_names, fill_value=0)\n\n# Make prediction\nif st.button(\'Predict AQI Category\'):\n    prediction = model.predict(final_input_df)\n    st.success(f\'The predicted Air Quality Index Category is: **{prediction[0]}**\')\n\nst.markdown("""\n---\nThis app uses a Random Forest Classifier to predict the Air Quality Index (AQI) Category based on various environmental parameters.\n""")\n')


# To run the Streamlit app, first ensure `streamlit_app.py` is saved in your Colab environment. Then execute the following command in a new code cell:

# In[13]:


get_ipython().system('streamlit run streamlit_app.py &>/dev/null&')


# A public URL will be provided, typically within a minute, that you can click to access your Streamlit application in a new browser tab.
