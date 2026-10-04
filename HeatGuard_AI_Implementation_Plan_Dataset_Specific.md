# HeatGuard AI — Dataset-Specific Implementation Plan

## 1. Project Title

**HeatGuard AI: GenAI-Powered Heatwave Prediction, Risk Explanation and Personalized Advisory System**

---

## 2. Project Objective

The project will build a web-based GenAI application using the supplied `heatguard_clean.csv` dataset.

The system will:

1. Analyze historical temperature and weather patterns.
2. Predict whether the **next day** is likely to be a heatwave day.
3. Predict the expected heatwave severity.
4. Display city-wise heatwave risk and historical patterns.
5. Use Generative AI to explain the prediction in simple language.
6. Generate personalized advisories for different stakeholders.
7. Provide a heatwave question-answering assistant.

The project is designed as a student-level prototype of the KJS-CES-01 heatwave intelligence use case. It does not attempt to reproduce a full operational IMD system.

---

# 3. Dataset Used

## Dataset File

`heatguard_clean.csv`

## Dataset Size

- **Rows:** 187,387
- **Columns:** 14
- **Cities:** 7
- **Time period:** 1951 to 2024
- **Cities:** Ahmedabad, Bengaluru, Chennai, Delhi, Kolkata, Mumbai and Pune

The dataset contains daily weather observations and already includes heatwave-related labels.

## Dataset Columns

| Column | Description | Use |
|---|---|---|
| `date` | Observation date | Time-series ordering |
| `city` | City name | Location feature |
| `latitude` | Latitude | Location information |
| `longitude` | Longitude | Mapping/location |
| `temp_max` | Maximum temperature | Main weather feature |
| `temp_min` | Minimum temperature | Weather feature |
| `rain` | Rainfall | Weather feature |
| `year` | Year | Temporal feature |
| `month` | Month | Seasonal feature |
| `day_of_year` | Day number in year | Temporal feature |
| `season` | Winter/Summer/Monsoon/Post-Monsoon | Seasonal feature |
| `heatwave_threshold` | Heatwave threshold | Reference/analysis |
| `is_heatwave_day` | Heatwave indicator | Target/label |
| `severity` | Normal/Warning/Severe/Extreme | Target/label |

---

# 4. Initial Dataset Findings

The dataset contains:

- **187,387 observations**
- **185,596 non-heatwave days**
- **1,791 heatwave days**
- **7 cities**
- **4 severity classes**
  - Normal: 185,596
  - Warning: 1,663
  - Severe: 126
  - Extreme: 2

There are also missing values:

- `temp_min`: 33 missing values
- `rain`: 51,594 missing values

`temp_max`, location, date, season, threshold, heatwave indicator and severity do not contain missing values.

## Important Class Imbalance

The heatwave classes are highly imbalanced. In particular, there are only **2 Extreme observations**.

Therefore, the project should not rely only on accuracy.

The evaluation should include:

- Precision
- Recall
- F1-score
- Confusion matrix
- Balanced accuracy where appropriate
- PR-AUC for the binary heatwave model

For the `Extreme` class, results should be reported cautiously because the dataset contains only two examples.

---

# 5. Important Modeling Decision: Avoid Data Leakage

The dataset already contains:

- `is_heatwave_day`
- `severity`
- `heatwave_threshold`

A model that uses the current day's `temp_max` and then predicts the current day's `is_heatwave_day` can learn the same information used to create the label. This would make the model look better than it really is.

Therefore, the main project will use a **next-day prediction approach**.

## Target

Create:

```text
target_heatwave_next_day
```

This will represent:

```text
is_heatwave_day of the following day
```

For example:

```text
Date        Current Data        Target
10 June     Weather data       11 June heatwave?
11 June     Weather data       12 June heatwave?
12 June     Weather data       13 June heatwave?
```

The model will use information available up to the current day to predict the following day.

---

# 6. Feature Engineering

The model should use historical information rather than future information.

## Current-day features

Potential features:

- `temp_max`
- `temp_min`
- `rain`
- `city`
- `latitude`
- `longitude`
- `month`
- `day_of_year`
- `season`

## Historical/Lag Features

Create:

```text
temp_max_lag1
temp_max_lag2
temp_max_lag3
temp_max_lag7

temp_min_lag1
temp_min_lag2
temp_min_lag3
temp_min_lag7

rain_lag1
rain_lag3
rain_lag7
```

## Rolling Features

Create:

```text
temp_max_3day_avg
temp_max_7day_avg
temp_max_7day_max

temp_min_3day_avg
temp_min_7day_avg

rain_7day_total
```

These features help the model understand persistent heat rather than relying only on one day's temperature.

## Features to Exclude from Prediction

Do not use these as direct input features:

```text
is_heatwave_day
severity
target_heatwave_next_day
```

`heatwave_threshold` should also not be used as a predictive feature unless there is a clearly justified methodology for doing so.

It can instead be retained for analysis and explanation.

---

# 7. Data Preprocessing

## Step 1: Load Dataset

```python
import pandas as pd

df = pd.read_csv("heatguard_clean.csv")
```

## Step 2: Convert Date

```python
df["date"] = pd.to_datetime(df["date"])
```

## Step 3: Sort Data

Sort by:

```text
city + date
```

This is essential before creating lag and rolling features.

## Step 4: Handle Missing Values

### `temp_min`

Only 33 values are missing.

Use an appropriate time-series interpolation or city-level strategy.

### `rain`

There are 51,594 missing values.

Do not automatically assume every missing rainfall value means zero rainfall.

The project should inspect the missingness pattern first.

If the dataset documentation supports interpreting missing rainfall as no recorded rainfall, zero imputation may be considered. Otherwise, use a clearly documented imputation strategy.

## Step 5: Create Target

Within each city:

```text
target_heatwave_next_day =
    next day's is_heatwave_day
```

The final row of each city's time series will not have a next-day target and should be excluded from supervised training.

---

# 8. Machine Learning Tasks

The project will use two related ML tasks.

## Task A: Next-Day Heatwave Prediction

### Target

```text
target_heatwave_next_day
```

Values:

```text
0 = No heatwave expected
1 = Heatwave expected
```

### Recommended Models

Start with:

1. Logistic Regression
2. Decision Tree
3. Random Forest

Optionally compare with:

4. XGBoost

The final model should be selected based on appropriate validation metrics rather than simply choosing the model with the highest accuracy.

---

# 9. Task B: Next-Day Severity Prediction

After creating the next-day target, create:

```text
target_severity_next_day
```

Possible classes:

```text
Normal
Warning
Severe
Extreme
```

Because the severity classes are extremely imbalanced, this task should be treated as a secondary model.

Recommended approach:

### Option 1 — Multiclass Model

Predict:

```text
Normal
Warning
Severe
Extreme
```

### Option 2 — Safer Student Prototype

Use a simpler risk classification:

```text
Normal
Warning+
```

and use the available severity information for analysis and advisory generation.

The final decision should depend on the model's validation results.

The project must explicitly mention the limitation caused by only two `Extreme` observations.

---

# 10. Train/Test Split

Because this is time-series data, **do not randomly shuffle the entire dataset for the main evaluation**.

Use a chronological split.

Example:

```text
1951–2015    Training
2016–2020    Validation
2021–2024    Testing
```

The exact split can be adjusted after checking the final usable date range.

This better represents the real-world situation where historical observations are used to predict future conditions.

---

# 11. Model Output

For a selected city, the system should produce something similar to:

```text
Location: Pune
Date: 2024-06-20

Current Maximum Temperature: 39.8°C
Recent 3-Day Average: 38.9°C
Recent 7-Day Average: 38.2°C

Predicted Next-Day Heatwave Risk: HIGH
Probability: 0.81
Predicted Severity: Warning/Severe
```

The actual values will come from the trained model.

---

# 12. GenAI Component

The GenAI component is the main Generative AI part of the project.

The ML model produces structured information.

The LLM converts that information into understandable communication.

## Flow

```text
Weather Data
      ↓
Feature Engineering
      ↓
ML Prediction
      ↓
Risk + Probability + Severity
      ↓
Structured Prompt
      ↓
Generative AI
      ↓
Explanation + Advisory
```

The LLM should not independently invent temperature values or heatwave conditions.

---

# 13. GenAI Advisory Generator

The user selects an audience.

### Supported audiences

1. Citizen
2. Farmer
3. Health Agency
4. Local Authority

Example structured input:

```json
{
    "city": "Pune",
    "temperature": 41.0,
    "risk": "High",
    "probability": 0.81,
    "severity": "Severe",
    "audience": "Citizen"
}
```

The GenAI model generates a short advisory based on those values.

---

# 14. GenAI Prompt

Use a controlled prompt similar to:

```text
You are a heatwave advisory assistant.

Generate a clear and concise heatwave advisory using
only the verified information provided below.

City: {city}
Current temperature: {temperature}
Predicted heatwave risk: {risk}
Prediction probability: {probability}
Predicted severity: {severity}
Audience: {audience}

Explain the risk in simple language.

Provide practical precautions appropriate for the
selected audience.

Do not invent weather measurements.
Do not provide medical diagnosis.
Do not issue official emergency orders.
Do not claim certainty when the prediction is uncertain.
```

This makes the GenAI output grounded in the ML system's results.

---

# 15. GenAI Heatwave Assistant

Add a chatbot to the application.

## Example questions

```text
Is Pune likely to experience a heatwave tomorrow?

Why is the heatwave risk high?

What precautions should I take?

What should farmers do during a heatwave?

Explain the prediction in simple words.

Which cities have the highest recent heatwave risk?
```

For location-specific questions, the backend should first retrieve the relevant dataset/model information and then provide it to the LLM.

---

# 16. Dashboard

## Dashboard Page

Display:

```text
HEATGUARD AI

Select City:
[ Pune ▼ ]

Current Maximum Temperature
39.8°C

Predicted Next-Day Risk
HIGH

Probability
81%

Predicted Severity
SEVERE
```

Also display:

- Temperature trend
- Heatwave history
- Rainfall trend
- Risk summary
- AI-generated explanation

---

# 17. Heatwave Map

The dataset provides:

```text
latitude
longitude
city
```

Therefore, an interactive map can be created.

Suggested technology:

- Leaflet.js
- OpenStreetMap

The map can display the seven cities:

```text
Ahmedabad
Bengaluru
Chennai
Delhi
Kolkata
Mumbai
Pune
```

Each marker can show:

```text
City
Current/selected temperature
Predicted risk
Predicted severity
```

---

# 18. Historical Analytics

Use the dataset's long historical period for analysis.

Possible visualizations:

### Temperature Trend

```text
Year vs Average Maximum Temperature
```

### Heatwave Count

```text
Year vs Number of Heatwave Days
```

### City Comparison

```text
City vs Heatwave Days
```

### Seasonal Analysis

```text
Season vs Heatwave Frequency
```

### Severity Distribution

```text
Normal
Warning
Severe
Extreme
```

Because the dataset contains observations from 1951 onward, these visualizations can provide a strong historical component.

---

# 19. Backend Architecture

Use Flask.

Suggested structure:

```text
heatguard-ai/
│
├── app.py
├── requirements.txt
│
├── data/
│   └── heatguard_clean.csv
│
├── models/
│   ├── heatwave_model.pkl
│   └── severity_model.pkl
│
├── preprocessing/
│   └── feature_engineering.py
│
├── services/
│   ├── prediction.py
│   ├── genai.py
│   └── analytics.py
│
├── templates/
│   ├── index.html
│   ├── dashboard.html
│   ├── advisory.html
│   └── chatbot.html
│
└── static/
    ├── css/
    └── js/
```

---

# 20. API Endpoints

## Get Cities

```text
GET /api/cities
```

Returns the seven available cities.

## Get Historical Data

```text
GET /api/history/<city>
```

Returns historical data for dashboard charts.

## Predict Heatwave Risk

```text
POST /api/predict
```

Input:

```json
{
    "city": "Pune",
    "date": "2024-06-18"
}
```

Output:

```json
{
    "city": "Pune",
    "risk": "High",
    "probability": 0.81,
    "severity": "Severe"
}
```

## Generate Advisory

```text
POST /api/advisory
```

Input:

```json
{
    "city": "Pune",
    "audience": "citizen"
}
```

The backend obtains the relevant prediction and sends structured information to the GenAI model.

## Chatbot

```text
POST /api/chat
```

Input:

```json
{
    "city": "Pune",
    "question": "Why is the risk high?"
}
```

---

# 21. Database

A database is optional for the first version because the historical dataset is already available as CSV.

If a database is required, MongoDB can store:

```text
users
prediction_history
generated_advisories
chat_history
```

The original CSV can remain the primary historical data source.

---

# 22. Application Pages

## Page 1 — Home

Contains:

- Project name
- Project objective
- Short explanation
- Start button

## Page 2 — Heatwave Dashboard

Contains:

- City selection
- Temperature information
- Predicted risk
- Severity
- Probability
- Historical graphs
- Map

## Page 3 — AI Advisory

Contains:

```text
Select city
Select audience
[Generate Advisory]
```

Then display the GenAI output.

## Page 4 — AI Assistant

Contains:

- Chat interface
- Suggested questions
- Location selection

## Page 5 — Analytics

Contains:

- Historical heatwave trends
- City comparisons
- Seasonal analysis
- Severity distribution

---

# 23. Recommended Development Sequence

## Phase 1 — Dataset Understanding

Tasks:

- Load CSV
- Check data types
- Check missing values
- Check duplicate records
- Check date range
- Check city distribution
- Check severity distribution
- Perform EDA

Deliverable:

```text
EDA notebook
```

---

## Phase 2 — Feature Engineering

Tasks:

- Sort by city and date
- Handle missing values
- Create lag features
- Create rolling averages
- Create next-day heatwave target
- Create next-day severity target

Deliverable:

```text
model_ready_dataset.csv
```

---

## Phase 3 — ML Model

Tasks:

- Chronological train/validation/test split
- Train Logistic Regression
- Train Random Forest
- Compare models
- Handle class imbalance
- Evaluate precision, recall and F1
- Select final model
- Save model

Deliverable:

```text
heatwave_model.pkl
```

---

## Phase 4 — Analytics

Tasks:

- City heatwave counts
- Seasonal analysis
- Temperature trends
- Heatwave trends
- Severity distribution

Deliverable:

```text
EDA + analytics notebook
```

---

## Phase 5 — Flask Backend

Tasks:

- Create Flask application
- Load trained model
- Create prediction API
- Create historical data API
- Create analytics API

Deliverable:

```text
Working backend
```

---

## Phase 6 — Frontend

Tasks:

- Create dashboard
- Add city selection
- Add charts
- Add map
- Add prediction cards
- Add advisory page
- Add chatbot page

Deliverable:

```text
Working web interface
```

---

## Phase 7 — GenAI Integration

Tasks:

- Select LLM/API
- Create controlled prompts
- Implement advisory generator
- Implement explanation generator
- Implement chatbot
- Connect ML outputs to GenAI
- Add basic safety checks

Deliverable:

```text
Working GenAI module
```

---

## Phase 8 — Full Integration

Final flow:

```text
CSV Dataset
     ↓
Preprocessing
     ↓
Feature Engineering
     ↓
ML Model
     ↓
Next-Day Heatwave Prediction
     ↓
Flask Backend
     ↓
       ┌──────────────┐
       │              │
       ↓              ↓
 Dashboard         GenAI
                       ↓
              Advisory + Explanation
                       ↓
                    Chatbot
```

---

# 24. Testing Plan

## ML Testing

Test:

- Normal weather conditions
- High-temperature conditions
- Different cities
- Different seasons
- Missing values
- Historical periods not seen during training

Metrics:

- Precision
- Recall
- F1-score
- Confusion matrix
- PR-AUC
- Balanced accuracy where useful

## GenAI Testing

Create a test set of questions/advisory scenarios.

Check:

- Factual consistency
- Relevance
- Clarity
- Stakeholder appropriateness
- Hallucination
- Safety
- Whether the response follows supplied prediction values

## Web Application Testing

Test:

- City selection
- Prediction generation
- Map
- Charts
- Advisory generation
- Chatbot
- Invalid input
- API failure
- Missing data

---

# 25. Team Division

For a three-member team:

## Member 1 — Data + ML

Responsible for:

- Dataset analysis
- EDA
- Feature engineering
- Heatwave prediction
- Severity model
- Model evaluation

## Member 2 — Backend + GenAI

Responsible for:

- Flask
- APIs
- Model integration
- LLM/API integration
- Prompt design
- Chatbot backend

## Member 3 — Frontend + Visualization

Responsible for:

- Dashboard
- Charts
- Heatmap
- Advisory interface
- Chatbot interface
- UI/UX

All members should participate in integration, testing, report writing and presentation.

---

# 26. Minimum Viable Product

If project time is limited, implement these first:

### Required

1. Load `heatguard_clean.csv`
2. Preprocess data
3. Create next-day target
4. Train heatwave prediction model
5. Evaluate model
6. Create Flask backend
7. Build dashboard
8. Display city-wise prediction
9. Add heatmap
10. Add GenAI advisory
11. Add GenAI chatbot

### Optional

- Severity model
- MongoDB
- Multilingual output
- RAG
- Email/SMS simulation
- Voice assistant
- IoT sensors

---

# 27. Final Demonstration Scenario

Use **Pune** as one demonstration city because it is present in the supplied dataset.

Example demonstration:

```text
Step 1
User selects Pune.

        ↓

Step 2
System retrieves historical observations.

        ↓

Step 3
Feature engineering creates recent temperature
and weather-history features.

        ↓

Step 4
ML model predicts next-day heatwave risk.

        ↓

Step 5
Dashboard displays probability and risk.

        ↓

Step 6
User selects "Citizen".

        ↓

Step 7
GenAI generates a personalized advisory.

        ↓

Step 8
User asks:
"Why is the heatwave risk high?"

        ↓

Step 9
GenAI explains the prediction using the
available model outputs.
```

The exact prediction values in the demonstration must come from the trained model rather than being manually inserted.

---

# 28. Final Deliverables

The mini project should produce:

1. Cleaned/preprocessed dataset
2. EDA notebook
3. Feature-engineering notebook/script
4. Trained ML model
5. Model evaluation results
6. Flask backend
7. Web dashboard
8. Interactive heatmap
9. GenAI advisory generator
10. GenAI chatbot
11. Source code
12. System architecture
13. Use case diagram
14. DFD
15. Testing results
16. Project report
17. Presentation/demo

---

# 29. Final Project Flow

The final system can be summarized as:

```text
             heatguard_clean.csv
                     |
                     v
            Data Preprocessing
                     |
                     v
             Feature Engineering
                     |
                     v
              ML Prediction
                     |
          +----------+----------+
          |                     |
          v                     v
    Heatwave Risk          Severity
          |                     |
          +----------+----------+
                     |
                     v
               Flask Backend
                     |
          +----------+----------+
          |                     |
          v                     v
     Dashboard               GenAI
          |              /      |       \
          |             /       |        \
          v            v        v         v
       Heatmap     Advisory  Explanation Chatbot
          |             |        |         |
          +-------------+--------+---------+
                        |
                        v
                    End User
```

## 30. Project Outcome

The final prototype will demonstrate how historical meteorological observations can be used to create a heatwave prediction system and how Generative AI can turn structured prediction results into understandable, audience-specific information.

The key distinction in the project is:

**ML predicts the heatwave risk.**

**GenAI explains the prediction and generates the advisory.**

This keeps the project clearly aligned with the GenAI requirement while making effective use of the supplied dataset.
