# HR Promotion Prediction - Full Pipeline

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Step 1: Load Data
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
sample_submission = pd.read_csv("sample_submission.csv")

print("✅ Data Loaded")
print("Train shape:", train.shape)
print("Test shape:", test.shape)
print("Sample Submission:", sample_submission.head())

# Step 2: Handle Missing Values
train['education'] = train['education'].fillna('Unknown')
test['education'] = test['education'].fillna('Unknown')

train['previous_year_rating'] = train['previous_year_rating'].fillna(train['previous_year_rating'].median())
test['previous_year_rating'] = test['previous_year_rating'].fillna(test['previous_year_rating'].median())

# Step 3: Encode Categorical Variables
categorical_columns = ['department', 'region', 'education', 'gender', 'recruitment_channel']
combined = pd.concat([train[categorical_columns], test[categorical_columns]])

le_dict = {}

for col in categorical_columns:
    le = LabelEncoder()
    le.fit(combined[col])
    train[col] = le.transform(train[col])
    test[col] = le.transform(test[col])
    le_dict[col] = le  # Save encoder for later use

print("✅ Categorical columns encoded")

# Step 4: Prepare Features and Target
X = train.drop(['employee_id', 'is_promoted'], axis=1)
y = train['is_promoted']
X_test = test.drop(['employee_id'], axis=1)

# Step 5: Train/Test Split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Step 6: Train Model
model = RandomForestClassifier(
    n_estimators=100,
    class_weight='balanced',
    random_state=42
)
model.fit(X_train, y_train)

# Step 7: Evaluate Model
y_pred = model.predict(X_val)
print("📊 Validation Results:")
print(confusion_matrix(y_val, y_pred))
print(classification_report(y_val, y_pred))

# Step 8: Predict on Test Set
test_preds = model.predict(X_test)

# Step 9: Create Submission File
submission = pd.DataFrame({
    'employee_id': test['employee_id'],
    'is_promoted': test_preds
})
submission.to_csv("submission.csv", index=False)
print("✅ submission.csv created successfully!")

# Step 10: Save Model and Encoders
joblib.dump(model, 'model.pkl')
joblib.dump(le_dict, 'encoders.pkl')
print("💾 Model and encoders saved successfully as model.pkl and encoders.pkl")
