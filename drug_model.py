import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Step 1: Create Synthetic Dataset
np.random.seed(42)

num_samples = 1000

data = pd.DataFrame({
    'molecular_weight': np.random.normal(300, 50, num_samples),
    'lipophilicity': np.random.normal(2.5, 1.0, num_samples),
    'toxicity_score': np.random.normal(5, 2, num_samples),
    'binding_affinity': np.random.normal(-7, 1, num_samples)
})

# Binary target: 1 = Side Effect, 0 = No Side Effect
data['side_effect'] = (data['toxicity_score'] + 
                       data['lipophilicity'] > 7).astype(int)

X = data.drop('side_effect', axis=1)
y = data['side_effect']

# Step 2: Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Step 3: Scale Data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 4: Build Neural Network
model = Sequential([
    Dense(16, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Step 5: Train Model
model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=1)

# Step 6: Evaluate
y_pred = model.predict(X_test)
y_pred_class = (y_pred > 0.5).astype(int)

print("Classification Report:")
print(classification_report(y_test, y_pred_class))

print("ROC-AUC Score:", roc_auc_score(y_test, y_pred))