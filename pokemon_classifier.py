import pandas as pd
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

## importing the data
# get the directory where the current script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# build full path to the CSV file in the same folder as script
csv_path = os.path.join(script_dir, 'dataset_.csv')

# load the CSV using the full path
df = pd.read_csv(csv_path)

# data clearing and editing
df = df.dropna()
counts = df['Type 1'].value_counts()
valid_types = counts[counts >= 5].index
df = df[df['Type 1'].isin(valid_types)]

# encoding the string type data
le = LabelEncoder()

# choosing the features, X and y
features = ['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed', 'Total', 'Legendary']
X = df[features]
y = le.fit_transform(df['Type 1'])

# scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# split the data into train and test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, )

# list of models
models = {
    'KNN (k=5)': KNeighborsClassifier(n_neighbors=5),
    'SVM (linear)': SVC(kernel='linear', class_weight='balanced'),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
}

# model predictions and score
for name, model in models.items():
    scores = cross_val_score(model, X_scaled, y, cv=5)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f'\n{name} - Accuracy: {acc:.2f}')
    print(classification_report(y_test, y_pred, zero_division=0))
