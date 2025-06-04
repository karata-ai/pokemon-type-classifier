import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# importing the data
df = pd.read_csv('dataset_.csv')

# data clearing and editing
df = df.dropna()
counts = df['Type 1'].value_counts()
valid_types = counts[counts >= 5].index
df = df[df['Type 1'].isin(valid_types)]

# choosing the features, X and y
features = ['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']
X = df[features]
y = df['Type 1']

# scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# split the data into train and test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, )

# list of models
models = {
    'KNN (k=5)': KNeighborsClassifier(n_neighbors=5),
    'SVM (linear)': SVC(kernel='linear'),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
}

# model predictions and score
for name, model in models.items():
    scores = cross_val_score(model, X_scaled, y, cv=5)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f'\n{name} - Accuracy: {acc:.2f}')
    print(classification_report(y_test, y_pred, zero_division=0))
