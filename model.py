# 1. Import Libraries
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

# 2. Load and Inspect Data
data = pd.read_csv('data/iris.csv')
print(data.head(5))

# 3. Split into Train/Test Sets
train, test = train_test_split(
    data,
    test_size=0.4,
    stratify=data['species'],
    random_state=42
)

X_train = train[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y_train = train['species']

X_test = test[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y_test = test['species']

# 4. Train Decision Tree Classifier
mod_dt = DecisionTreeClassifier(max_depth=3, random_state=1)
mod_dt.fit(X_train, y_train)

# 5. Make Predictions
prediction = mod_dt.predict(X_test)

# 6. Evaluate Model
accuracy = metrics.accuracy_score(prediction, y_test)
print(f'The accuracy of the Decision Tree is {accuracy:.3f}')
