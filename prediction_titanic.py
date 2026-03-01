from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import fetch_openml

titanic = fetch_openml('titanic',version=1,as_frame=True,parser='auto')
df = titanic.frame

columns_to_keep = ['pclass','sex','age','sibsp','parch','fare','survived']
df = df[columns_to_keep].copy()

df['sex'] = df['sex'].map({'male': 0, 'female':1})
df = df.dropna()

X = df[['pclass','sex','age','sibsp','parch','fare']]
y = df['survived']

X_train, X_test, y_train, y_test = train_test_split(
    X,y,
    test_size = 0.2,
    random_state = 42
)

model = LogisticRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print(f'real(10 people): {list(y_test[:10])}')
print(f'predictions(10 people): {list(predictions[:10])}')
print(f'accuracy: {accuracy:.3f}')