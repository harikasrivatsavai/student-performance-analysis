import pandas as pd

df = pd.read_csv("StudentsPerformance.csv")

print(df.head())
print(df.shape)
print(df.columns)
print(df.info())
print(df.isnull().sum())
print(df.describe())
import seaborn as sns
import matplotlib.pyplot as plt
sns.histplot(df['math score'])
plt.savefig("histplot.png")
plt.show()
sns.boxplot(x=df['math score'])
plt.savefig("boxplot.png")
plt.show()
plt.figure(figsize=(6,4))

sns.heatmap(
    df[['math score', 'reading score', 'writing score']].corr(),
    annot=True
)

plt.title("Correlation Heatmap")

plt.savefig("correlationheatmap.png")
plt.show()
plt.figure(figsize=(6,4))

sns.barplot(
    x='gender',
    y='math score',
    data=df
)

plt.title("Gender vs Math Score")

plt.savefig("gvsmbarplot.png")
plt.show()
plt.figure(figsize=(6,4))

sns.barplot(
    x='lunch',
    y='math score',
    data=df
)

plt.title("Lunch Type vs Math Score")

plt.savefig("lvsmbarplot.png")
plt.show()
plt.figure(figsize=(6,4))

sns.barplot(
    x='test preparation course',
    y='math score',
    data=df
)

plt.title("Test Preparation vs Math Score")

plt.savefig("tvsmbarplot.png")
plt.show()

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
X = df[['reading score', 'writing score']]
y = df['math score']
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
model = LinearRegression()

model.fit(X_train, y_train)
predictions = model.predict(X_test)
error = mean_absolute_error(y_test, predictions)

print("Mean Absolute Error:", error)
plt.figure(figsize=(6,4))

plt.scatter(y_test, predictions)

plt.xlabel("Actual Scores")
plt.ylabel("Predicted Scores")

plt.title("Actual vs Predicted Scores")

plt.savefig("avspscatter.png")
plt.show()
