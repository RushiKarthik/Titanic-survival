import pandas as pd
df = pd.read_csv("train.csv")
#Age fix
#Remove Cabin column
df.drop(columns = "Cabin", inplace = True)
#Embarked fix
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
#print(df[["SibSp", "Parch", "FamilySize"]].head())
#print(df["FamilySize"].value_counts())
df["IsAlone"] = 0
df.loc[df["FamilySize"] == 1, "IsAlone"] = 1
#print(df[["FamilySize", "IsAlone"]].head(10))
#print(df["Name"].head())
df["Title"] = df["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
#print(df[["Name", "Title"]].head())
#print(df["Title"].value_counts())
rare_titles = [
         "Mlle", "Mme", "Ms", "Lady", "Countess", "Capt", "Col", "Don", "Dr", "Major", "Rev", "Sir", "Jonkheer", "Dona"
]
df["Title"] = df["Title"].replace(rare_titles, "Rare")
df["Age"] = df["Age"].fillna(df["Age"].median())
#print(df["Title"].value_counts())
#print(df[[ "Pclass","Sex","Age","Fare","Embarked","FamilySize","IsAlone","Title"]].head())
#print(df.columns)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df["Sex"] = le.fit_transform(df["Sex"])
#print(df["Sex"].head())
df = pd.get_dummies(df, columns=["Embarked"], dtype=int)
df = pd.get_dummies(df, columns=["Title"], dtype=int)
df.drop(columns=["PassengerId", "Name", "Ticket"], inplace=True)
df["FarePerPerson"] = df["Fare"] / df["FamilySize"]
X = df.drop(columns = ["Survived"])
y = df["Survived"]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)
#print(X_train.shape)
#print(X_test.shape)
#print(y_train.shape)
#print(y_test.shape)
#print(df.columns)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=5,
    min_samples_split=5,
    random_state=42
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(" RandomForest Accuracy:", accuracy)
"""importance = pd.DataFrame({
         "Feature": X.columns,
          "Importance": model.feature_importances_
})
print(importance.sort_values(by="Importance", ascending=False)) """
print(model.score(X_train, y_train))
print(model.score(X_test, y_test))
#print(df.isnull().sum())
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
#print(cm)
from sklearn.metrics import precision_score, recall_score, f1_score

#print("Precision:", precision_score(y_test, y_pred))
#print("Recall:", recall_score(y_test, y_pred))
#print("F1 Score:", f1_score(y_test, y_pred))
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)
y_pred_knn = knn.predict(X_test_scaled)
print("KNN Accuracy:", accuracy_score(y_test, y_pred_knn))
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(X_train, y_train)
y_pred_nb = nb.predict(X_test)
print("Naive Bayes Accuracy:", accuracy_score(y_test, y_pred_nb))
from sklearn.svm import SVC
svm = SVC()
svm.fit(X_train_scaled, y_train)
y_pred_svm = svm.predict(X_test_scaled)
print("SVM Accuracy:", accuracy_score(y_test, y_pred_svm))
import matplotlib.pyplot as plt

models = ["KNN", "Naive Bayes", "SVM", "Random Forest"]
scores = [82.12, 78.21, 81.01, 83.24]

plt.figure(figsize=(6,4))
plt.bar(models, scores)

plt.title("Model Comparison")
plt.ylabel("Accuracy (%)")

plt.show()



