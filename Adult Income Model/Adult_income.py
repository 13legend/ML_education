import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder


FILE_NAME = "adult.csv"

df = pd.read_csv("adult.csv")
df = df.replace("?", np.nan)

# статистика по всем столбцам
# print(df.describe(include="all"))

data_pandas = df
X = data_pandas.drop("income", axis=1)
y = (df["income"] == ">50K").astype(int)

num = ["age", "fnlwgt", "education.num", "capital.gain", "capital.loss", "hours.per.week"]
cat = [col for col in X.columns if col not in num]

# В столбце native.country слишком много городов, которые мало встречаются в выборке, поэтому логично объединить их в 1 категорию
counts = df["native.country"].value_counts()
rare = counts[counts < 100].index
df["native.country"] = df["native.country"].replace(rare, "Other")

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Проверка на отсутствующие значения в признаках
# missing_mask = data_pandas.isnull()
# print(missing_mask)

# Графики для анализа данных
#Распределение числовых признаков
# fig, axes = plt.subplots(1, 5, figsize=(15, 4))
# for ax, col in zip(axes, num):
#     sns.histplot(df[col], bins=40, kde=True, ax=ax)
#     ax.set_title(col)
# plt.tight_layout()
# plt.show()

# Боксплоты для числовых признаков
# fig, axes = plt.subplots(1, 5, figsize=(15, 4))
# for ax, col in zip(axes, num):
#     sns.boxplot(x="income", y=col, data=data_pandas, ax=ax)
#     ax.set_title(col)
# plt.tight_layout()
# plt.show()

# Каунтплоты для категориальных признаков
# for col in cat:
#     plt.figure(figsize=(8, 4))
#     sns.countplot(data=data_pandas, x=col, hue="income")
#     plt.title(col)
#     plt.xticks(rotation=45, ha="right")
#     plt.tight_layout()
#     plt.show()

# Преобразование данных
preprocessor = ColumnTransformer(
    transformers=[
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), num),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore"))
        ]), cat),
    ],
    remainder="drop"
)

preprocessor_tree = ColumnTransformer(
    transformers=[
        ("num", SimpleImputer(strategy="median"), num),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ]), cat),
    ]
)

# Подготовка моделей
model1 = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced"))
])

model2 = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", SGDClassifier(loss="log_loss", penalty="l2", class_weight="balanced"))
])

model3 = Pipeline(steps=[
    ("preprocessor", preprocessor_tree),
    ("classifier", RandomForestClassifier(class_weight="balanced"))
])

model4 = Pipeline(steps=[
    ("preprocessor", preprocessor_tree),
    ("classifier", GradientBoostingClassifier())
])

models = {
    "LogReg": model1,
    "SGDclassifier": model2,
    "RandForest": model3,
    "GradBoost": model4
}



for name, mod in models.items():
    mod.fit(X_train, y_train)

    y_pred = mod.predict(X_test)

    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_pred)
    print(f"{name}: f1 = {f1:.5f}, precision = {prec:.5f}, recall = {rec:.5f}, roc-auc = {roc:.5f}")

