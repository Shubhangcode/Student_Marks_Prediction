import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

df = pd.read_csv("D:\AI_ML_Projects\Student_marks_Predictor\student_data.csv")
df.head()
df.info()
df.shape
sns.set(style="whitegrid")
df.isna().sum().sum()
df.dropna(inplace=True)
df = df.dropna()
df.isna().sum()
df.duplicated().sum()
df.describe()
df.describe(include='object')
categorical_cols = ['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']
for cols in categorical_cols:
    print(f"Value counts for {cols}:\n{df[cols].value_counts()}\n")
    df.hist(bins=20, edgecolor='black')
plt.tight_layout()
plt.show()
    
     
for col in categorical_cols:
  sns.countplot(data=df, x=col)
  plt.title(f"Distribution of {col}")
  plt.xticks(rotation=45)
  plt.show()
  df.corr(numeric_only=True)
  sns.heatmap(df.corr(numeric_only=True), annot=True,cmap="coolwarm",fmt=".2f")
plt.title("Correlation Matrix")
plt.show()
df.describe().columns
num_features=['math score', 'reading score', 'writing score', 'exam_score']
df['exam_score'] = df[['math score', 'reading score', 'writing score']].mean(axis=1)
for feature in num_features:
  sns.scatterplot(data = df, x=feature, y='exam_score')
  plt.title(f"{feature} vs Exams Score")
  plt.show()
df['exam_score'] = df[['math score', 'reading score', 'writing score']].mean(axis=1)
print(df.head())

for col in categorical_cols:
  sns.boxplot(data=df, x=col, y='exam_score')
  plt.title(f" Exams Score by {col}")
  plt.xticks(rotation=45)
  plt.show()
  df.columns
  df.head(2)
features = ['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']
target = "exam_score"
import os

os.makedirs("data/processed", exist_ok=True)

df.to_csv("data/processed/processed_data.csv", index=False)
print("Processed data saved successfully.")