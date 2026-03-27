import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 🔹 STEP 1: Load data
df = pd.read_csv("/Users/manavdoshi/Downloads/diabetes.csv")

# 🔹 STEP 2: Data cleaning
cols = ['Glucose', 'BloodPressure', 'BMI']
for col in cols:
    df[col] = df[col].replace(0, df[col].median())

# 🔹 STEP 3: Insights
print("Total Patients:", len(df))
print("\nDiabetic vs Non-Diabetic Count:\n", df['Outcome'].value_counts())
print("\nAverage Glucose by Outcome:\n", df.groupby('Outcome')['Glucose'].mean())

# 🔹 STEP 4: Dashboard Visualization
sns.set(style="whitegrid")
plt.figure(figsize=(15,10))

# 1. Outcome Distribution
plt.subplot(2,2,1)
sns.countplot(x='Outcome', data=df)
plt.title("Diabetic vs Non-Diabetic")

# 2. BMI Distribution
plt.subplot(2,2,2)
sns.histplot(df['BMI'], bins=20)
plt.title("BMI Distribution")

# 3. Age vs Glucose
plt.subplot(2,2,3)
sns.scatterplot(x='Age', y='Glucose', hue='Outcome', data=df)
plt.title("Age vs Glucose")

# 4. Glucose Distribution
plt.subplot(2,2,4)
sns.histplot(df['Glucose'], bins=20)
plt.title("Glucose Distribution")

# 🔹 Layout
plt.tight_layout()
plt.show()