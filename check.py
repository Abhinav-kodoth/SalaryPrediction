import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("data/Salary_Data.csv")
print(df.head())

plt.scatter(df["YearsExperience"], df["Salary"])
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.title("Experience vs Salary")
plt.show()