import numpy as np            
import pandas as pd             
import matplotlib.pyplot as plt

df = pd.read_csv("data/Salary_Data.csv")

print("First 5 rows of the dataset:")
print(df.head())

X_raw = df["YearsExperience"].values
y = df["Salary"].values

X_mean = X_raw.mean()
X_std = X_raw.std()
X = (X_raw - X_mean) / X_std

def compute_cost(X,y,w,b):
    m=len(y) #no of ex
    predictions=w*X+b
    cost = (1 / (2 * m)) * np.sum((predictions - y) ** 2)
 
    return cost

def gradient_descent(X, y, w, b, learning_rate, iterations):
    m=len(y)
    cost_history=[]

    for i in range(iterations):
        predictions=w*X+b

        dw = (1 / m) * np.sum((predictions - y) * X)

        db=  (1/m) * np.sum(predictions-y)

        w = w - learning_rate * dw
        b = b - learning_rate * db

        cost = compute_cost(X, y, w, b)
        cost_history.append(cost)

        if i % 100 == 0:
            print(f"Iteration {i}: cost = {cost:.2f}")

    return w, b, cost_history

w=0
b=0
learning_rate=0.1
iterations=1000

w, b, cost_history = gradient_descent(X, y, w, b, learning_rate, iterations)

print(f"\nFinal learned weight (w): {w:.2f}")
print(f"Final learned bias (b): {b:.2f}")


plt.plot(cost_history)
plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.title("Cost over iterations")
plt.show()


plt.scatter(X_raw, y, label="Actual data")
plt.plot(X_raw, w * X + b, color="red", label="Our fitted line")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.title("Linear Regression Fit (from scratch)")
plt.legend()
plt.show()


def predict_salary(years_experience, w, b, X_mean, X_std):
    """
    
    We must scale the input the same way we scaled our training data,
    since the model was trained on scaled X.
    """
    scaled_input = (years_experience - X_mean) / X_std
    return w * scaled_input + b

new_experience = 7
predicted_salary = predict_salary(new_experience, w, b, X_mean, X_std)
print(f"\nPredicted salary for {new_experience} years of experience: {predicted_salary:.2f}")