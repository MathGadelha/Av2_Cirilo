import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("aerogerador.dat", sep="\t", header=None)
data.columns = ["Wind_Speed", "Power_Output"]

degree = 3

def polynomial_features(X, degree):
    X_poly = np.ones((len(X), degree + 1))
    for i in range(1, degree + 1):
        X_poly[:, i] = (X.flatten() ** i)
    return X_poly

X = data["Wind_Speed"].values.reshape(-1, 1)
y = data["Power_Output"].values

n_simulations = 1000
results = []

for _ in range(n_simulations):
    indices = np.random.permutation(len(X))
    train_size = int(0.8 * len(X))
    X_train, X_test = X[indices[:train_size]], X[indices[train_size:]]
    y_train, y_test = y[indices[:train_size]], y[indices[train_size:]]
    
    X_train_poly = polynomial_features(X_train, degree)
    X_test_poly = polynomial_features(X_test, degree)
    
    theta_best = np.linalg.inv(X_train_poly.T.dot(X_train_poly)).dot(X_train_poly.T).dot(y_train)
    
    y_pred = X_test_poly.dot(theta_best)
    rss = np.sum((y_test - y_pred) ** 2) 
    results.append(rss)

rss_mean = np.mean(results)
rss_std = np.std(results)
rss_max = np.max(results)
rss_min = np.min(results)

print("Resultados para a Regressão Polinomial (grau {}):".format(degree))
print("Média do RSS:", rss_mean)
print("Desvio-Padrão do RSS:", rss_std)
print("Maior RSS:", rss_max)
print("Menor RSS:", rss_min)

X_poly = polynomial_features(X, degree)
y_pred_full = X_poly.dot(theta_best)

plt.scatter(data["Wind_Speed"], data["Power_Output"], color="blue", label="Dados Observados")
plt.plot(np.sort(X, axis=0), y_pred_full[np.argsort(X, axis=0)], color="red", label="Regressão Polinomial")
plt.xlabel("Velocidade do Vento")
plt.ylabel("Potência Gerada")
plt.title("Regressão Polinomial - Velocidade do Vento vs Potência Gerada")
plt.legend()
plt.show()
