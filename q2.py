import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate

data = pd.read_csv("EMGsDataset.csv", header=None)
X = data.iloc[:2, :].T.values 
y = data.iloc[2, :].values

plt.scatter(X[:, 0], X[:, 1], c=y, cmap="viridis", alpha=0.5)
plt.xlabel("Sensor 1 (Corrugador do Supercílio)")
plt.ylabel("Sensor 2 (Zigomático Maior)")
plt.title("Gráfico de Dispersão dos Dados EMG por Categoria")
plt.colorbar(label="Classe")
plt.show()

lambdas = [0.25, 0.5, 0.75] 
n_simulations = 500 
results = []

def gaussian_density(x, mean, cov):
    size = len(x)
    cov = cov + np.eye(size) * 1e-6 
    det = np.linalg.det(cov)
    norm_const = 1.0 / (np.power(2 * np.pi, float(size) / 2) * np.sqrt(det))
    x_mu = x - mean
    inv = np.linalg.inv(cov)
    result = np.exp(-0.5 * (x_mu @ inv @ x_mu.T))
    return max(norm_const * result, 1e-300)

def run_models(X_train, X_test, y_train, y_test):
    accuracies = {}
    classes = np.unique(y_train)
    
    means = {c: np.mean(X_train[y_train == c], axis=0) for c in classes}
    covariances = {c: np.cov(X_train[y_train == c], rowvar=False) + np.eye(X_train.shape[1]) * 1e-6 for c in classes}
    priors = {c: max(np.mean(y_train == c), 1e-6) for c in classes} 

    from numpy.linalg import inv
    X_train_b = np.c_[np.ones((len(X_train), 1)), X_train]
    theta_best = inv(X_train_b.T @ X_train_b) @ X_train_b.T @ y_train
    y_pred_mqo = (X_test @ theta_best[1:] + theta_best[0] > 0.5).astype(int)
    accuracies["MQO Tradicional"] = np.mean(y_pred_mqo == y_test)

    predictions = []
    for x in X_test:
        class_probs = {c: np.log(priors[c]) + np.log(gaussian_density(x, means[c], covariances[c])) for c in classes}
        predictions.append(max(class_probs, key=class_probs.get))
    accuracies["Classificador Gaussiano Tradicional"] = np.mean(predictions == y_test)

    mean_cov = np.mean([covariances[c] for c in classes], axis=0)
    predictions = []
    for x in X_test:
        class_probs = {c: np.log(priors[c]) + np.log(gaussian_density(x, means[c], mean_cov)) for c in classes}
        predictions.append(max(class_probs, key=class_probs.get))
    accuracies["Classificador Gaussiano com Covariâncias Iguais"] = np.mean(predictions == y_test)

    shared_cov = np.cov(X_train, rowvar=False) + np.eye(X_train.shape[1]) * 1e-6
    predictions = []
    for x in X_test:
        class_probs = {c: np.log(priors[c]) + np.log(gaussian_density(x, means[c], shared_cov)) for c in classes}
        predictions.append(max(class_probs, key=class_probs.get))
    accuracies["Classificador Gaussiano com Matriz Agregada"] = np.mean(predictions == y_test)

    for lmbd in lambdas:
        reg_cov = shared_cov + lmbd * np.eye(X_train.shape[1])
        predictions = []
        for x in X_test:
            class_probs = {c: np.log(priors[c]) + np.log(gaussian_density(x, means[c], reg_cov)) for c in classes}
            predictions.append(max(class_probs, key=class_probs.get))
        accuracies[f"Classificador Gaussiano Regularizado (λ={lmbd})"] = np.mean(predictions == y_test)

    naive_means = means
    naive_vars = {c: np.var(X_train[y_train == c], axis=0) + 1e-6 for c in classes}
    predictions = []
    for x in X_test:
        class_probs = {c: np.log(priors[c]) + np.sum(np.log(gaussian_density(x, naive_means[c], np.diag(naive_vars[c])))) for c in classes}
        predictions.append(max(class_probs, key=class_probs.get))
    accuracies["Classificador de Bayes Ingênuo"] = np.mean(predictions == y_test)

    return accuracies

for i in range(n_simulations):
    indices = np.random.permutation(len(X))
    train_size = int(0.8 * len(X))
    X_train, X_test = X[indices[:train_size]], X[indices[train_size:]]
    y_train, y_test = y[indices[:train_size]], y[indices[train_size:]]
    
    accuracies = run_models(X_train, X_test, y_train, y_test)
    results.append(accuracies)
    
    if i % 100 == 0:
        print(f"Simulação {i+1}/{n_simulations} concluída")
        print("Exemplo de acurácias:", accuracies)

results_df = pd.DataFrame(results)
summary_df = pd.DataFrame({
    "Modelo": results_df.columns,
    "Média": results_df.mean(),
    "Desvio-Padrão": results_df.std(),
    "Maior Valor": results_df.max(),
    "Menor Valor": results_df.min()
}).reset_index(drop=True)

print("\nTabela de Resultados Consolidados:")
print(tabulate(summary_df, headers="keys", tablefmt="grid", showindex=False))

summary_df_formatada = pd.DataFrame({
    "Modelo": [
        "MQO trad", 
        "Gaussian trad", 
        "Gaussian_Cov_Train", 
        "Gaussian_Cov_Aggregated", 
        "Naive_Bayes",
        "Regularized_Gaussian_0.25",
        "Regularized_Gaussian_0.5",
        "Regularized_Gaussian_0.75"
    ],
    "Mean": summary_df["Média"].round(4),
    "Std": summary_df["Desvio-Padrão"].round(4),
    "Max": summary_df["Maior Valor"].round(4),
    "Min": summary_df["Menor Valor"].round(4)
})

print("\nTABLE II")
print("TAXA DE ACERTO DE DIFERENTES MODELOS (R = 500)\n")
print(tabulate(summary_df_formatada, headers="keys", tablefmt="grid", showindex=False))
