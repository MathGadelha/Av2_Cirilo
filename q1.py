import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Carregar os dados do arquivo
data = pd.read_csv("aerogerador.dat", sep="\t", header=None)
data.columns = ["Wind_Speed", "Power_Output"]

# 1. Visualização inicial dos dados
plt.scatter(data["Wind_Speed"], data["Power_Output"], color="blue")
plt.xlabel("Velocidade do Vento")
plt.ylabel("Potência Gerada")
plt.title("Gráfico de Dispersão: Velocidade do Vento vs Potência Gerada")
plt.show()

# 2. Organização dos dados em X e y
X = data[["Wind_Speed"]].values  # variável independente (velocidade do vento)
y = data["Power_Output"].values  # variável dependente (potência gerada)

# Adiciona uma coluna de 1s para o termo de bias no modelo
X_b = np.c_[np.ones((len(X), 1)), X]

# Parâmetros
lambdas = [0, 0.25, 0.5, 0.75, 1]  # valores de lambda para regularização
n_simulations = 500  # número de simulações de Monte Carlo
results = []

# 3. Implementação dos modelos com Simulação Monte Carlo
for lmbd in lambdas:
    rss_values = []
    for _ in range(n_simulations):
        # Divisão dos dados em 80% treinamento e 20% teste
        indices = np.random.permutation(len(X_b))
        train_size = int(0.8 * len(X_b))
        X_train, X_test = X_b[indices[:train_size]], X_b[indices[train_size:]]
        y_train, y_test = y[indices[:train_size]], y[indices[train_size:]]
        
        # Calculo dos coeficientes para MQO tradicional e MQO Regularizado (Ridge)
        if lmbd == 0:
            # MQO tradicional
            theta_best = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ y_train
        else:
            # MQO Regularizado (Tikhonov - Ridge)
            regularization_matrix = lmbd * np.eye(X_train.shape[1])
            theta_best = np.linalg.inv(X_train.T @ X_train + regularization_matrix) @ X_train.T @ y_train
        
        # Previsões e cálculo do RSS
        y_pred = X_test @ theta_best
        rss = np.sum((y_test - y_pred) ** 2)
        rss_values.append(rss)
    
    # Armazenar as métricas para cada valor de lambda
    rss_mean = np.mean(rss_values)
    rss_std = np.std(rss_values)
    rss_max = np.max(rss_values)
    rss_min = np.min(rss_values)
    results.append((f"MQO Regularizado (λ={lmbd})" if lmbd != 0 else "MQO Tradicional", rss_mean, rss_std, rss_max, rss_min))

# 4. Tabela de Resultados
results_df = pd.DataFrame(results, columns=["Modelo", "Média do RSS", "Desvio-Padrão", "Máximo", "Mínimo"])
print("\nTabela de Resultados Final:")
print(results_df.to_string(index=False))

# 5. Visualização dos Resultados
plt.errorbar(results_df["Modelo"], results_df["Média do RSS"], yerr=results_df["Desvio-Padrão"], fmt='-o', capsize=5)
plt.xlabel("Modelo")
plt.ylabel("Média do RSS")
plt.title("Média do RSS com Desvio-Padrão para cada Modelo")
plt.xticks(rotation=45, ha="right")
plt.show()
