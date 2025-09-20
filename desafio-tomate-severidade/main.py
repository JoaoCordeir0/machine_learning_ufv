"""
@author: João Victor Cordeiro
"""

#%% importando as libs
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

# %% Importando o dataset
df = pd.read_csv('dataset/dataset_tomate_com_severidade.csv')

X = df.drop(['id', 'Severidade'], axis = 1)
y = df['Severidade']

# %% Separando dados de treinamento e dados de teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# %% Padronizando os dados
scaler = StandardScaler()  
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

X_train_sc = pd.DataFrame(X_train_sc)
X_train_sc.columns = X_train.columns

X_test_sc = pd.DataFrame(X_test_sc)
X_test_sc.columns = X_train.columns

# %% Declarando o modelo
# model = LinearRegression() # 0.887634277870405
# model = ElasticNet(alpha=0.1, l1_ratio=0.5) # 0.8918284807596035
# model = Lasso(alpha=0.01) # 0.8962638374263361
model = RandomForestRegressor(max_depth=10, min_samples_split=2, min_samples_leaf=1, random_state=0) # 0.9021049779712956

# %% Escolha da quantidade de features
score_list = list()
for i in range(1, 21):
    selector = RFE(model, n_features_to_select=i, step=1)
    selector = selector.fit(X_train_sc, y_train)
    mask = selector.support_
    features = X_train_sc.columns
    sel_features = features[mask]
    X_sel = X_train_sc[sel_features]
    scores = cross_val_score(model, X_sel, y_train, cv=10, scoring='r2')
    score_list.append((i, np.mean(scores)))

n_features, score = sorted(score_list, key=lambda x: x[1], reverse=True)[0]

print(n_features, score)

# %% Aplicando as features
selector = RFE(model, n_features_to_select=n_features, step=1)
selector = selector.fit(X_train_sc, y_train)
mask = selector.support_
features = X_train_sc.columns
sel_features = features[mask]
X_sel = X_train_sc[sel_features]
scores = cross_val_score(model, X_sel, y_train, cv=10, scoring='r2')

print(np.mean(scores))
print(sel_features)

# %% Modelo final
model.fit(X_sel, y_train)

# %% Testando o modelo com os dados de teste
y_pred = model.predict(X_test_sc[sel_features])
r2 = model.score(X_test_sc[sel_features], y_test)
rmse = (mean_squared_error(y_test, y_pred)**0.5)
mae = mean_absolute_error(y_test, y_pred)

print('r2', r2)
print('rmse', rmse)
print('mae', mae)

# %% Plotando o gráfico de desempenho em função da quantidade de features
plt.plot([i for i, _ in score_list], [score for _, score in score_list], marker='o')
plt.xlabel("Número de Features Selecionadas")
plt.ylabel("R² Médio (Cross-Validation)")
plt.title("Desempenho em função da quantidade de features")
plt.show()

# %%
