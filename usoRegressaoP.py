import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm
import scipy.stats as stats
import seaborn as sns


base = pd.read_csv('Data/mt_cars.csv')
base.drop(['Unnamed: 0'], axis=1, inplace=True)
print(base.head())

#fiz a correlação e vou usar mpg como variavel independente e as variaveis com a correlação
# mais forte (próximas de 1) são:
corr = base.corr()
sns.heatmap(corr, cmap='coolwarm', annot=True, fmt='.2f')

plt.tight_layout()
#plt.show()

#gráfico de dispersão para pares de colunas que vou definir

column_pairs = [('mpg', 'cyl'), ('mpg', 'disp'), ('mpg', 'hp'), ('mpg', 'wt'), ('mpg', 'drat'), ('mpg', 'vs'), ]
n_plots = len(column_pairs)
fig, axes = plt.subplots(nrows=n_plots, ncols=1, figsize=(6, 4*n_plots))

for i, pair in enumerate(column_pairs):
    x_col, y_col = pair
    sns.scatterplot(x=x_col, y=y_col, data=base, ax=axes[i])
    axes[i].set_title(f'{x_col} vs {y_col}')

plt.tight_layout()
#plt.show()

#os gráficos que mais estiverem próximos de formar uma linha são os melhores
#mpg vs cyl é ruim, muito disperso
#mpg vs disp, por outro lado, é um bom candidadto

#criando modelo


# modelo = sm.ols(formula='mpg ~ wt + disp + hp', data=base) modelo que confirma hipotese #aic 156.6 bic 162.5
# modelo = sm.ols(formula='mpg ~ disp + cyl', data=base) #aic: 165.1 bic:169.5 subiu, é ruim
modelo = sm.ols(formula='mpg ~ drat + vs', data=base) # aic: 179.1 bic: 183.5
modelo = modelo.fit()
print(modelo.summary())



residuos = modelo.resid
fig2, ax = plt.subplots(figsize=(8,6))
ax.hist(residuos, bins=20)
ax.set_xlabel('Residuos')
ax.set_ylabel('Frequencia')
ax.set_title('Histograma de Residuos')

fig3, ax2 = plt.subplots(figsize=(8,6))
stats.probplot(residuos, dist='norm', plot=ax2)
plt.title("Q-Q Plot de Residuos")
plt.show()

#teste de shapiro wilk
#quanto mais perto de1 melhor
#h0 (hipotese nula) - dados estão normalmente distribuidos
# p <= 0.05 rejeito a hipótese nula (não estão normalmente distribuidos)
# p > 0.05 não é possível rejeitar h0
stat, pval = stats.shapiro(residuos)
print(f'Shapir-Wilk statistica: {stat:.3f}, p-value: {pval:.3f}')