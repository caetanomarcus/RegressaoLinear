from numpy import array
from Classes.RegressaoLinearClass import LinearRegression

x = array([1, 2, 3, 4, 5])
y = array([2, 4, 6, 8, 10])

lr = LinearRegression(x, y)

previsao = lr.previsao(1.5)
print("previsão:", previsao)
