import matplotlib.pyplot as plt

LSTM = [0.18, 0.04, -0.01, -0.12, 0.04, -0.22, -0.71, -0.52, -0.43, -1.62, -1.26]
SVR = [-0.03, -0.05, -0.1, -0.05, -0.1, 0.07, -0.09, -0.14, 0.31, 0.3, 0.37]
SVR_1 = [-0.14, -0.09, -0.17, -0.26, -0.01, 0.11, -0.18, -0.13, -0.25, 0.16, 0.33]
Toeline = [-0.02, 0.7, -0.32, 0.72, -0.28, 0.26, -0.33, -2.17, -0.82, 1.84, -0.07]

plt.plot(range(len(LSTM)), LSTM, color='blue', marker='o', label='LSTM')
plt.plot(range(len(SVR)), SVR, color='yellow', marker='o', label='SVR kogu andmestik')
plt.plot(range(len(SVR_1)), SVR_1, color='green', marker='o', label="SVR 1 tulp")
plt.plot(range(len(Toeline)), Toeline, color='red', marker='o', label='Toeline')

plt.xlabel('Ajasamm')
plt.ylabel('Hinna muutus')
plt.title('Mudelid vs Tõeline')
plt.legend()

plt.show()
