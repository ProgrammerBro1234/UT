import matplotlib.pyplot as plt

LSTM = [0.18, 0.04, -0.01, -0.12, 0.04, -0.22, -0.71, -0.52, -0.43, -1.62, -1.26]

Toeline = [-0.02, 0.7, -0.32, 0.72, -0.28, 0.26, -0.33, -2.17, -0.82, 1.84, -0.07]

plt.plot(range(len(LSTM)), LSTM, color='blue', marker='o', label='LSTM')
plt.plot(range(len(Toeline)), Toeline, color='red', marker='o', label='Toeline')

plt.xlabel('Ajasamm')
plt.ylabel('Hinna muutus')
plt.title('LSTM ja Tõeline')
plt.legend()

plt.show()


