import matplotlib.pyplot as plt
SVR = [-0.03,
  -0.05,
  -0.1,
  -0.05,
  -0.1,
  0.07,
  -0.09,
  -0.14,
  0.31,
  0.3,
  0.37]
Toeline = [-0.02, 0.7, -0.32, 0.72, -0.28, 0.26, -0.33, -2.17, -0.82, 1.84, -0.07]
plt.plot(range(len(SVR)), SVR, color='blue', marker='o', label='SVR')
plt.plot(range(len(Toeline)), Toeline, color='red', marker='o', label='Toeline')

plt.xlabel('Ajasamm')
plt.ylabel('Hinna muutus')
plt.title('SVR ja Tõeline')
plt.legend()

plt.show()
