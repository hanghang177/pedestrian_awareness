import matplotlib.pyplot as plt
import numpy as np

def sigmoid(x, ymin, ymax, L50, U50):
    a = (L50 + U50)/2
    b = 2 / abs(L50 - U50)
    c = ymin
    d = ymax - c
    sig = c + (d/(1+np.exp(b*(a-x))))
    return sig

head_orientations = np.arange(0, 90, 0.001)
p_looking_aways = sigmoid(head_orientations, 0.0, 1.0, 30, 60)

cellphone_confs = np.arange(0, 1.0, 0.001)
p_cellphones = sigmoid(cellphone_confs, 0.0, 1.0, 0.5, 0.8)

plt.figure(1)
plt.plot(head_orientations, p_looking_aways)
plt.xlabel("Head Orientation (Degrees)")
plt.ylabel("Probability of Looking Away")
plt.title("Head Orientation Sigmoid")
plt.grid()

plt.figure(2)
plt.plot(cellphone_confs, p_cellphones)
plt.xlabel("Confidence of Detecting Cellphone")
plt.ylabel("Probability of Having Cellphone")
plt.title("Cellphone Probability Sigmoid")
plt.grid()

plt.show()
