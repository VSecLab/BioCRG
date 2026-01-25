import numpy as np
import math
import matplotlib.pyplot as plt

# Inserisci i parametri
k = 4
N = 3
m = math.ceil(22/N)

# Definizione della funzione
def Gf(n, m, k, N):
    return np.exp(-m * (n**k / N))

t = 25
# Range di n
n_vals = np.linspace(0, t, 500)  # per curva continua
n_points = np.arange(1, t)       # punti interi 1..25

# Calcolo
y_vals = Gf(n_vals, m, k, N)
y_points = Gf(n_points, m, k, N)

print(f"Gf(2) = {Gf(2, m, k, N)}")

# Plot
plt.figure(figsize=(16,8))
plt.plot(n_vals, y_vals, 'b-', label="Gf(n)")  # curva blu
plt.scatter(n_points, y_points, color='r', zorder=5)  # punti rossi
plt.title(rf"$G_f = e^{{-{m} \cdot n^{k} / {N}}}$", fontsize=14)
plt.xlabel("n")
plt.ylabel("Gf")

# Aggiungo i valori accanto ai punti
for n, y in zip(n_points, y_points):
    plt.text(n, y, f"{y:.4f}", fontsize=8, ha='left', va='bottom')

# Grid più fine-grained
plt.grid(True, which='major', linestyle='-', alpha=0.6, linewidth=0.8)  # griglia principale
plt.grid(True, which='minor', linestyle=':', alpha=0.4, linewidth=0.5)  # griglia secondaria più fine
plt.minorticks_on()  # abilita i minor ticks

# Imposta i limiti degli assi per mostrare solo valori positivi a partire dall'origine
plt.xlim(0, None)  # asse x parte da 0
plt.ylim(0, None)  # asse y parte da 0

plt.show()
