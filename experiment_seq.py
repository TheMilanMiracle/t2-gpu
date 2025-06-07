import os
import pandas as pd
import matplotlib.pyplot as plt
import sys

STEPS = 10

res_folder = "./results"
out_file = os.path.join(res_folder, f"seq.csv")
out_csv = open(out_file, "w")
out_csv.write("gridN;gridM;create_data;full_cycle;computing_kernel;average_cell_per_sec\n")


i = 32
while i <= 2 ** 13:
    N, M = i, i

    result = os.popen("./bin/seq " + str(N) + " " + str(M) + " " + str(STEPS)).read().split('\n')

    cellsRead = N * M * 9

    avg_time = float(result[len(result) - 2].split()[-2])
    t_kernel = int(result[len(result) - 3].split()[-2])
    t_create_data = int(result[0].split()[-2])

    t_computing = 0
    for r in [x.split()[-2] for x in list(filter(lambda x: 'step' in x, result[2: len(result) - 3]))]:
        t_computing += int(r)

    average_cells_per_second = avg_time / cellsRead * 1_000_000
    out_csv.write("{};{};{};{};{};{:.10f}\n".format(N, M, t_create_data, t_kernel, t_computing, average_cells_per_second))

    out_csv.flush()

    print(f"seq with size {i}x{i} done!")

    i = i * 2

df = pd.read_csv(out_file, delimiter=";")

plt.figure(figsize=(8, 5))
plt.plot(df["gridN"], df["average_cell_per_sec"], color="#0000aa", label = "sequential implementation")
plt.ylabel("Evaluaciones/s")
plt.xlabel("Tamaño de la grilla")
plt.xscale('log', base=2)

plt.legend()
plt.grid(True)
plt.savefig(f"results/seq_comps_second.png")


plt.figure(figsize=(8, 5))
plt.plot(df["gridN"], df["computing_kernel"], color="#0000aa", label = "sequential implementation")

plt.ylabel("Tiempo total de ejecuciones del algoritmo (μs)")
plt.xlabel("Tamaño de la grilla")
plt.xscale('log', base=2)

plt.legend()
plt.grid(True)
plt.savefig(f"results/seq_computing_kernel_comp.png")


plt.figure(figsize=(8, 5))
plt.plot(df["gridN"], df["full_cycle"], color="#0000aa", label = "sequential implementation")

plt.ylabel("Tiempo ciclo completo (μs)")
plt.xlabel("Tamaño de la grilla")
plt.xscale('log', base=2)

plt.legend()
plt.grid(True)
plt.savefig(f"results/seq_full_cycle_comp.png")




