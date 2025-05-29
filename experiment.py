import os
import pandas as pd
import matplotlib.pyplot as plt

# first getting the data on csv (to use pandas)
STEPS = 10

mode = 0
out_files = ["results/simple.csv", "results/2d.csv", "results/groups.csv"]

while mode <= 2:
    out_csv = open(out_files[mode], "w")
    out_csv.write("gridN;gridM;create_data;copy_data;full_cycle;computing_kernel;updating_buffers;average_cell/s\n")

    i = 32
    while i <= 2 ** 13:
        N = i
        M = i

        result = os.popen("./bin/openclConway " + str(N) + " " + str(M) + " " + str(STEPS) + " " + str(mode)).read().splitlines()[-11 :]
        t_create_data, t_copy_data, t_kernel = result[0].split()

        # calculo celdas evaluadas por segundo
        # en cada iteración se deberian consultar N*M * 9 celdas
        cellsRead = N * M * 9

        #tiempo del kernel computando y pasando memoria
        t_computing = 0
        t_kernel_memory = 0 

        # vemos celdas consultadas/s para cada iteración y sacamos un promedio
        average_cells_per_second = 0
        for line in result[1:]:
            line = line.split()
            t_computing += int(line[0])
            t_kernel_memory += int(line[1])

            average_cells_per_second += (int(line[0]) / cellsRead) / STEPS 

        out_csv.write("{};{};{};{};{};{};{};{:.10f}\n".format(N, M, t_create_data, t_copy_data, t_kernel, t_computing, t_kernel_memory, average_cells_per_second))

        i = i * 2

    print(out_files[mode] + " done!")
    out_csv.close()
    
    mode += 1


# now epic graphs!
simpleDF = pd.read_csv(out_files[0], delimiter=";")
dosDimDF = pd.read_csv(out_files[1], delimiter=";")
gruposDF = pd.read_csv(out_files[2], delimiter=";")

plt.figure(figsize=(8, 5))
plt.plot(simpleDF["gridN"], simpleDF["full_cycle"], color="#0000aa", label = "simple parallel")
plt.plot(dosDimDF["gridN"], dosDimDF["full_cycle"], color="#00aa00", label = "2 dimensions parallel")
plt.plot(gruposDF["gridN"], gruposDF["full_cycle"], color="#aa0000", label = "local memory parallel")

plt.ylabel("Tiempo ciclo completo (μs)")
plt.xlabel("Tamaño de la grilla")

plt.legend()

plt.grid(True)
plt.savefig("results/full_cycle_comp.png")

plt.figure(figsize=(8, 5))
plt.plot(simpleDF["gridN"], simpleDF["computing_kernel"], color="#0000aa", label = "simple parallel")
plt.plot(dosDimDF["gridN"], dosDimDF["computing_kernel"], color="#00aa00", label = "2 dimensions parallel")
plt.plot(gruposDF["gridN"], gruposDF["computing_kernel"], color="#aa0000", label = "local memory parallel")

plt.ylabel("Tiempo total de ejecuciones del kernel (μs)")
plt.xlabel("Tamaño de la grilla")

plt.legend()
plt.grid(True)
plt.savefig("results/computing_kernel_comp.png")