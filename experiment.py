import os
import pandas as pd
import matplotlib.pyplot as plt
import sys


if len(sys.argv) != 2:
    print(f"Usage: {sys.argv[0]} {'{'}opencl|cuda{'}'}")
    exit(1)

_bin = sys.argv[1]

# first getting the data on csv (to use pandas)
STEPS = 10

modes = ["simple", "2d", "groups"]
res_folder = "./results"

out_files = []
for mode in modes:
    out_file = os.path.join(res_folder, f"{_bin}-{mode}.csv")
    out_files.append(out_file)

    out_csv = open(out_file, "w")
    out_csv.write("gridN;gridM;create_data;copy_data;full_cycle;computing_kernel;updating_buffers;average_cell/s\n")

    i = 32
    while i <= 2 ** 13:
        N, M = i, i

        if _bin == "opencl":

            mode_arg = {
                "simple" : 0,
                "2d" : 1,
                "groups" : 2
            }

            result = os.popen("./bin/opencl " + str(N) + " " + str(M) + " " + str(STEPS) + " " + str(mode_arg[mode]))
            t_create_data, t_copy_data, t_kernel = result.read().splitlines()[-11 :][0].split()

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

            out_csv.write("{};{};{};{};{};{};{:.10f}\n".format(N, M, t_create_data, t_copy_data, t_computing, t_kernel_memory, average_cells_per_second))

        elif _bin == "cuda":

            mode_arg = {
                "simple" : "simple",
                "2d" : "array2d",
                "groups" : "localmemory"
            }

            result = os.popen("./bin/cuda " + str(N) + " " + str(M) + " " + str(STEPS) + " " + str(mode_arg[mode]))
            result = result.read()

            t_create_data = result.split('\n')[0].split()[4]
            t_copy_data = result.split('\n')[1].split()[5]
            t_computing = result.split('\n')[len(result.split('\n')) - 3].split()[2]
            t_kernel_memory = 0
            for r in [x.split()[4] for x in list(filter(lambda x: 'Updated' not in x, result.split('\n')[2: len(result.split('\n')) - 3]))]:
                t_kernel_memory += int(r)
            average_cells_per_second = 9 / float(result.split('\n')[len(result.split('\n')) - 2].split()[3])

            out_csv.write("{};{};{};{};{};{};{:.10f}\n".format(N, M, t_create_data, t_copy_data, t_computing, t_kernel_memory, average_cells_per_second))

        print(f"{mode} mode with size {i}x{i} done!")
        out_csv.flush()

        i = i * 2

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
plt.xscale('log', base=2)

plt.legend()
plt.grid(True)
plt.savefig(f"results/{_bin}_full_cycle_comp.png")

plt.figure(figsize=(8, 5))
plt.plot(simpleDF["gridN"], simpleDF["computing_kernel"], color="#0000aa", label = "simple parallel")
plt.plot(dosDimDF["gridN"], dosDimDF["computing_kernel"], color="#00aa00", label = "2 dimensions parallel")
plt.plot(gruposDF["gridN"], gruposDF["computing_kernel"], color="#aa0000", label = "local memory parallel")

plt.ylabel("Tiempo total de ejecuciones del kernel (μs)")
plt.xlabel("Tamaño de la grilla")
plt.xscale('log', base=2)

plt.legend()
plt.grid(True)
plt.savefig(f"results/{_bin}_computing_kernel_comp.png")

simpleDF["update%"] = simpleDF["updating_buffers"] * 100 / simpleDF["full_cycle"]
dosDimDF["update%"] = dosDimDF["updating_buffers"] * 100 / dosDimDF["full_cycle"]
gruposDF["update%"] = gruposDF["updating_buffers"] * 100 / gruposDF["full_cycle"]

plt.figure(figsize=(8, 5))
plt.plot(simpleDF["gridN"], simpleDF["update%"], color="#0000aa", label = "simple parallel")
plt.plot(dosDimDF["gridN"], dosDimDF["update%"], color="#00aa00", label = "2 dimensions parallel")
plt.plot(gruposDF["gridN"], gruposDF["update%"], color="#aa0000", label = "local memory parallel")

plt.ylabel("Porcentaje del tiempo del ciclo usado en actualizar los buffers")
plt.xlabel("Tamaño de la grilla")
plt.xscale('log', base=2)

plt.legend()
plt.grid(True)
plt.savefig(f"results/{_bin}_updatingTime.png")