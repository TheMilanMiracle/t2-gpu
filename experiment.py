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
    out_csv.write("gridN;gridM;create_data;copy_data;full_cycle;computing_kernel;updating_buffers;average_cell_per_sec\n")

    i = 32
    while i <= 2 ** 13:
        N, M = i, i

        if _bin == "opencl":

            mode_arg = {
                "simple" : 0,
                "2d" : 1,
                "groups" : 2
            }

            result = os.popen("./bin/opencl " + str(N) + " " + str(M) + " " + str(STEPS) + " " + str(mode_arg[mode])).read().splitlines()[-11 :]
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

                average_cells_per_second += (int(line[0]) / cellsRead) / STEPS * 1_000_000

            out_csv.write("{};{};{};{};{};{};{};{:.10f}\n".format(N, M, t_create_data, t_copy_data, t_kernel, t_computing, t_kernel_memory, average_cells_per_second))

        elif _bin == "cuda":

            mode_arg = {
                "simple" : "simple",
                "2d" : "array2d",
                "groups" : "localmemory"
            }

            result = os.popen("./bin/cuda " + str(N) + " " + str(M) + " " + str(STEPS) + " " + str(mode_arg[mode]))
            result = result.read()

            cellsRead = N * M * 9

            t_create_data = result.split('\n')[0].split()[4]
            t_copy_data = result.split('\n')[1].split()[5]
            t_kernel = result.split('\n')[len(result.split('\n')) - 3].split()[2]

            t_kernel_memory = 0
            t_computing = 0
            for r in [x.split()[3] for x in list(filter(lambda x: 'Updated' in x, result.split('\n')[2: len(result.split('\n')) - 3]))]:
                t_kernel_memory += int(r)
            for r in [x.split()[4] for x in list(filter(lambda x: 'step' in x, result.split('\n')[2: len(result.split('\n')) - 3]))]:
                t_computing += int(r)

            average_cells_per_second = float(result.split('\n')[len(result.split('\n')) - 2].split()[3]) / cellsRead * 1_000_000 # multiply by 10^6 for conversion comp/μs to comps/s

            out_csv.write("{};{};{};{};{};{};{};{:.10f}\n".format(N, M, t_create_data, t_copy_data, t_kernel, t_computing, t_kernel_memory, average_cells_per_second))

        print(f"{mode} mode with size {i}x{i} done!")
        out_csv.flush()

        i = i * 2

# now epic graphs!
simpleDF = pd.read_csv(out_files[0], delimiter=";")
dosDimDF = pd.read_csv(out_files[1], delimiter=";")
gruposDF = pd.read_csv(out_files[2], delimiter=";")

def plot(ylabel: str, title: str, file: str):
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(simpleDF["gridN"], simpleDF[ylabel], color="#0000aa", label="simple parallel")
    ax.plot(dosDimDF["gridN"], dosDimDF[ylabel], color="#00aa00", label="2 dimensions parallel")
    ax.plot(gruposDF["gridN"], gruposDF[ylabel], color="#aa0000", label="local memory parallel")

    ax.set_ylabel(title)
    ax.set_xlabel("Tamaño de la grilla")

    ax.set_xscale('log', base=2)

    ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(file)


plot("average_cell_per_sec", "Evaluaciones/s", f"results/{_bin}_comps_second.png")
plot("full_cycle", "Tiempo ciclo completo (μs)", f"results/{_bin}_full_cycle_comp.png")
plot("computing_kernel", "Tiempo total usado para ejecutar el kernel (μs)", f"results/{_bin}_computing_kernel_comp.png")
plot("updating_buffers", "Tiempo total usado para actualizar buffers (μs)", f"results/{_bin}_updatingTime.png")