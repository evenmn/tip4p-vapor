# import numpy as np
# import matplotlib.pyplot as plt
# from lammps_analyzer import average
# from lammps_analyzer.log import Log
from lammps_simulator import Simulator
from lammps_simulator.computer import CPU


def run_simulation(var):
    path = "simulation/vapor_pressure"
    sim = Simulator(path, overwrite=True)
    sim.copy_to_wd("H2O.TIP4P", "water_molecule.data")
    sim.set_input_script("vapor_pressure.in", **var)
    sim.run(CPU(num_procs=32))
    return path


'''
def analyze_simulation(path, plot=False):
    logger = Log(path + "/log.lammps", ignore_first=-1)
    temp = logger.find("Temp")
    energy = logger.find("TotEng")
    enthalpy = logger.find("Enthalpy")
    atoms = logger.find("Atoms")
    molecules = atoms[0] // 3

    # find average values
    window = int(len(energy)/100)
    temp_ave = average(temp, window)
    energy_ave = average(energy, window)
    enthalpy_ave = average(enthalpy, window)

    boil_ind = np.argmax(np.diff(energy_ave))

    liquid_ind = boil_ind - 2
    vapor_ind = boil_ind + 2

    enthalpy_liquid = enthalpy_ave[liquid_ind]
    enthalpy_vapor = enthalpy_ave[vapor_ind]
    H_vapor = enthalpy_vapor - enthalpy_liquid

    T = temp_ave[boil_ind]
    H = H_vapor / molecules

    if plot:
        plt.figure()
        plt.title(f"T_boil : {T:.3f} K. H_vapor : {H:.3f} kcal/mol")
        plt.plot(temp, enthalpy, label="raw")
        plt.plot(temp_ave, enthalpy_ave, label="smoothed")
        plt.axvline(T, color='r', linestyle='--')
        plt.axvline(temp_ave[liquid_ind], color='b', linestyle='--')
        plt.axvline(temp_ave[vapor_ind], color='b', linestyle='--')
        plt.xlabel("Temperature T/K")
        plt.ylabel("Enthalpy H/kcal/mol")
        plt.legend(loc="best")
        plt.savefig(path + f"{T}.png")
        plt.show()

    return T, H
'''


def main():

    var = {"datafile": "water_molecule.data",
           "paramfile": "H2O.TIP4P",
           "heat_time": 1000000,
           "low_temp": 300,
           "high_temp": 700,
           "seed": 432887}

    run_simulation(var)
    # T, H = analyze_simulation(path, plot=True)
    # print("Boiling points: ", T)
    # print("Vaporization enthalpy: ", H)


if __name__ == "__main__":
    main()
