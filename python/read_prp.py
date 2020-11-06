import numpy as np
import pandas as pd
from io import StringIO


class File:
    """Class for handling Cassandra prp-files.
    Parameters
    ----------------------
    :param filename: path to lammps log file
    :type filename: string or file
    """
    def __init__(self, filename):
        # Identifiers for places in the log file
        if hasattr(filename, "read"):
            logfile = filename
        else:
            logfile = open(filename, 'r')
        self.read_file_to_dataframe(logfile)

    def read_file_to_dataframe(self, logfile):
        # read three first lines, which should be information lines
        logfile.readline()
        tmpString = logfile.readline()[1:]  # string should start with the kws
        self.keywords = tmpString.split()
        self.units = logfile.readline().split()[1:]
        contents = logfile.readlines()
        i = 0
        while i < len(contents):
            line = contents[i]
            if line.startswith("#"):
                break
            else:
                tmpString += line
                i += 1
        self.contents = pd.read_table(StringIO(tmpString), sep=r'\s+')

    def find(self, entry_name):
        return np.asarray(self.contents[entry_name])

    def get_keywords(self):
        """Return list of available data columns in the log file."""
        print(", ".join(self.keywords))

    @staticmethod
    def average(arr, window):
        """Average an array arr over a certain window size
        """
        if window == 1:
            return arr
        elif window > len(arr):
            raise IndexError("Window is larger than array size")
        else:
            remainder = len(arr) % window
            if remainder == 0:
                avg = np.mean(arr.reshape(-1, window), axis=1)
            else:
                avg = np.mean(arr[:-remainder].reshape(-1, window), axis=1)
        return avg


if __name__ == "__main__":
    window = 10000
    filenames = ["../src/gemc_cassandra/gemc_nvt.out.box1.prp",
                 "../src/gemc_cassandra/gemc_nvt.out.box2.prp"]
    phases = ["liquid", "vapor"]

    fileobjs = []
    dependents = []
    for filename in filenames:
        file = File(filename)
        fileobjs.append(file)
        dependents.append(file.average(file.find("MC_SWEEP"), window))

    import matplotlib.pyplot as plt
    keywords = ["Mass_Density", "Volume", "Nmols", "Enthalpy", "Pressure"]
    keywords = ["Pressure"]
    for keyword in keywords:
        for fileobj, dependent, phase in zip(fileobjs, dependents, phases):
            arr = fileobj.find(keyword)
            arr_avg = fileobj.average(arr, window)
            plt.plot(dependent, arr_avg, label=phase)
        plt.xlabel("Cycles")
        plt.ylabel(keyword)
        plt.legend(loc='best')
        plt.grid()
        plt.show()
