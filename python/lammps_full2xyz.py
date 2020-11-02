import numpy as np
from ase.io import read, write

atoms = read("../data/lammps_full.data", format="lammps-data")

format = 'OHH'
molsize = len(format)
numatom = len(atoms.get_positions())
elements = np.tile(list(format), numatom // molsize)

atoms.set_chemical_symbols(elements)
atoms.write("../data/lattice.xyz", format="xyz")
