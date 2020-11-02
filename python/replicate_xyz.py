import datetime
import numpy as np
import pandas as pd
from io import StringIO


class Convert:
    def __init__(self, filename, format=None, box=None):
        """Identifiers for places in the log file
        """

        self.masses = {}
        self.charges = {}
        self.angles = {}
        self.bonds = {}
        self.pair_coeffs = {}
        self.corners = {}

        self.num_bond_types = 0
        self.num_angle_types = 0
        self.num_box = 1

        self.comment = ""

        if hasattr(filename, "read"):
            logfile = filename
        else:
            logfile = open(filename, 'r')

        if format is None:
            format = filename.split('.')[-1]
        if format == 'xyz':
            self.read_xyz(logfile)
            self.element2id()
            self.atom_mol_link()
            self.find_corners()
            self.define_box_location()
        elif format == 'lammps-atomic':
            raise NotImplementedError
        elif format == 'lammps-full':
            raise NotImplementedError
        elif format == 'towhee_coord':
            self.read_towhee_coord(logfile)
        else:
            raise NotImplementedError(f"Cannot read file format {format}")

    def read_lammps_atomic(self, filename):
        pass

    def read_lammps_full(self, filename):
        pass

    def read_towhee_coord(self, filename, format='OHH'):
        string = "X Y Z \n" + filename.read()
        self.contents = pd.read_table(StringIO(string), sep=r'\s+')
        self.molsize = len(format)
        self.numatom = len(self.contents)
        elements = np.tile(list(format), self.numatom // self.molsize)
        self.contents['Element'] = elements

    def read_towhee_restart(self, filename):
        self.version = int(filename.readline().split()[0])
        self.seeds = filename.readline().split()
        self.numcycles, self.numboxes, self.num_mol_types = filename.readline().split()

    def read_xyz(self, filename):
        """Read XYZ file and store contents in a Pandas DataFrame
        """
        # first line contains number of atoms
        self.numatom = int(filename.readline().split()[0])
        # second line contains a comment
        self.comment = filename.readline()[:-3]
        # rest of the lines contain coordinates structured Element X Y Z
        string = "Element X Y Z \n" + filename.read()
        self.contents = pd.read_table(StringIO(string), sep=r'\s+')

    def element2id(self):
        """Convert element symbol to number id
        """
        elements = self.contents['Element']
        unique_elements, indices = np.unique(elements, return_inverse=True)
        self.contents['Sub_ID'] = indices + 1
        self.contents['ID'] = np.arange(1, len(self.contents)+1)
        self.num_atom_types = len(unique_elements)

    def find_corners(self):
        """Find corners, center and length of box,
        assuming it is orthorhombic
        """
        for dir in ['x', 'y', 'z']:
            self.corners[f'{dir}lo'] = self.contents[dir.upper()].min()
            self.corners[f'{dir}hi'] = self.contents[dir.upper()].max()
        self.lengthx = self.corners['xhi'] - self.corners['xlo']
        self.lengthy = self.corners['yhi'] - self.corners['ylo']
        self.lengthz = self.corners['zhi'] - self.corners['zlo']
        self.length = [self.lengthx, self.lengthy, self.lengthz]

        sumx = self.corners['xhi'] + self.corners['xlo']
        sumy = self.corners['yhi'] + self.corners['ylo']
        sumz = self.corners['zhi'] + self.corners['zlo']
        self.center = [sumx/2, sumy/2, sumz/2]

    def define_box_location(self):
        """Define which box the particles belong to
        """
        self.contents['Box_ID'] = np.ones(self.numatom) * self.num_box

    def atom_mol_link(self, order='chron', molsize=3, moltype=1):
        """Link atoms ids to molecule ids, ensures the
        molecules and atoms are sorted in desceding order
        """
        self.molsize = molsize
        if order == 'chron':
            mol_id = np.repeat(range(self.numatom//molsize), molsize)
            self.contents['Mol_ID'] = mol_id
            self.contents['Mol_type'] = np.ones(self.numatom) * moltype
        else:
            raise NotImplementedError("Only chronological ordering supported")

    def set_charge(self, type, charge):
        """Set charge of atom type
        """
        assert type < self.num_atom_types + 1
        assert type > 0
        self.charges[type] = charge

    def set_mass(self, type, mass):
        """Set mass of atom type
        """
        assert type < self.num_atom_types + 1
        assert type > 0
        self.masses[type] = mass

    def set_bond_coeff(self, a, b, coeff):
        """Set bond coeff
        """
        self.bonds[self.num_bond_types + 1] = [a, b, coeff]
        self.num_bond_types += 1

    def set_angle_coeff(self, a, b, c, coeff):
        """Set angle coeff
        """
        self.angles[self.num_angle_types + 1] = [a, b, c, coeff]
        self.num_angle_types += 1

    def set_pair_coeffs(self, id, sigma, epsilon):
        """Set force-field pair coeffs
        """
        self.pair_coeffs[id] = [sigma, epsilon]

    def list_charges(self):
        """Add column with charges to content data frame
        """
        charges = self.contents['Sub_ID']
        for i in range(1, self.num_atom_types + 1):
            charges = np.where(charges == i, float(self.charges[i]), charges)
        self.contents['Charge'] = charges

    def list_masses(self):
        """Add column with masses to content data frame
        """
        masses = self.contents['Sub_ID']
        for i in range(self.num_atom_types):
            masses = np.where(masses == i, float(self.masses[i]), masses)
        self.contents['Mass'] = masses

    def extract_bonds(self):
        """Create data frame containing all bonds
        """
        atom_types = self.contents['Sub_ID']
        atom_ids = self.contents['ID']
        bond_list = []
        for key, value in self.bonds.items():
            a = value[0]
            b = value[1]

            A = np.asarray(atom_types).reshape(-1, 3)
            B = np.asarray(atom_ids).reshape(-1, 3)

            D = np.where(A == a, B, np.nan)
            E = np.where(A == b, B, np.nan)

            D = D[:, ~np.all(np.isnan(D), axis=0)]
            E = E[:, ~np.all(np.isnan(E), axis=0)]

            D_ = np.tile(D, (1, E.shape[1]))
            E_ = np.repeat(E, D.shape[1], axis=1)

            F = np.asarray([D_, E_]).T

            idd = np.ones((F.shape[1], F.shape[0])) * key
            # g = np.arange(1, )
            fi = np.arange(F.shape[1])
            iff = np.repeat(fi[:,np.newaxis], 2, axis=1)

            concate = np.concatenate((iff[:,:,np.newaxis], idd[:,:,np.newaxis], F.swapaxes(0, 1)), axis=-1)
            concate = concate.reshape(-1, 4)
            df = pd.DataFrame(data=concate, columns=['Mol_ID', 'Bond_type', 'Atom_1', 'Atom_2'])
            bond_list.append(df)
        self.bond_df = pd.concat(bond_list)
        self.num_bonds = len(self.bond_df)

    def extract_angles(self):
        """Create data frame containing all angles
        """
        atom_ids = self.contents['ID']
        angle_list = []
        for key, value in self.angles.items():
            a = value[0]
            b = value[1]
            c = value[2]

            lst = [a, b, c]

            A_ = np.asarray(atom_ids).reshape(-1, 3)

            sorted = np.argsort(lst)
            A_sorted = A_[:, sorted]

            idd = np.ones(len(A_sorted)) * key
            iff = np.arange(1, len(A_sorted) + 1)

            concate = np.concatenate((iff[:,np.newaxis], idd[:,np.newaxis], A_sorted), axis=-1)
            df = pd.DataFrame(data=concate, columns=['Mol_ID', 'Angle_type', 'Atom_1', 'Atom_2', 'Atom_3'])
            angle_list.append(df)
        self.angle_df = pd.concat(angle_list)
        self.num_angles = len(self.angle_df)

    def add_velocity(self, temp=0):
        """Add velocity of the particles, given a system temperature
        """
        vel = np.random.normal(0, np.sqrt(temp), size=(self.numatom, 3))
        self.contents['VX'] = vel[:, 0]
        self.contents['VY'] = vel[:, 1]
        self.contents['VZ'] = vel[:, 2]

    def shift_negative_coordinates(self):
        """Sometimes Packmol gives negative coordinates. To use the coordinate
        file with Towhee, we need to shift where positions
        """
        X = self.contents['X']
        Y = self.contents['Y']
        Z = self.contents['Z']

        X_new = np.where(X > 0, X, 0)
        Y_new = np.where(Y > 0, Y, 0)
        Z_new = np.where(Z > 0, Z, 0)

        self.contents['X'] = X_new
        self.contents['Y'] = Y_new
        self.contents['Z'] = Z_new


    def replicate(self, nx, ny, nz):
        """Replicate initial file nx times in x-direction, ny times in
        y-direction and nz times in z-direction
        """
        contents_list = []
        numreplicate = 0
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    new_df = self.contents.copy()
                    new_df['X'] += i * self.lengthx
                    new_df['Y'] += j * self.lengthy
                    new_df['Z'] += k * self.lengthz
                    contents_list.append(new_df)
                    numreplicate += 1
        self.numatom *= numreplicate
        self.contents = pd.concat(contents_list)

    def write_xyz(self, filename):
        """Write to new XYZ-file
        """
        df = self.contents[['Element', 'X', 'Y', 'Z']].copy()
        np.savetxt(filename, df.values, fmt='%s' + '%20.15f' * 3,
                   header=f"{self.numatom}\n{self.comment}", comments="")

    def write_lammps_atomic(self, filename):
        pass

    def write_lammps_full(self, filename):
        """Write LAMMPS data file, full format
        """
        now = datetime.datetime.now()
        with open(filename, 'w') as f:
            f.write(f"LAMMPS full data file. DATE: {now:%Y-%m-%d %H:%M:%S}\n\n")
            f.write(f"{self.numatom} atoms\n")
            f.write(f"{self.num_atom_types} atom types\n")
            f.write(f"{self.num_bonds} bonds\n")
            f.write(f"{self.num_bond_types} bond types\n")
            f.write(f"{self.num_angles} angles\n")
            f.write(f"{self.num_angle_types} angle types\n\n")
            f.write(f"{self.corners['xlo']:.8E} {self.corners['xhi']:.8E} xlo xhi\n")
            f.write(f"{self.corners['ylo']:.8E} {self.corners['yhi']:.8E} ylo yhi\n")
            f.write(f"{self.corners['zlo']:.8E} {self.corners['zhi']:.8E} zlo zhi\n\n")
            f.write("Masses\n\n")
            for key, value in self.masses.items():
                f.write(f"{key} {value}\n")
            f.write("\n")
            f.write("Pair Coeffs\n\n")
            for key, value in self.pair_coeffs.items():
                f.write(f"{key} {value[0]} {value[1]}\n")
            f.write("\n")
            f.write("Bond Coeffs\n\n")
            for key, value in self.bonds.items():
                f.write(f"{key} 0 {value[-1]}\n")
            f.write("\n")
            f.write("Angle Coeffs\n\n")
            for key, value in self.angles.items():
                f.write(f"{key} 0 {value[-1]}\n")
            f.write("\n")
            f.write("Atoms\n\n")
            df = self.contents[['ID', 'Mol_ID', 'Sub_ID', 'Charge', 'X', 'Y', 'Z']].copy()
            np.savetxt(f, df.values, fmt=f"%{int(np.log10(self.numatom)+2)}d "*3 + "   %.8E"*4)
            f.write("\n")
            f.write("Velocities\n\n")
            df = self.contents[['ID', 'VX', 'VY', 'VZ']].copy()
            np.savetxt(f, df.values, fmt=f"%{int(np.log10(self.numatom)+2)}d " + "   %.8E"*3)
            f.write("\n")
            f.write("Bonds\n\n")
            np.savetxt(f, self.bond_df.astype(int).values, fmt=f"%{int(np.log10(self.numatom)+2)}d ")
            f.write("\n")
            f.write("Angles\n\n")
            np.savetxt(f, self.angle_df.astype(int).values, fmt=f"%{int(np.log10(self.numatom)+2)}d ")

    def write_towhee_coord(self, filename):
        """Write Towhee input file, coord format
        """
        with open(filename, 'w') as f:
            df = self.contents[['X', 'Y', 'Z']].copy()
            np.savetxt(f, df.values, fmt="  %20.15f"*3)

    def write_towhee_restart(self, filename, pbc=(0, 0, 0)):
        """Write Towhee input file, restart format
        """
        seeds = [5054388, 14604618, 6176650, 7526479, 6525, 10097385, 9059353,
                 14349506, 535, 7287374, 12195841, 7272997, 5692437, 11292972,
                 1589479, 16351161, 14342867, 3500530, 14385737, 2924396,
                 11857489, 6765405, 12074244, 5940539, 3050519]
        version = 6
        numcycles = 0
        numboxes = 1
        nummoltypes = 1
        maxtranssingle = 0.5
        maxtranscom = 0.15305334
        maxrot = 0.672082
        maxvoldis = [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
        if type(pbc) is list or type(pbc) is tuple:
            pbc = np.asarray(pbc)
        length = np.asarray(self.length) + pbc
        hmatrix = np.diag(length)
        maxvoldis.extend(list(hmatrix))
        with open(filename, 'w') as f:
            f.write(f"\t\t{version}\n")
            f.write("\t" + "\t  ".join(map(str, seeds)) + "\n")
            f.write(f"\t{numcycles} \t\t{numboxes} \t\t{nummoltypes}\n")
            f.write(f"\t{maxtranssingle:20.15f}\n")
            f.write(f"\t{maxtranscom:20.15f}\n")
            f.write(f"\t{maxrot:20.15f}\n\n")
            for row in maxvoldis:
                f.write(f"\t{row[0]:20.15f} \t{row[1]:20.15f} \t{row[2]:20.15f}\n")
            f.write(f"\t{self.numatom // self.molsize}\n")
            f.write(f"\t{self.molsize}\n")
            f.write("\t  ".join(map(str, list(self.contents['Mol_type'].astype(int)))) + "\n")
            f.write("\t  ".join(map(str, list(self.contents['Box_ID'].astype(int)))) + "\n")
            df = self.contents[['X', 'Y', 'Z']].copy()
            np.savetxt(f, df.values, fmt="  %20.15f"*3)


if __name__ == "__main__":
    # convert xyz-file to lammps full format
    # obj = Convert("../src/gemc_towhee/towhee_coords", format="towhee_coord")
    # obj.write_xyz("../data/test.xyz")
    # stop

    obj = Convert("../src/coord_file/liquid_towhee.xyz")
    obj.shift_negative_coordinates()
    obj.write_xyz("../data/liquid_towhee_shifted.xyz")
    stop

    obj = Convert("../data/liquid.xyz")
    obj.set_charge(1, 0.5564)
    obj.set_charge(2, -1.1128)
    obj.list_charges()
    obj.set_mass(1, 1.00794)
    obj.set_mass(2, 15.9994)
    obj.set_pair_coeffs(1, 0, 0)
    obj.set_pair_coeffs(2, 0.1852, 3.1589)
    obj.set_bond_coeff(1, 2, 0.9572)
    obj.extract_bonds()
    obj.set_angle_coeff(1, 2, 1, 104.52)
    obj.extract_angles()
    obj.add_velocity(temp=0)
    obj.write_lammps_full("../data/liquid.data")
    obj.write_towhee_restart("../data/liquid.towhee")
