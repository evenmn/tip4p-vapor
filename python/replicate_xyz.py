import numpy as np
import pandas as pd
from io import StringIO


class ReplicateXYZ:
    def __init__(self, filename):
        """Identifiers for places in the log file
        """
        if hasattr(filename, "read"):
            logfile = filename
        else:
            logfile = open(filename, 'r')
        self.read_file(logfile)

    def read_file(self, filename):
        """Read XYZ file and store contents in a Pandas DataFrame
        """
        # first line contains number of atoms
        self.numatom = int(filename.readline().split()[0])
        # second line contains a comment
        self.comment = filename.readline()[:-3]
        # rest of the lines contain coordinates structured Element X Y Z
        string = "Element X Y Z \n" + filename.read()
        self.contents = pd.read_table(StringIO(string), sep=r'\s+')

    def replicate(self, nx, ny, nz):
        """Replicate initial file nx times in x-direction, ny times in
        y-direction and nz times in z-direction
        """
        maxx = self.contents['X'].max()
        maxy = self.contents['Y'].max()
        maxz = self.contents['Z'].max()
        minx = self.contents['X'].min()
        miny = self.contents['Y'].min()
        minz = self.contents['Z'].min()
        contents_list = []
        numreplicate = 0
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    new_df = self.contents.copy()
                    new_df['X'] += i * (maxx - minx)
                    new_df['Y'] += j * (maxy - miny)
                    new_df['Z'] += k * (maxz - minz)
                    contents_list.append(new_df)
                    numreplicate += 1
        self.numatom *= numreplicate
        self.contents = pd.concat(contents_list)

    def write(self, filename):
        """Write to new XYZ-file
        """
        np.savetxt(filename, self.contents.values, fmt='%s',
                   header=f"{self.numatom}\n{self.comment}", comments="")


if __name__ == "__main__":
    obj = ReplicateXYZ("../tip4p_cassandra/nvt.inp.xyz")
    obj.replicate(1, 1, 2)
    obj.write("../tip4p_cassandra/gemc_nvt/replicated.xyz")
