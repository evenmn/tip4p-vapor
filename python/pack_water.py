from molecular_builder import pack_water
from molecular_builder.geometry import CubeGeometry

# pack liquid
geometry = CubeGeometry(length=22.586, center=(11.293, 11.293, 11.293))
water = pack_water(nummol=384, geometry=geometry)
water.write('liquid.data', format="lammps-data")

# pack vapor
geometry = CubeGeometry(length=191., center=(95.5, 95.5, 95.5))
water = pack_water(nummol=128, geometry=geometry)
water.write('vapor.data', format="lammps-data")
