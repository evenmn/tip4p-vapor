from gomc_wrapper import read, psfgen

# file names
configfile = "in.conf"
topofile = "topology_tip4p.inp"
paramfile = "Par_TIP4P-2005_Charmm.inp"
pdb1 = "liquid_gomc2.pdb"
pdb2 = "vapor_gomc2.pdb"
psf1 = "liquid_gomc2.psf"
psf2 = "vapor_gomc2.psf"

# generate PSF files
psfgen(coordinates=pdb1, topology=topofile, genfile=psf1)
psfgen(coordinates=pdb2, topology=topofile, genfile=psf2)

# read config file into GOMC object and run
gomc = read(configfile)
gomc.set("Parameters", paramfile)
gomc.set("Coordinates", 0, pdb1)
gomc.set("Coordinates", 1, pdb2)
gomc.set("Structure", 0, psf1)
gomc.set("Structure", 1, psf2)
gomc.run(gomc_exec="GOMC_GPU_GEMC", num_procs=4, gomc_input="in.conf2")
