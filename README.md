# Hierarchical Python Jobs (hrcl_jobs)

# Installation
```bash 
pip install hrcl-jobs
```
**NOTE** if you are using macos, you might need to do the following to install with pip
```bash 
brew install mpich
sudo find / -name mpicc
```
And then run the pip install with the path to your mpicc
```bash
env MPICC=/yourpath/mpicc pip3 install hrcl-jobs
```

## Features To Add
### QCElemental + BSExchange for NBF
```
I think the demo I cooked up for the qcsk talk might be handy here:
import json
import qcelemental as qcel
import basis_set_exchange as bse

mol = qcel.models.Molecule.from_data("""
C 0 0 0
O 0 0 1.2
H 0 -1.0 -0.3
H 0  1.0 -0.3
""")
print(mol)
#> Molecule(name='CH2O', formula='CH2O', hash='2af8547')

# get all elements from BSE
sbas = bse.get_basis("cc-pvdz", elements=mol.symbols.tolist(), fmt="qcschema")
# prepare schema specialized for molecule
bas = qcel.models.BasisSet(**json.loads(sbas), 
          atom_map=["c_cc-pVDZ", "o_cc-pVDZ", "h_cc-pVDZ", "h_cc-pVDZ"])
print(f"{bas.name=} {bas.nbf=}")
#> bas.name='cc-pVDZ' bas.nbf=26
the above gets nbf applied to a molecule and https://github.com/MolSSI/QCElemental/blob/master/qcelemental/models/molecule.py#L1181 mol.nelectrons gets occupancy, and from those you can get nocc/nvir
molecule.py
    def nelectrons(self, ifr: int = None) -> int:
```
