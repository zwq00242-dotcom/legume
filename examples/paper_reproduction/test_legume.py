import numpy as np
import legume

from legume import Lattice, PhotCryst, Circle, GuidedModeExp


print("=== legume environment test ===")
print("legume imported successfully")


# 1. Square lattice
lattice = Lattice("square")

# 2. Simple photonic-crystal slab
# Lower/upper cladding: air, eps = 1
phc = PhotCryst(lattice, eps_l=1.0, eps_u=1.0)

# Slab: relative permittivity = 12, thickness = 0.5a
phc.add_layer(d=0.5, eps_b=12.0)

# Circular air hole, radius = 0.2a
phc.add_shape(Circle(eps=1.0, r=0.2))

# 3. GME solver
gme = GuidedModeExp(phc, gmax=3.0)

# 4. Only two k-points for a very small test
kpoints = np.array([
    [0.0, 0.1],
    [0.0, 0.0]
])

# 5. Compute several eigenmodes
gme.run(
    kpoints=kpoints,
    gmode_inds=[0],
    numeig=5,
    compute_im=False,
    verbose=False
)

print("Computed normalized frequencies:")
print(gme.freqs)

print("=== test finished successfully ===")
