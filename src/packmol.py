# author: Roy Kid
# contact: lijichen365@126.com
# date: 2023-06-19
# version: 0.0.1

from collections import defaultdict, namedtuple
import functools
import packmol
import numpy as np

CONSTRAIN_STYLE_RAW = [
    ['fixed', ('x', 'y', 'z', 'a', 'b', 'g'), []],
    ['inside_cube', ('x_min', 'y_min', 'z_min', 'd'), []],
    ['inside_box', ('x_min', 'y_min', 'z_min', 'x_max', 'y_max', 'z_max'), []],
    ['inside_sphere', ('x', 'y', 'z', 'r', 'd'), []],
    ['inside_ellipsoid', ('a1', 'b1', 'c1', 'a2', 'b2', 'c2', 'd'), []],
    ['outside_cube', ('x_min', 'y_min', 'z_min', 'd'), []],
    ['outside_box', ('x_min', 'y_min', 'z_min', 'x_max', 'y_max', 'z_max'), []],
    ['outside_sphere', ('x', 'y', 'z', 'r', 'd'), []],
    ['outside_ellipsoid', ('a1', 'b1', 'c1', 'a2', 'b2', 'c2', 'd'), []],
    ['outside_cylinder', ('a1', 'b1', 'c1', 'a2', 'b2', 'c2', 'd', 'l'), []],
    ['over_plane', ('a', 'b', 'c', 'd'), []],
    ['over_xygauss', ('a1', 'b1', 'a2', 'b2', 'c', 'h'), []],
    ['below_plane', ('a', 'b', 'c', 'd'), []],
    ['below_xygauss', ('a1', 'b1', 'a2', 'b2', 'c', 'h'), []]
]

CONSTRAIN_STYLE = dict(zip(map(lambda x: x[0], CONSTRAIN_STYLE_RAW), lambda x: {'definition': namedtuple(x[0], x[1]), 'check': x[2]}))


def def_constrain(style, restriction, params):
    definition = CONSTRAIN_STYLE[style][restriction]["definition"]
    constrain = definition(**params)
    return constrain


class Structure:
    def __init__(
        self,
        name,
        xyz,
        connect,
        nmols,
        constrain,
        n_loop=0,
        nloop0=0,
        constrain_rotation=(0, 0, 0),
    ):
        self.name = name
        self.xyz = xyz
        self.connect = connect
        self.nmols = nmols
        self.n_loop = n_loop
        self.nloop0 = nloop0
        self.constrain = constrain
        self.constrain_rotation = constrain_rotation

    @property
    def nTotalAtoms(self):
        return self.natoms * self.nmols
    
    @property
    def natoms(self):
        return len(self._xyz)


def def_structure(name, xyz, connect, nmols, constrain=None, n_loop=0, nloop0=0):
    return Structure(name, xyz, connect, nmols, constrain, n_loop, nloop0)


class Packmol:
    def __init__(self):
        # Printing title

        packmol.title()

        self._structures = []

    def add_structure(self, structure):
        if structure.name in self._structures:
            raise Exception(f"Structure name {structure.name} has been used!")

        self._structures.append(structure)

    def pack(
        self,
        seed=1234,
        tol=1e-4,
        randominitialpoint=False,
        check=False,
        writebad=False,
        movebadrandom=False,
        chkgrad=False,
        writeout=10,
        maxit=20,
        nloop=0,
        nloop0=0,
        discale=1.1,
        sidemax=1000,
        precision=1e-2,
        movefrac=0.05,
        fbins=3.0,
        add_amber_ter=False,
        amber_ter_preserve=False,
        add_box_sides=0,
        add_sides_fix=0.0,
        short_tol_dist=0.0,
        short_tol_scale=0.0,
        avoid_overlap=True,
        packall=False,
        use_short_tol=False,
        writecrd=False,
        iprint1=2,
        iprint2=2,
    ):
        # inside_structure = .false. # (input) if inside structure block

        # ! Getting random seed and optional optimization parameters if set

        if seed == -1:
            seed = packmol.random.seed_from_time(seed)
        print(" Seed for random number generator: ", seed)
        packmol.random.init_random_number(seed)

        if randominitialpoint:
            randini = True
        else:
            randini = False

        if precision != 1e-2:
            print(f" Optional precision set: {precision}")

        if movefrac != 0.05:
            print(f" Optional movefrac set: {movefrac}")

        if movebadrandom:
            print(f" Will move randomly bad molecules (movebadrandom)")

        if writeout != 10:
            print(f" Output frequency: {writeout}")

        if discale != 1.1:
            print(f" Optional initial tolerance scale: {discale}")

        if sidemax != 1000:
            print(f" User set maximum system dimensions: {sidemax}")

        if fbins != 3.0:
            print(f" User set linked-cell bin parameter: {fbins}")

        if add_amber_ter:
            print(" Will add the TER flag between molecules. ")

        if amber_ter_preserve:
            print(" TER flags for fixed molecules will be kept if found. ")

        if avoid_overlap:
            print(" Will avoid overlap to fixed molecules at initial point. ")
        else:
            print(" Will NOT avoid overlap to fixed molecules at initial point. ")

        if packall:
            print(" Will pack all molecule types from the beginning. ")

        if use_short_tol:
            print(" Will use a short distance penalty for all atoms. ")

        if writecrd:
            print(" Will write output also in CRD format ")

        if add_box_sides:
            print(" Will print BOX SIDE informations. ")
            print(" Will sum ", add_sides_fix, " to each side length on print")

        if iprint1 != 2:
            print(f" Optional printvalue 1 set: {iprint1}")
        if iprint2 != 2:
            print(f" Optional printvalue 2 set: {iprint2}")

        # --- packmol-python not implement ---
        # * Checking for the name of the output file to be created
        # * Reading structure files
        # * Reading constrain
        # - - - - - - - - - - - - - - - - - - -

        # Reading the structures
        ntype = len(self._structures)  # number of structure type
        packmol.set_ntype(ntype)

        # Setting the vectors for the number of GENCAN loops
        if nloop == 0:
            nloop_all = 200 * ntype
            nloop = nloop_all
        else:
            nloop_all = nloop
        print(" Maximum number of GENCAN loops for all molecule packing: ", nloop_all)

        nloop_type = np.zeros(ntype, dtype=int)
        for itype, struc in enumerate(self._structures):
            if nloop_type[itype] == 0:
                nloop_type[itype] = nloop_all
            else:
                print(
                    " Maximum number of GENCAN loops for type: ",
                    itype,
                    ": ",
                    nloop_type[itype],
                )

        # TODO: register nloop_type

        # nloop0 are the number of loops for the initial phase packing
        if nloop0 == 0:
            nloop0 = 20 * ntype
        else:
            print(
                " Maximum number of GENCAN loops-0 for all molecule packing: ", nloop0
            )

        nloop0_type = np.zeros(ntype, dtype=int)
        for itype, struc in enumerate(self._structures):
            if nloop0_type[itype] == 0:
                nloop0_type[itype] = nloop0
            else:
                print(
                    " Maximum number of GENCAN loops-0 for type: ",
                    itype,
                    ": ",
                    nloop0_type[itype],
                )

        # TODO: register nloop0_type

        # Reading the constrain that were set

        nrest = 0
        for struc in self._structures:
            if struc.constrain:
                nrest += 1
        
        print(' Total number of restrictions: ', nrest)

        # Getting the tolerance
        dism = tol
        print(' Distance tolerance: ', dism)

        # Reading, if defined, the short distance penalty parameters
        if short_tol_dist:
            if short_tol_dist > dism:
                print(' ERROR: The short_tol_dist parameter must be smaller than the tolerance. ')
            print(' User defined short tolerance distance: ', short_tol_dist)
            short_tol_dist = short_tol_dist ** 2
        else:
            short_tol_dist = dism / 2

        if short_tol_scale != 3.0:
            if short_tol_scale < 0.0:
                print(' ERROR: The short_tol_scale parameter must be positive. ')
            print(' User defined short tolerance scale: ', short_tol_scale)
        else:
            short_tol_scale = 3.0

        for itype, struc in enumerate(self._structures):

            print(' Number of molecules of type ', itype, ': ', struc.number)



        packmol.set_tolerance(tol, short_tol_dist, short_tol_scale)


        for i, struc in enumerate(self._structures.values()):
            packmol.set_nmol_natoms(
                i, struc.number, struc.natoms, struc.n_loop, struc.nloop0
            )

        packmol.allocate_compute_data()

        packmol.set_linked_cell(self._bins)
        # set constrain


        # Put molecules in their center of mass

        packmol.cenmass()

        # Writting some input data
        # summary of structures
        nmols = 0
        natoms = 0
        for key, value in self._structures.items():
            print(f" Number of molecules of {key}: {value.number}")
            nmols += value.number
            natoms += value.natoms

        print(f" Total number of molecules: {nmols}")
        print(f" Total number of atoms: {natoms}")

        # Put fixed molecules in the specified position
        for i, struc in enumerate(self._structures.values()):
            packmol.fix_mol(
                i,
            )

        # Checking if restart files will be used for each structure or for the whole system

        # Start time computation
        # time0 =
