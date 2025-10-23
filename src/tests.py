
import pytest
from main import main
from mpi4py import MPI


rank = MPI.COMM_WORLD.Get_rank()
size = MPI.COMM_WORLD.Get_size()


# time we run each solver for
time = 60

# Boundary Conditions
T0 = 373.15
T1 = 273.15

def anaylticalSolver(kappas, lengths,rank):
    s = 0
    for i in range(len(kappas)):
        s += lengths[i]/kappas[i]
    Q = (T0-T1)/s

    interfaceTemps = [T0]
    for i in range(1, len(kappas)+1): 
        T_theory = interfaceTemps[-1] - Q*lengths[i-1]/kappas[i-1]
        interfaceTemps.append(T_theory)

    return interfaceTemps[rank:rank+2]

def run_test(monkeypatch, couplingMethod, lengths, kappas, alphas, nodes):
   
    # run python solver with appropriate arguments
    monkeypatch.setattr("sys.argv", ["main.py", "--length", str(lengths[rank]), "--kappa", str(kappas[rank]),"--alpha", str(alphas[rank]),  "--nodes", str(nodes[rank])] )
    interfaceTemps = main(time, couplingMethod)

    #get theoretical result for this solver
    theoreticalInterfaceTemps = anaylticalSolver(kappas, lengths, rank)
    for i in range(2):
        assert interfaceTemps[i] == pytest.approx(theoreticalInterfaceTemps[i], 1e-2)


def test_simple_linearInterpolation(monkeypatch):
    '''
    same material, length and number of nodes. Linear Interpolation coupling.
    '''
    couplingMethod = "linearInterpolation"
    lengths = [0.025 for i in range(size)]
    kappas = [318 for i in range(size)]
    alphas = [1.27e-4 for i in range(size)]
    nodes = [20 for i in range(size)]

    run_test(monkeypatch, couplingMethod, lengths, kappas, alphas, nodes)

def test_simple_neumannDirichlet(monkeypatch):
    '''
    same material, length and number of nodes. neumann-dirichlett coupling.
    '''
    couplingMethod = "dirichNeum"
    lengths = [0.025 for i in range(size)]
    kappas = [318 for i in range(size)]
    alphas = [1.27e-4 for i in range(size)]
    nodes = [20 for i in range(size)]

    run_test(monkeypatch, couplingMethod, lengths, kappas, alphas, nodes)


def test_material_linearInterpolation(monkeypatch):
    '''
    same length and number of nodes. set 1st solver (rank 0) to different material (twice as conductive). Linear Interpolation coupling.
    '''
    couplingMethod = "linearInterpolation"
    lengths = [0.025 for i in range(size)]
    kappas = [318 for i in range(size)]
    alphas = [1.27e-4 for i in range(size)]
    nodes = [20 for i in range(size)]

    kappas[0] *= 2
    alphas[0] *= 2  

    run_test(monkeypatch, couplingMethod, lengths, kappas, alphas, nodes)


def test_material_neumannDirichlet(monkeypatch):
    '''
    same length and number of nodes. set 1st solver (rank 0) to different material (twice as conductive). neumann-dirichlett coupling.
    '''
    couplingMethod = "dirichNeum"
    lengths = [0.025 for i in range(size)]
    kappas = [318 for i in range(size)]
    alphas = [1.27e-4 for i in range(size)]
    nodes = [20 for i in range(size)]

    kappas[0] *= 2
    alphas[0] *= 2  

    run_test(monkeypatch, couplingMethod, lengths, kappas, alphas, nodes)

def test_length_linearInterpolation(monkeypatch):
    '''
    same material and number of nodes. set 1st solver (rank 0) to different length (2.5 times as long). Linear Interpolation coupling.
    '''
    couplingMethod = "linearInterpolation"
    lengths = [0.025 for i in range(size)]
    kappas = [318 for i in range(size)]
    alphas = [1.27e-4 for i in range(size)]
    nodes = [20 for i in range(size)]

    lengths[0] *= 2.5

    run_test(monkeypatch, couplingMethod, lengths, kappas, alphas, nodes)


def test_length_neumannDirichlet(monkeypatch):
    '''
    same length and number of nodes. set 1d solver (rank 0) to different length (2.5 times as long). neumann-dirichlett coupling.
    '''
    couplingMethod = "dirichNeum"
    lengths = [0.025 for i in range(size)]
    kappas = [318 for i in range(size)]
    alphas = [1.27e-4 for i in range(size)]
    nodes = [20 for i in range(size)]

    lengths[0] *= 2.5

    run_test(monkeypatch, couplingMethod, lengths, kappas, alphas, nodes)

def test_length_linearInterpolation(monkeypatch):
    '''
    same length and number of nodes. set 1st solver (rank 0) to different number of nodes (3 times more). Linear Interpolation coupling.
    '''
    couplingMethod = "linearInterpolation"
    lengths = [0.025 for i in range(size)]
    kappas = [318 for i in range(size)]
    alphas = [1.27e-4 for i in range(size)]
    nodes = [20 for i in range(size)]

    nodes[0] *= 3

    run_test(monkeypatch, couplingMethod, lengths, kappas, alphas, nodes)


def test_length_neumannDirichlet(monkeypatch):
    '''
    same material and length. set 1st solver (rank 0) to different number of nodes (3 times more) neumann-dirichlett coupling.
    '''
    couplingMethod = "dirichNeum"
    lengths = [0.025 for i in range(size)]
    kappas = [318 for i in range(size)]
    alphas = [1.27e-4 for i in range(size)]
    nodes = [20 for i in range(size)]

    nodes[0] *= 3

    run_test(monkeypatch, couplingMethod, lengths, kappas, alphas, nodes)
