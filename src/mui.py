import mui4py
from mpi4py import MPI
import numpy as np


class MUI:

    def __init__(self, nodes, dt):
        self.nodes = nodes
        self.dt = dt

        # Interface setup
        MUI_COMM_WORLD = mui4py.mpi_split_by_app()

        self.MPI_COMM_WORLD = MPI.COMM_WORLD

        dims = 1
        config = mui4py.Config(dims, mui4py.FLOAT64)
        
        self.solverNum = self.MPI_COMM_WORLD.Get_rank()
        self.numSolvers = self.MPI_COMM_WORLD.Get_size()


        if self.numSolvers == 1:
            return None

        if self.solverNum == 0:
            iface =  ["ifs1"]
        elif self.solverNum == self.numSolvers-1:
            iface =  ["ifs" + str(self.solverNum)]
        else:
            iface = ["ifs"+str(self.solverNum), "ifs"+str(self.solverNum+1)]

        domain = "Solver" + str(self.solverNum)
        unifaces = mui4py.create_unifaces(domain, iface, config)

        if self.solverNum != 0:
            self.leftUniface = unifaces["ifs" + str(self.solverNum)]
            self.leftUniface.set_data_types({"temp": mui4py.FLOAT64,
                                     "weight": mui4py.FLOAT64,
                                     "coupling": mui4py.FLOAT64,
                                     "flux": mui4py.FLOAT64})
    
    
        if self.solverNum != self.numSolvers-1:
            self.rightUniface = unifaces["ifs" + str(self.solverNum+1)]

            self.rightUniface.set_data_types({"temp": mui4py.FLOAT64,
                                     "weight": mui4py.FLOAT64,
                                     "flux": mui4py.FLOAT64,
                                     "coupling": mui4py.FLOAT64,
                                     "nodes":mui4py.INT})

        # Use MPI allreduce to find the solver with the smallest dt and the most nodes
        self.minDt = self.findSuperlativeParameters()
        cutoff = 0
        sigma = 1
        if self.solverNum == 0:
            cutoff = 5
            sigma  = 1.0
        else:
            cutoff = 5
            sigma  = 1.0
            
        self.s_sampler = mui4py.SamplerPseudoNearestNeighbor(0.5)
        self.t_sampler = mui4py.TemporalSamplerExact()


        ## Point array for fetching using fetch_many and push_many
        # needs to be normalised to the maximum number nodes for any solver
        i = 0
        dy = 1/self.nodes
        self.points = np.zeros((self.nodes, 1))
        point2dList = []
        while i < self.nodes:
            self.points[i] = [i*dy]
            if self.solverNum == 0:
                point2dList.append(self.rightUniface.Point([i*dy]))
            elif self.solverNum == 1:
                point2dList.append(self.leftUniface.Point([i*dy]))
            i += 1


        #Define parameters of the RBF sampler
        #rSampler = 0.8                           # Define the search radius of the RBF sampler (radius size should be balanced to try and maintain)
        #basisFunc = 0                                 # Specify RBF basis function 0-Gaussian; 1-WendlandC0; 2-WendlandC2; 3-WendlandC4; 4-WendlandC6
        #conservative = True                             # Enable conservative OR consistent RBF form
        #cutOff = 1e-10                                   # Cut-off value for Gaussian RBF basis function
        #smoothFunc = False                              # Enable/disable RBF smoothing function during matrix creation
        #generateMatrix = True                             # Enable/disable writing of the matrix (if not reading)
        #cgSolveTol = 1e-7                              # Conjugate Gradient solver tolerance
        #cgMaxIter = 500                                # Conjugate Gradient solver maximum iterations (-1 = value determined by tolerance)
        #preconditioner = 1                             # Preconditioner of Conjugate Gradient solver
        #pouSize = 40                                   # RBF Partition of Unity patch size
        #rbfMatrixFolder = "rbfCoarseMatrix" + str(self.solverNum) # Output folder for the RBF matrix files

        
    

        #self.rbf_sampler = mui4py.SamplerRbf(rSampler, point2dList, basisFunc, conservative, smoothFunc, generateMatrix, rbfMatrixFolder, cutOff, cgSolveTol, cgMaxIter, pouSize, preconditioner, MUI_COMM_WORLD)


    def findSuperlativeParameters(self):
        sendbuf = np.zeros(1)
        sendbuf[0] = self.dt
        recvbuf = np.zeros(1)
        self.MPI_COMM_WORLD.Allreduce([sendbuf, MPI.DOUBLE], [recvbuf, MPI.DOUBLE], op=MPI.MIN)
        minDt = recvbuf[0]

        sendbuf = np.zeros(1)


        return minDt


    def findBoundaryTypes(self, coef):
        rightBoudaryDirich = None
        leftBoundaryDirich = None
        if self.solverNum > 0:
            self.leftUniface.push("coupling", [0], coef)
            self.leftUniface.commit( 0 )
            cN = self.leftUniface.fetch("coupling", [0], 0, self.s_sampler, self.t_sampler)
            leftBoundaryDirich = (coef/cN < 1)
    
        if self.solverNum < self.numSolvers-1:
            self.rightUniface.push("coupling", [0], coef)
            self.rightUniface.commit( 0 )
            cN = self.rightUniface.fetch("coupling", [0], 0, self.s_sampler, self.t_sampler)
            rightBoudaryDirich = coef/cN <= 1

            
        return (leftBoundaryDirich, rightBoudaryDirich)


    def pushRight(self, vals, data="temp"):
        self.rightUniface.push_many(data, self.points, vals)
    
    def commitRight(self, time):
        self.rightUniface.commit( time )

    def pushLeft(self, vals, data="temp"):
        self.leftUniface.push_many(data, self.points, vals)
    
    def commitLeft(self, time):
        self.leftUniface.commit( time )
            

    def fetchRightPrev(self, time, data="temp"):
        vals = self.leftUniface.fetch_many(data, self.points, time, 
                                       self.s_sampler, self.t_sampler)
        # forget to save memory
        self.leftUniface.forget( time-self.dt)
        
        return vals
    
    def fetchLeftNext(self, time, data="temp"):
        vals = self.rightUniface.fetch_many(data, self.points, time, 
                                       self.s_sampler, self.t_sampler)
        # forget to save memory
        self.rightUniface.forget( time-self.dt)
        #print( vals )
        return vals
