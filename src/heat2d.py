import numpy as np
import matplotlib.pyplot as plt
from temperatureSolver.src.args import Args
from temperatureSolver.src.mui import MUI

class Heat2d:
    def __init__(self, time=20, couplingMethod ='dirichNeum'):
        # Parse solver arguments
        args = Args().args

        self.height = args.height # geometry
        self.width = args.length # geometry
        self.thickness = args.thickness #geometry
        self.time = time # simulation time
        self.nodes = args.nodes # mesh res
        self.alpha = args.alpha # diffusivity
        self.kappa = args.kappa   # conductivity
        self.couplingMethod = couplingMethod

        # derived properties
        # we assume the boundary for coupling has zero width
        self.dx = self.width/(self.nodes-1)
        self.dy = self.height/(self.nodes)

        # Define dt according to Courant-Friedrichs-Lewy condition in 2 dimensions
        # (https://en.wikipedia.org/wiki/Courant%E2%80%93Friedrichs%E2%80%93Lewy_condition)  
        self.dt = min(self.dx**2/(2*self.alpha), self.dy**2/(2*self.alpha))  

        self.cellArea = self.dy*self.thickness

        # create MUI interface 
        self.mui = MUI( self.nodes, self.dt)

        # get solverNum and numSolvers from mui
        self.solverNum = self.mui.solverNum
        self.numSolvers = self.mui.numSolvers
        
        # if doing multisolver create mui interface and sync dt and alpha with the other interfaces
        if self.numSolvers > 1:
            self.dt = self.mui.minDt

        print("Solver: {:d}, Number of Solvers: {:d}".format(self.solverNum, self.numSolvers))

        print("Created heat solver {:d} with size {:f}m x {:f}m, {:d} nodes, diffusivity {:10f} and kappa {:f}".format(self.solverNum, self.height, self.width, self.nodes**2, self.alpha, self.kappa))

        print("Using coupling method " + self.couplingMethod)
        
        if self.couplingMethod == 'dirichNeum':
            # decide which boundary will be Neumann and which will be dirichlett
            c = self.nodes*self.cellArea*self.kappa/self.dx
            # boolean values, if True then boundary type is dirichlet else neumann
            self.leftBoundaryDirich, self.rightBoundaryDirich = self.mui.findBoundaryTypes(c)
        elif self.couplingMethod == 'linearInterpolation':
            self.kDelta = self.kappa/self.dx


        ## value of "0" temp  
        zeroK = 273.15

        # Temperature array
        self.T = np.zeros((self.nodes, self.nodes))
        self.T.fill(zeroK)

        # Visualisation
        fig, self.axis = plt.subplots()
        self.pcm = self.axis.pcolormesh(self.T, cmap=plt.cm.jet, vmin=zeroK, vmax = zeroK+100)
        plt.colorbar(self.pcm, ax=self.axis)

        # draw gridlines
        #self.axis.grid(which='major', axis='both', linestyle='-', color='0.8', linewidth=0.5)
        #self.axis.set_xticks(np.arange(0, self.nodes, 1))
        #self.axis.set_yticks(np.arange(0, self.nodes, 1))

        # setup for heat eq with numpy
        self.zerosX = np.full((1, self.nodes), zeroK)
        self.zerosY = np.zeros((self.nodes, 1))
    
        # set default BC
        if self.solverNum == 0:
            self.setLeftBoundaryCondition('temp', zeroK+100)

        if self.solverNum == self.numSolvers-1:
            self.setRightBoundaryCondition('temp', zeroK)
 

    def initialiseTempField(self, val: float):
        self.T[:, :] = val

    def setLeftBoundaryCondition(self, type: str, val: float):
        if type == 'flux':
            self.leftBoundaryType = 'flux'
            self.leftBoundaryValue = (np.full(shape=(1, self.nodes), fill_value=val, dtype=np.float64)*self.dx)/(self.kappa*self.height*self.thickness)
        elif type == 'temp':
            self.leftBoundaryType = 'temp'
            self.leftBoundaryValue = np.full(shape=(1, self.nodes), fill_value=val, dtype=np.float64)
            
        else:
            raise ValueError('Boundary condition can have type \'temp\' or \'flux\', not ' + type + ".")


    def setRightBoundaryCondition(self, type: str, val: float):
        if type == 'flux':
            self.rightBoundaryType = 'flux'
            self.rightBoundaryValue = (np.full(shape=(1, self.nodes), fill_value=val, dtype=np.float64)*self.dx)/(self.kappa*self.height*self.thickness)
        elif type == 'temp':
            self.rightBoundaryType = 'temp'
            self.rightBoundaryValue = np.full(shape=(1, self.nodes), fill_value=val, dtype=np.float64)
            
        else:
            raise ValueError('Boundary condition can have type \'temp\' or \'flux\', not ' + type + ".")

    def setColorMapScale(self, min, max):
        self.pcm.set_clim(min, max)
    


    def enforceBoundaryConditionsNeumDirich(self, time):
        # X
        # set left boundary based on heat flux of previous solver (Neumann)
        if self.solverNum > 0:
            if self.leftBoundaryDirich:
                # set left boundary as T[-1, : ] of previous solver (Dirichlet)
                q = self.cellArea*self.kappa*(self.T[0, : ]-self.T[1, : ])/self.dx 
                self.T[0, :] = self.mui.fetchRightPrev(time, data="temp")  
                self.mui.pushLeft(q*self.nodes, data="flux")
                self.mui.commitLeft( time )

            else:
                # set left boundary based on heat flux of previous solver (Neumann)
                self.mui.pushLeft(self.T[0, : ])
                self.mui.commitLeft( time )
                q = self.mui.fetchRightPrev(time, data="flux")/self.nodes
                self.T[0, :] = self.T[1, :] - (q/self.kappa * self.dx/self.cellArea)
        # if we are leftmost solver enforce initial condition
        else:
            if self.leftBoundaryType == 'temp':
                self.T[0, :] = self.leftBoundaryValue
            else:
                self.T[0, :] = self.leftBoundaryValue + self.T[1, :]

        # set right boundary as T[0, : ] of next solver (Dirichlet)
        if self.solverNum != self.numSolvers-1:
            if self.rightBoundaryDirich:
                # set right boundary as T[0, : ] of next solver (Dirichlet)
                q = self.cellArea/self.dx * self.kappa*(self.T[-1, : ]-self.T[-2, : ])
                self.T[-1, :] = self.mui.fetchLeftNext(time)
                self.mui.pushRight(q*self.nodes, data="flux")
                self.mui.commitRight( time )
            else:
                # set right boundary based on heat flux of next solver (Neumann)
                self.mui.pushRight(self.T[-1, : ], data="temp")
                self.mui.commitRight( time )
                q = self.mui.fetchLeftNext(time, data="flux")/self.nodes
                self.T[-1, :] = self.T[-2, :] - (q/self.kappa * self.dx/self.cellArea)
        # if we are the rightmost solver enforce initial condition
        else:
            if self.rightBoundaryType == 'temp':
                self.T[-1, :] = self.rightBoundaryValue
            else:
                self.T[-1, :] = self.rightBoundaryValue + self.T[-2, :]
        #Y
        # set gradient condition y boundaries, for now with 0 flux
        yFlux = 0
        self.T[:, 0] = self.T[:, 1] - (yFlux/self.kappa * self.dy/(self.dx*self.thickness))
        self.T[:, -1] = self.T[:, 1] - (yFlux/self.kappa * self.dy/(self.dx*self.thickness))

    def enforceBoundaryConditionsLI(self, time):
        # X
        # set left boundary based on heat flux of previous solver (Neumann)
        if self.solverNum > 0:
            ## push T[1, :] (left boundary)
            self.mui.pushLeft(self.T[1, : ])
            self.mui.leftUniface.push("weight", [0], self.kDelta)
            self.mui.commitLeft( time )

            nbrKDelta = self.mui.fetchRightPrev(time, data="weight")[0]
            boundaryWeight = nbrKDelta / (nbrKDelta + self.kDelta)
            
            # Enforce boundary condition here
            self.T[0, :] = (1-boundaryWeight)*self.T[1, :] + boundaryWeight*self.mui.fetchRightPrev(time)
    
        # if we are leftmost solver enforce initial condition
        else:
            if self.leftBoundaryType == 'temp':
                self.T[0, :] = self.leftBoundaryValue
            else:
                self.T[0, :] = self.leftBoundaryValue + self.T[1, :]

        # set right boundary as T[1, : ] of next solver (Dirichlet)
        if self.solverNum != self.numSolvers-1:

            self.mui.pushRight(self.T[-2])
            self.mui.rightUniface.push("weight", [0], self.kDelta)
            self.mui.commitRight( time )
            
            nbrKDelta = self.mui.fetchLeftNext(time, data="weight")[0]
            boundaryWeight = nbrKDelta/(nbrKDelta+self.kDelta)

            #Enforce boundary condition
            self.T[-1, :] = boundaryWeight*self.mui.fetchLeftNext(time) + (1-boundaryWeight)*self.T[-2, :]
            
        # if we are the rightmost solver enforce initial condition
        else:
            if self.rightBoundaryType == 'temp':
                self.T[-1, :] = self.rightBoundaryValue
            else:
                self.T[-1, :] = self.rightBoundaryValue + self.T[-2, :]
        #Y
        # set gradient condition y boundaries, for now with 0 flux
        yFlux = 0
        self.T[:, 0] = self.T[:, 1] - (yFlux/self.kappa * self.dy/(self.dx*self.thickness))
        self.T[:, -1] = self.T[:, 1] - (yFlux/self.kappa * self.dy/(self.dx*self.thickness))


    def calculateHeatEquation(self, time, animate = False):
        ## Idea: Do everything at once i.e xComponent =(dt/dx^2) Txplus1 + -2T + Txminus1       
        # set up shifted versions of T (xComponent)
        if self.couplingMethod == 'dirichNeum':
            self.enforceBoundaryConditionsNeumDirich(time)
        else:
            self.enforceBoundaryConditionsLI(time)

        Txplus1 = self.T[2:, :]
        Txminus1 = self.T[:-2, :]
    
        xComponent = Txplus1 - 2*self.T[1:-1, :] + Txminus1
        xComponent *= self.dt/(self.dx**2)

        # yComponent
        Typlus1 = self.T[:, 2:]
        Tyminus1 = self.T[:, :-2]

        yComponent = Typlus1 -2*self.T[:, 1:-1] + Tyminus1
        yComponent *= self.dt/(self.dy**2)

        self.T[1:-1, 1:-1] += self.alpha*(yComponent[1:-1, :] + xComponent[:, 1:-1])
    
        if animate:
            self.pcm.set_array(self.T.T)
            self.axis.set_title("Temperature at time t: {:.3f}s".format(time))
            plt.pause(0.005)

    def plotTemperature(self):
        self.pcm.set_array(self.T.T)
        self.axis.set_title("Temperature at time t: {:.3f}s".format(self.time))


        fig, ax = plt.subplots()

        xRange = np.linspace(self.solverNum*self.width, (self.solverNum+1)*self.width, self.nodes)
        tempData = np.average(self.T, axis=1)

        slope, intercept = np.polyfit(xRange, tempData, 1)

        ax.plot(xRange, tempData)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("Temp (K)")
        ax.set_title("Temp from Solver {:d}. Equation of line: T = {:.3f}x + {:.2f}".format(self.solverNum, slope, intercept))

        
        print("Solver {:d}       Left boundary has temperature: {:3f}K         Right boundary has temperature: {:3f}K".format(self.solverNum, np.average(self.T[0]), np.average(self.T[-1])))

        print((self.T[:, int(self.nodes/2)]+self.T[:, int((self.nodes/2)+1)])/2)

        plt.show()

    def getInterfaceTemperature(self):
        return (np.average(self.T[0]), np.average(self.T[-1]))