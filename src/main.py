from temperatureSolver.src.heat2d import Heat2d


def main(time, couplingMethod, testing=True):
    solver = Heat2d(time, couplingMethod)
    i = 0
    while i < solver.time:
        solver.calculateHeatEquation(i, animate=False)
        i += solver.dt
    if not testing:
        solver.plotTemperature()
    
    return solver.getInterfaceTemperature()

if __name__ == "__main__":
    main(time=100, couplingMethod="linearInterpolation", testing=False)
