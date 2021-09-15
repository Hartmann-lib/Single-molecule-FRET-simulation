import numpy as np

def density_estimator_2D(x, y, tau = 0.05):
    """
    density estimator around an individual (x,y) coordinate using a kernel density estimator (KDE) as described by 
    Tomov et al., Biophysical Journal (2012)

    x,y = array of events for desnity calculation
    tau = 1/e decay length of the kernel

    return density array len(x)

    09.15.2021 | Andreas Hartmann
    """

    density = np.zeros(len(x))

    for i in range(len(x)):

        density[i] = np.sum(np.exp(-np.absolute(x[i] - x)/tau)*np.exp(-np.absolute(y[i] - y)/tau))

    return density