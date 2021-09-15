import numpy as np

# 15.09.2021 | Andreas Hartmann

def DonorOnly(nD0, nA_D0, TB):
    """
    Photon trajectory of a donor only molecule
    nD0 = count rate of the donor in absence of acceptor
    nA_D0 = count rate of the acceptor in absence of acceptor (spectral crosstalk of the donor)
    TB = burst duration
    return photon arrival times of the donor photon after donor excitation, acceptor photon after donor excitation, acceptor photon after acceptor excitation
    """

    iptGGGR = np.random.exponential(1/(nD0 + nA_D0), (round(10*(nD0 + nA_D0)*TB), 1))
        
    tGGGR = np.cumsum(iptGGGR) # arrival times
    tGGGR = tGGGR[tGGGR < TB]  

    boolChannel = np.random.binomial(1, nD0/(nD0 + nA_D0), len(tGGGR))

    tD = tGGGR[boolChannel == 1]
    tA = tGGGR[boolChannel == 0]
    tA0 = np.array([])

    return tD, tA, tA0
    
def FRET(nD, nA, nA0, TB):
    """
    Photon trajectory of a FRET molecule
    nD = count rate of the donor of a FRET molecule
    nA = count rate of the acceptor of a FRET molecule
    nA0 = count rate of the acceptor after acceptor excitation
    TB = burst duration
    return photon arrival times of the donor photon after donor excitation, acceptor photon after donor excitation, acceptor photon after acceptor excitation
    """

    iptGGGR = np.random.exponential(1/(nD + nA), (round(10*(nD + nA)*TB), 1))
    iptRR = np.random.exponential(1/nA0, (round(10*nA0*TB), 1))

    tGGGR = np.cumsum(iptGGGR) # arrival times
    tRR = np.cumsum(iptRR) # arrival times
    tGGGR = tGGGR[tGGGR < TB]  
    tRR = tRR[tRR < TB]  

    boolChannel = np.random.binomial(1, nD/(nD + nA), len(tGGGR))

    tD = tGGGR[boolChannel == 1]
    tA = tGGGR[boolChannel == 0]
    tA0 = tRR

    return tD, tA, tA0

def AcceptorOnly(nA0, TB):
    """
    Photon trajectory of a acceptor only molecule
    nA0 = count rate of the acceptor after acceptor excitation
    TB = burst duration
    return photon arrival times of the donor photon after donor excitation, acceptor photon after donor excitation, acceptor photon after acceptor excitation
    """   
            
    iptRR = np.random.exponential(1/nA0, (round(10*nA0*TB), 1))
        
    tRR = np.cumsum(iptRR) # arrival times
    tA0 = tRR[tRR < TB]  

    tD = np.array([])
    tA = np.array([])
    
    return tD, tA, tA0

def Macrotimes(boolFraction, TB, TBleachD, TBleachA, nD0, nA_D0, nA0, nD, nA):
    """
    Photon trajectory
    boolFraction = labeling stoichiometry: 1-> donor only; 2-> FRET molecule; 3-> acceptor only
    TB = burst duration
    TBleachD = bleaching time of the donor
    TBleachA = bleaching time of the acceptor
    nD0 = count rate of the donor in absence of acceptor
    nA_D0 = count rate of the acceptor in absence of acceptor (spectral crosstalk of the donor)
    nA0 = count rate of the acceptor after acceptor excitation
    nD = count rate of the donor of a FRET molecule
    nA = count rate of the acceptor of a FRET molecule   
    return photon arrival times of the donor photon after donor excitation, acceptor photon after donor excitation, acceptor photon after acceptor excitation
    """

    if boolFraction == 1: # donor only

        tD, tA, tA0 = DonorOnly(nD0, nA_D0, TB)

    elif boolFraction == 2: # double labeled FRET

        # acceptor bleaching
        if TBleachA < TB and TBleachD >= TB:

            tD_FRET, tA_FRET, tA0_FRET = FRET(nD, nA, nA0, TBleachA)   
            tD_D0, tA_D0, tA0_D0 = DonorOnly(nD0, nA_D0, TB-TBleachA)   
            
            tD = np.unique(np.concatenate((tD_FRET, tD_D0 + TBleachA)))
            tA = np.unique(np.concatenate((tA_FRET, tA_D0 + TBleachA)))
            tA0 = np.unique(np.concatenate((tA0_FRET, tA0_D0 + TBleachA)))

        # donor bleaching
        elif TBleachD < TB and TBleachA >= TB:

            tD_FRET, tA_FRET, tA0_FRET = FRET(nD, nA, nA0, TBleachD)   
            tD_A0, tA_A0, tA0_A0 = AcceptorOnly(nA0, TB-TBleachD)   
            
            tD = np.unique(np.concatenate((tD_FRET, tD_A0 + TBleachD)))
            tA = np.unique(np.concatenate((tA_FRET, tA_A0 + TBleachD)))
            tA0 = np.unique(np.concatenate((tA0_FRET, tA0_A0 + TBleachD)))

        # both bleaching within the burst duration
        elif TBleachD < TB and TBleachA < TB:

            if TBleachD > TBleachA:

                TB = TBleachD
            
                tD_FRET, tA_FRET, tA0_FRET = FRET(nD, nA, nA0, TBleachA)   
                tD_D0, tA_D0, tA0_D0 = DonorOnly(nD0, nA_D0, TB-TBleachA)   
            
                tD = np.unique(np.concatenate((tD_FRET, tD_D0 + TBleachA)))
                tA = np.unique(np.concatenate((tA_FRET, tA_D0 + TBleachA)))
                tA0 = np.unique(np.concatenate((tA0_FRET, tA0_D0 + TBleachA)))
            
            elif TBleachD < TBleachA:

                TB = TBleachA
            
                tD_FRET, tA_FRET, tA0_FRET = FRET(nD, nA, nA0, TBleachD)   
                tD_A0, tA_A0, tA0_A0 = AcceptorOnly(nA0, TB-TBleachD)   
            
                tD = np.unique(np.concatenate((tD_FRET, tD_A0 + TBleachD)))
                tA = np.unique(np.concatenate((tA_FRET, tA_A0 + TBleachD)))
                tA0 = np.unique(np.concatenate((tA0_FRET, tA0_A0 + TBleachD)))

            elif TBleachD == TBleachA:

                TB = TBleachA

                tD, tA, tA0 = FRET(nD, nA, nA0, TB)
        
        # no bleaching
        elif TBleachD >= TB and TBleachA >= TB:
            
            tD, tA, tA0 = FRET(nD, nA, nA0, TB)

    elif boolFraction == 3: # acceptor only

        tD, tA, tA0 = AcceptorOnly(nA0, TB)

    return tD, tA, tA0
