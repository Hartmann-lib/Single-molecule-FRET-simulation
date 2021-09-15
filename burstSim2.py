import numpy as np
import matplotlib.pyplot as plt
from arrivalTime import *

"""
Simulation of fluorescence bursts of FRET labeled (donor and acceptor) molecules excited by pulsed interleaved excitation (PIE)
The simulation mimics a single FRET population and can be fed with a certain ratio of single and double labeled molecules
It also accounts for differences in quantum yield, detection efficiency; direct excitation of the acceptor; spectral crosstalk of the donor and fluorescence background
Bleaching of the acceptor and donor as well as multiple molecules are also available

09 - 09 - 2021 | Andreas Hartmann
"""

# parameters
#########################################################################################################################
# output filename
strOut = 'Simulation_E0p4_E0p6'

E = [0.4, 0.6] # FRET efficiency -> if multiple FRET efficiencies the number of molecules is numIter*len(E)
TB = 1.5 # average burst duration (exponentially distributed)
minN = 50 # lower threshold -> minimum number of photons per burst

phiD = 0.9 # quantum yield donor
phiA = 0.65 # quantum yield acceptor
nDex = 50 # (kHz) donor excitation rate by green laser
nAex = 50 # (kHz) acceptor excitation rate by red laser
epsG = 0.8 # detection efficiency of green signal
epsR = 0.7 # detection efficiency of red signal

gamma = (epsR*phiA)/(epsG*phiD) # correction for differences in quantum yield and detection efficiency
alpha = 0.025 # direct excitation of the acceptor
beta = 0.054 # spectral crosstalk

print("Alpha = {a:.4f}".format(a=alpha))
print("Beta = {b:.4f}".format(b=beta))
print("Gamma = {g:.4f}".format(g=gamma))

# background brightness
BG_GG = 7 # (kHz = photons/ms) background in GG channel
BG_GR = 3 # (kHz) background in GR channel
BG_RR = 5 # (kHz) background in RR channel

# fractions of labeling stoichiometry
fD0 = 0.2 # donor only fraction
fA0 = 0.2 # acceptor only fraction
fE = 1 - fD0 - fA0 # FRET (double labeled) fraction
# !!! fractions have to sum up to 1 -> fD0 + fA0 + fE = 1

# bleaching times
tbleachD = 4 # (ms) bleaching time of the donor (exponentially distributed)
tbleachA = 4 # (ms) bleaching time of the acceptor

# multiple molecule
meanN = 1.4 # average number of multiple molecules meanN >= 1 (poisson distributed)
boolSame = 0 # boolean: 0 -> independent multiple molecules; 1 -> multiple molecules in the same complex

numIter = 10000 # number of molecules per FRET efficiency value
#########################################################################################################################

idxSpecies = np.array([1,2,3])

all_FGG = []
all_FGR = []
all_FRR = []
all_arrE = []
all_arrS = []
all_arrMT_GG = []
all_arrMT_GR = []
all_arrMT_RR = []
all_arrTB_m = []
all_boolSelect = []

if isinstance(E, list):

    lenE = len(E)
else:
    lenE = 1


for iterE in range(lenE):

    pointer = 0

    FGG = np.zeros(numIter) # background corrected fluorescence intensity in GG channel
    FGR = np.zeros(numIter) # background corrected fluorescence intensity in GR channel
    FRR = np.zeros(numIter) # background corrected fluorescence intensity in RR channel
    arrS = np.zeros(numIter) # corrected stoichiometry
    arrE = np.zeros(numIter) # corrected FRET efficiency
    arrMT_GG = np.zeros(numIter) # average microtime of GG channel normalized on TB
    arrMT_GR = np.zeros(numIter) # average microtime of GR channel normalized on TB
    arrMT_RR = np.zeros(numIter) # average microtime of RR channel normalized on TB
    arrTB_m = np.zeros(numIter) # measured burst duration (first to last photon)
    boolSelect = np.zeros(numIter) # boolean: 1 -> single FRET molecule without bleaching effect

    # drawing labeling fraction
    boolFraction = np.dot(np.random.multinomial(1,[fD0, fE, fA0], numIter), idxSpecies.T)

    # drawing bleaching times
    arrTBleachD = np.random.exponential(tbleachD, numIter)
    arrTBleachA = np.random.exponential(tbleachA, numIter)

    # drawing burst duration
    arrTB = np.random.exponential(TB, numIter)

    # drawing number of molecules in confocal volume
    arrNbust = np.random.poisson(meanN - 1, numIter)+1
       
    # calculating count rates   
    if isinstance(E, list):

        nD = nDex*(1 - alpha)*(1 - E[iterE])*phiD*epsG # (kHz) count rate of the donor in the case of FRET
        nA = nDex*alpha*phiA*epsR + nDex*(1 - alpha)*E[iterE]*phiA*epsR + nD*beta # (kHz) count rate of the acceptor in the case of FRET
    else:
        nD = nDex*(1 - alpha)*(1 - E)*phiD*epsG # (kHz) count rate of the donor in the case of FRET
        nA = nDex*alpha*phiA*epsR + nDex*(1 - alpha)*E*phiA*epsR + nD*beta # (kHz) count rate of the acceptor in the case of FRET

    nD0 = nDex*phiD*epsG # (kHz) count rate of the donor in the absence of acceptor
    nA_D0 = nDex*beta # (kHz) count rate of the acceptor in the absence of acceptor (spectral crosstalk)
    nA0 = nAex*phiA*epsR # (kHz) count rate of the acceptor in the absence of donor

    pD = nD/(nD + nA) # ratio of donor photons in the case of FRET
    pD0 = nD0/(nD0 + nA_D0) # ratio of donor photons in the absence of acceptor

    for iter in range(numIter):

        # calculate arrival times of background photons
        iptBG_GG = np.random.exponential(1/BG_GG, (round(10*BG_GG*arrTB[iter]), 1))
        iptBG_GR = np.random.exponential(1/BG_GR, (round(10*BG_GR*arrTB[iter]), 1))
        iptBG_RR = np.random.exponential(1/BG_RR, (round(10*BG_RR*arrTB[iter]), 1))

        tBG_GG = np.cumsum(iptBG_GG)  
        tBG_GR = np.cumsum(iptBG_GR)
        tBG_RR = np.cumsum(iptBG_RR)

        tBG_GG = tBG_GG[tBG_GG < arrTB[iter]]
        tBG_GR = tBG_GR[tBG_GR < arrTB[iter]]
        tBG_RR = tBG_RR[tBG_RR < arrTB[iter]]

        tD = np.array([])
        tA = np.array([])
        tA0 = np.array([])

        boolFRET_only = 1

        # iteration through multiple molecules
        for iterN in range(arrNbust[iter]):

            if iterN == 0:


                curr_tD, curr_tA, curr_tA0 = Macrotimes(boolFraction[iter], arrTB[iter], arrTBleachD[iter], arrTBleachA[iter], nD0, nA_D0, nA0, nD, nA) 

                # only FRET molecules without bleaching stay labeled with 1
                if arrTBleachD[iter] < arrTB[iter] or arrTBleachA[iter] < arrTB[iter] or boolFraction[iter] != 2:

                    boolFRET_only = 0
            else:

                boolFRET_only = 0

                new_boolFraction = np.dot(np.random.multinomial(1,[fD0, fE, fA0], 1), idxSpecies.T)
                new_TBleachD = np.random.exponential(tbleachD)
                new_TBleachA = np.random.exponential(tbleachA)

                if boolSame == 0:
                
                    new_TB = np.random.exponential(TB)
                else:
                    new_TB = arrTB[iter]

                curr_tD, curr_tA, curr_tA0 = Macrotimes(new_boolFraction, new_TB, new_TBleachD, new_TBleachA, nD0, nA_D0, nA0, nD, nA)
            
            # merge arrival times
            tD = np.unique(np.concatenate((tD, curr_tD)))
            tA = np.unique(np.concatenate((tA, curr_tA)))
            tA0 = np.unique(np.concatenate((tA0, curr_tA0)))

        # correction of burst time and background photons for multiple molecules
        if (len(tD) + len(tA) + len(tA0)) != 0:    
        
            max_t = max(np.concatenate((tD, tA, tA0)))

            if max_t > arrTB[iter]:

                arrTB[iter] = max_t

                # calculate arrival times of background photons
                iptBG_GG = np.random.exponential(1/BG_GG, (round(10*BG_GG*arrTB[iter]), 1))
                iptBG_GR = np.random.exponential(1/BG_GR, (round(10*BG_GR*arrTB[iter]), 1))
                iptBG_RR = np.random.exponential(1/BG_RR, (round(10*BG_RR*arrTB[iter]), 1))

                tBG_GG = np.cumsum(iptBG_GG)  
                tBG_GR = np.cumsum(iptBG_GR)
                tBG_RR = np.cumsum(iptBG_RR)

                tBG_GG = tBG_GG[tBG_GG < arrTB[iter]]
                tBG_GR = tBG_GR[tBG_GR < arrTB[iter]]
                tBG_RR = tBG_RR[tBG_RR < arrTB[iter]]

        # adding background photons
        tD_BG = np.unique(np.concatenate((tD, tBG_GG)))
        tA_BG = np.unique(np.concatenate((tA, tBG_GR)))
        tA0_BG = np.unique(np.concatenate((tA0, tBG_RR)))

        # saving characteristic values in arrays
        if (len(tD_BG) + len(tA_BG) + len(tA0_BG)) >= minN:

            FGG[pointer] = len(tD_BG) - BG_GG*arrTB[iter]
            FGR[pointer] = len(tA_BG) - BG_GR*arrTB[iter]
            FRR[pointer] = len(tA0_BG) - BG_RR*arrTB[iter]

            # normalized mean arrival time
            if len(tD_BG) == 0:

                arrMT_GG[pointer] = 0
            else:
                arrMT_GG[pointer] = np.mean(tD_BG/arrTB[iter])

            if len(tA_BG) == 0:

                arrMT_GR[pointer] = 0
            else:
                arrMT_GR[pointer] = np.mean(tA_BG/arrTB[iter])

            if len(tA0_BG) == 0:

                arrMT_RR[pointer] = 0
            else:
                arrMT_RR[pointer] = np.mean(tA0_BG/arrTB[iter])

            # measuring the burst duration 
            arrTB_m[pointer] = max(np.concatenate((tD_BG, tA_BG, tA0_BG))) - min(np.concatenate((tD_BG, tA_BG, tA0_BG)))

            boolSelect[pointer] = boolFRET_only

            pointer += 1

        # print progress
        if (iter % 1000) == 0 or iter == (numIter - 1):

            print("Progress: {p: 4.0f} %".format(p = iter/(numIter-1)*100))
        
    if isinstance(E, list):
        
        print("----> E={e: 1.3f} <---- Done! ".format(e = E[iterE]))
    else:
        print("----> E={e: 1.3f} <---- Done! ".format(e = E))

    FGG = FGG[0:pointer]
    FGR = FGR[0:pointer]
    FRR = FRR[0:pointer]

    arrMT_GG = arrMT_GG[0:pointer]
    arrMT_GR = arrMT_GR[0:pointer]
    arrMT_RR = arrMT_RR[0:pointer]

    arrTB_m = arrTB_m[0:pointer]

    boolSelect = boolSelect[0:pointer]

    # calculation of the corrected FRET efficiency and stoichiometry
    Ecorr = (FGR - alpha*FRR - beta*FGG)/(FGR - alpha*FRR - beta*FGG + gamma*FGG)
    Scorr = (FGR - alpha*FRR - beta*FGG + gamma*FGG)/(FGR - alpha*FRR - beta*FGG + gamma*FGG + FRR)

    all_FGG = np.concatenate((all_FGG,FGG)) # background corrected fluorescence intensity in GG channel 
    all_FGR = np.concatenate((all_FGR, FGR)) # background corrected fluorescence intensity in GR channel
    all_FRR = np.concatenate((all_FRR, FRR)) # background corrected fluorescence intensity in RR channel
    all_arrE = np.concatenate((all_arrE, Ecorr)) # corrected FRET efficiency
    all_arrS = np.concatenate((all_arrS, Scorr)) # corrected stoichiometry
    all_arrMT_GG = np.concatenate((all_arrMT_GG, arrMT_GG)) # average microtime of GG channel normalized on TB
    all_arrMT_GR = np.concatenate((all_arrMT_GR, arrMT_GR)) # average microtime of GR channel normalized on TB
    all_arrMT_RR = np.concatenate((all_arrMT_RR, arrMT_RR)) # average microtime of RR channel normalized on TB
    all_arrTB_m = np.concatenate((all_arrTB_m, arrTB_m)) # measured burst duration (first to last photon)
    all_boolSelect = np.concatenate((all_boolSelect, boolSelect)) # boolean: 1 -> single FRET molecule without bleaching effect

# combining all data in one array
# with rows being #molecule and columns being the 10 values above
data = np.append(np.expand_dims(all_FGG, axis = 1), np.expand_dims(all_FGR, axis = 1), axis=1)
data = np.append(data, np.expand_dims(all_FRR, axis = 1), axis=1)
data = np.append(data, np.expand_dims(all_arrE, axis = 1), axis=1)
data = np.append(data, np.expand_dims(all_arrS, axis = 1), axis=1)
data = np.append(data, np.expand_dims(all_arrTB_m, axis = 1), axis=1)
data = np.append(data, np.expand_dims(all_arrMT_GG, axis = 1), axis=1)
data = np.append(data, np.expand_dims(all_arrMT_GR, axis = 1), axis=1)
data = np.append(data, np.expand_dims(all_arrMT_RR, axis = 1), axis=1)
data = np.append(data, np.expand_dims(all_boolSelect, axis = 1), axis=1)

# saving data
np.save(strOut, data)

# plotting of the data
plt.figure(figsize=(12, 9))
plt.subplot(2,2,1)
plt.scatter(all_arrE, all_arrS, s=2)
plt.xlabel('FRET efficiency, $\it{E}_{\mathrm{corr}}$')
plt.ylabel('Stoichiometry, $\it{S}_{\mathrm{corr}}$')
plt.title('All simulated bursts')
plt.axis([-0.1, 1.1, -0.1, 1.1])

plt.subplot(2,2,2)
plt.scatter(all_arrE[all_boolSelect==1], all_arrS[all_boolSelect==1], s=2)
plt.xlabel('FRET efficiency, $\it{E}_{\mathrm{corr}}$')
plt.ylabel('Stoichiometry, $\it{S}_{\mathrm{corr}}$')
plt.title('FRET bursts only')
plt.axis([-0.1, 1.1, -0.1, 1.1])

plt.subplot(2,2,3)
plt.hist(all_arrTB_m, 50)
plt.xlabel('Burst duration, $\it{T}_{\mathrm{B}}$ (ms)')
plt.ylabel('Number of molecules')

edges=np.arange(0,2*max([nDex, nAex]),max([nDex, nAex])/50)
hist_MBGG, bin_edges = np.histogram(all_FGG/all_arrTB_m, edges)
hist_MBGR, bin_edges = np.histogram(all_FGR/all_arrTB_m, edges)
hist_MBRR, bin_edges = np.histogram(all_FRR/all_arrTB_m, edges)

plt.subplot(2,2,4)
plt.step(edges[1:], hist_MBGG)
plt.step(edges[1:], hist_MBGR)
plt.step(edges[1:], hist_MBRR)
plt.xlabel('Molecular brightness (kHz)')
plt.ylabel('Number of molecules')
plt.legend(['GG', 'GR', 'RR'])
plt.show()
