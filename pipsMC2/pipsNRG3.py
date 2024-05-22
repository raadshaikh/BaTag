import numpy as np
import pickle
import matplotlib.pyplot as plt
import scipy.stats as ss
import scipy.signal as ssig

def save(filename, *args):
    # Get global dictionary
    glob = globals()
    d = {}
    for v in args:
        # Copy over desired values
        d[v] = glob[v]
    with open(filename, 'wb') as f:
        # Put them in the file 
        pickle.dump(d, f)

def load(filename):
    # Get global dictionary
    glob = globals()
    with open(filename, 'rb') as f:
        for k, v in pickle.load(f).items():
            # Set each global variable to the value from the file
            glob[k] = v

def hist_of_addition(A, B, bins=10, plot=False):
    A_heights, A_edges = np.histogram(A, bins=bins)
    # make sure the histogram is equally spaced
    assert(np.allclose(np.diff(A_edges), A_edges[1] - A_edges[0]))
    # make sure to use the same interval
    step = A_edges[1] - A_edges[0]
    
    # specify parameters to make sure the histogram of B will
    # have the same bin size as the histogram of A
    nBbin = int(np.ceil((np.max(B) - np.min(B))/step))
    left = np.min(B)
    B_heights, B_edges = np.histogram(B, range=(left, left + step * nBbin), bins=nBbin)
    
    # check that the bins for the second histogram matches the first
    assert(np.allclose(np.diff(B_edges), step))
    
    C_heights = np.convolve(A_heights, B_heights)/len(B) #/len(B) was cuz otherwise the counts were too much
    C_edges = B_edges[0] + A_edges[0] + np.arange(0, len(C_heights) + 1) * step
    
    if plot:
        plt.figure(figsize=(12, 4))
        plt.subplot(131)
        plt.bar(A_edges[:-1], Normalizer*A_heights, step)
        plt.xlim(0,10)
        plt.xlabel('energy (MeV)')
        plt.ylabel('counts/min')
        plt.yscale('log')
        plt.title('data')
        plt.subplot(132)
        plt.bar(B_edges[:-1], B_heights/len(B), step)
        plt.title('response')
        plt.subplot(133)
        plt.bar(C_edges[:-1], Normalizer*C_heights, step)
        plt.xlim(0,10)
        plt.xlabel('energy (MeV)')
        plt.ylabel('counts/min')
        plt.yscale('log')
        plt.title('convolved')
    return C_edges, C_heights

version=5
dt=1
timesteps=int(24*60*60/dt)
timesteps=10000
load('pipsMC{}_output_{}'.format(version,timesteps))

#we have N, volume, decayDensity, num_successes, efficiency, successes, fielddecays, Normalizer, po218successes, po214successes
#Normalizer=1

#just checking what we have:
print(f"{N=}")
print(f"{volume=}", 'mm^3')
print(f"{decayDensity=}", 'decays per mm^3')
print(f"{num_successes=}")
print(f"{fielddecays=}")
print(f"{Normalizer=}")
print('po218 non-field successes= ', len(po218successes))
print('po214 non-field successes= ', len(po214successes))
#print(f"{successes=}")
#print(successes[:,0][0]) #first successful decay position as xyz list
MC_distances=np.array(successes[:,2], dtype=float)*0.1 #converting mm to cm

# Normalizer = 1/(24*60) #converting counts to counts/min
Normalizer *= 0.8/1.2 #pressures and hence densities of chambers affecting density of decays


astar=np.loadtxt('apdata.pl.txt') #data table from ASTAR
energies=astar[:,0]
dEdxs=astar[:,1]
densityAr=0.00115 #g/cc
dEdxs=densityAr*dEdxs #normalizing ASTAR stopping powers with density of the medium
lengths=np.linspace(0,13,2**10) #13cm for good measure, that's about how far Po214s reach

Rn222Q=5.5904 #MeV
Po218Q=6.1
Po214Q=7.8

#this function gives a list of what energy an alpha with a particular inital energy will have after travelling a certain length.
#the lengths of interest are just 0 to 13cm with a thousand divisions in between
def alpha_energies(init_E): 
    dx=lengths[1]-lengths[0]
    alpha_energies=np.zeros(lengths.shape)
    alpha_energies[0]=init_E #MeV
    for i in range(1,len(alpha_energies)):
        alpha_energy=alpha_energies[i-1]
        dE=dx*np.interp(alpha_energy, energies, dEdxs)
        alpha_energy-=dE
        alpha_energies[i]=alpha_energy
    return alpha_energies

#the pipsMC4.py simulation gave a lsit of lengths the alphas had to go through, now we plug these into ASTAR data to get a list of energies the alphas will have when hitting the detector
MC_energies_detected=np.interp(MC_distances, lengths, alpha_energies(Rn222Q))

#adding poloniums stuck to pips. the gross number is as many radons as initialised in field region, the net number is half of that
po218s=Po218Q*np.ones(int(fielddecays/2))
po214s=Po214Q*np.ones(int(fielddecays/2))
monopoloniums=np.append(po218s,po214s)

#adding poloniums from ambient, as found in pipsMC4.py
MC_distances_po218=[]
if(po218successes.size>0):
    MC_distances_po218=np.array(po218successes[:,2], dtype=float)*0.1
MC_distances_po214=[]
if(po214successes.size>0):
    MC_distances_po214=np.array(po214successes[:,2], dtype=float)*0.1

#putting together the poloniums from ambient and from PIPS
Po218s=np.interp(MC_distances_po218, lengths, alpha_energies(Po218Q))
Po214s=np.interp(MC_distances_po214, lengths, alpha_energies(Po214Q))
poloniums=np.append(Po218s, Po214s)
poloniums=poloniums[poloniums>0]
poloniums=np.append(poloniums, monopoloniums)

MC_energies_detected=np.append(MC_energies_detected, poloniums)

#the detector region i cut out is a bit too large, so some particles that would have been exhausted
#by the time they hit the detector are inadvertently included. cutting them out here with this mask
MC_energies_detected=MC_energies_detected[MC_energies_detected>0]


#convolve with detector response
#FWHM=14e-3 #MeV. From Mirion datasheet for PD300-15-300AM'
#sigma=FWHM/(2*np.sqrt(2*np.log(2)))
sigma=0.475 #MeV. From Americium test
sigma=0.160 #updated detector resolution
response=ssig.windows.gaussian(len(MC_energies_detected), sigma)
convolved=ssig.convolve(MC_energies_detected, response, mode='same')/sum(response)

def FDbins(v):  #Freedman-Diaconis bins rule
    try:
        return int(np.ptp(v)/(2*ss.iqr(v)*len(v)**(-1./3)))
    except:
        return 1

hist_of_addition(MC_energies_detected, np.random.normal(0,sigma,2**22), bins=FDbins(MC_energies_detected), plot=True)

plt.savefig('{}_detectedenergies_{}.png'.format(version,timesteps), bbox_inches='tight')
plt.show()
