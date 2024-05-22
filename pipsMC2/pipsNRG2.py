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
        '''
        plt.figure(figsize=(12, 4))
        plt.subplot(131)
        plt.bar(A_edges[:-1], Normalizer*A_heights, step)
        plt.title('data')
        plt.subplot(132)
        plt.bar(B_edges[:-1], B_heights/len(B), step)
        plt.title('response')
        plt.subplot(133)
        plt.bar(C_edges[:-1], Normalizer*C_heights, step)
        plt.title('convolved')
        '''
        
        plt.bar(A_edges[:-1], A_heights, step, label='simulated data')
        plt.bar(C_edges[:-1], C_heights, step, color='orange', alpha=0.8, label='convolved with detector response, $\sigma=0.475$ MeV')
        plt.xlim(0,7)
        plt.xlabel('energy of alpha particle as it hits detector (MeV)')
        plt.ylabel('abundance')
        plt.legend()
        plt.title('')
        
    return C_edges, C_heights


load('pipsMC2_output')

#we have N, volume, decayDensity, num_successes, efficiency, successes, fielddecays, Normalizer
#Normalizer=4000*volume/(2**22)
#Normalizer=1

print(f"{N=}")
print(f"{volume=}", 'mm^3')
print(f"{decayDensity=}", 'decays per mm^3')
print(f"{num_successes=}")
print(f"{efficiency=}", '=', efficiency*100, '%')
print(f"{fielddecays=}")
print(f"{Normalizer=}")
#print(f"{successes=}")
#print(successes[:,0][0]) #first successful decay position as xyz list
MC_distances=np.array(successes[:,2], dtype=float)*0.1 #converting mm to cm

#finding how many successful decays were in field region, i.e. 5<z<41.25 and sqrt(x**2+y**2)<8.2
decayPos=np.array(successes[:,0])
decayPos=np.array([np.array(xi) for xi in decayPos])
zs=decayPos[:,2]
rhosquareds=np.power(decayPos[:,0], 2)+np.power(decayPos[:,1], 2)
fieldsuccesses=np.logical_and.reduce([zs>5, zs<41.25, rhosquareds<67.24])
print(fieldsuccesses.sum())
#poloniums that stick to the pips come from everything in the field, not just this! are monoenergetic, and half of them are detected. add them manually

astar=np.loadtxt('apdata.pl.txt')
energies=astar[:,0]
dEdxs=astar[:,1]
densityAr=0.00115 #g/cc
dEdxs=densityAr*dEdxs
lengths=np.linspace(0,13,2**10) #13cm for good measure

Rn222Q=5.5904 #MeV
Po218Q=6.1
Po214Q=7.8

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

MC_energies_detected=np.interp(MC_distances, lengths, alpha_energies(Rn222Q))

'''
#adding poloniums stuck to pips
po218s=Po218Q*np.ones(int(fielddecays/2))
po214s=Po214Q*np.ones(int(fielddecays/2))
monopoloniums=np.append(po218s,po214s)
#adding poloniums hanging around in argon (but outside field region), recycling the radon's positions
MC_distances=MC_distances[np.logical_not(fieldsuccesses)] #only keep the positions outside the field region since we're doing polonium ions now
Po218s=np.interp(MC_distances, lengths, alpha_energies(Po218Q))
Po214s=np.interp(MC_distances, lengths, alpha_energies(Po214Q))
poloniums=np.append(Po218s, Po214s)
poloniums=poloniums[poloniums>0]
poloniums=np.append(poloniums, monopoloniums)

MC_energies_detected=np.append(MC_energies_detected, poloniums)
#MC_energies_detected=poloniums
'''

#the detector region i cut out is a bit too large, so some particles that would have been exhausted
#by the time they hit the detector are inadvertently included. cutting them out here with this mask
MC_energies_detected=MC_energies_detected[MC_energies_detected>0]


#convolve with detector response
#FWHM=14e-3 #MeV. From Mirion datasheet for PD300-15-300AM'
#sigma=FWHM/(2*np.sqrt(2*np.log(2)))
sigma=0.475 #MeV. From Americium test
response=ssig.windows.gaussian(len(MC_energies_detected), sigma)
convolved=ssig.convolve(MC_energies_detected, response, mode='same')/sum(response)

FDbins = lambda v: int(np.ptp(v)/(2*ss.iqr(v)*len(v)**(-1./3))) #Freedman-Diaconis bins rule

hist_of_addition(MC_energies_detected, np.random.normal(0,sigma,2**22), bins=FDbins(MC_energies_detected), plot=True)

#plt.hist(MC_energies_detected, bins=2*FDbins(MC_energies_detected), label='simulated data')

'''
response2=np.random.normal(0,sigma,len(MC_energies_detected))
plt.hist(response2, bins=FDbins(response2), alpha=0.7)
convolved2=np.convolve(MC_energies_detected,response2, mode='full')
plt.hist(convolved2, bins=FDbins(convolved2), alpha=0.7)
#'''

'''
plt.hist(convolved, bins=FDbins(convolved), color='orange', alpha=0.8, label='convolved with detector response, $\sigma=0.475$ MeV')
plt.xlabel('energy of alpha particle as it hits detector (MeV)')
plt.ylabel('abundance')
plt.legend()
#'''
plt.show()
