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
    
    C_heights = np.convolve(A_heights, B_heights)/len(B)
    C_edges = B_edges[0] + A_edges[0] + np.arange(0, len(C_heights) + 1) * step
    
    if plot:
        plt.figure(figsize=(12, 4))
        plt.subplot(131)
        plt.bar(A_edges[:-1], Normalizer*A_heights, step)
        plt.xlim(0,10)
        plt.xlabel('energy (MeV)')
        plt.ylabel('counts/min')
        # plt.yscale('log')
        plt.title('data')
        plt.subplot(132)
        plt.bar(B_edges[:-1], B_heights/len(B), step)
        plt.title('response')
        plt.subplot(133)
        plt.bar(C_edges[:-1], Normalizer*C_heights, step)
        plt.xlim(0,10)
        plt.xlabel('energy (MeV)')
        plt.ylabel('counts/min')
        # plt.yscale('log')
        plt.title('convolved')
    return C_edges, C_heights, step

def convolve_gaussian(data, sigma):
    return np.random.normal(data, sigma)


manual_adjustments = True
mono_adjustments=False
isPlated=True
shuffle = True

version=6
dt=1
# timesteps=int(3*60*60/dt)
timesteps=9500
timesteps2=25000 #see following comment.
if manual_adjustments == True:  timesteps=25000 #to make manual adjustments, keep the pool of possible datapoints large to choose from
conditions='deflection'
if isPlated:    load('pipsMC{}_output_{}_noions_plated'.format(version,timesteps))
else:   load('pipsMC{}_output_{}_noions'.format(version,timesteps))
load('integrated_activities_50000')

# '''
volume=905142
total_volume=3.3e6
#values of various species to manually set so as to match the data.
#this is for reverse-field conditions
if conditions == 'reverse':
    n_Rn222s = int(Rn222AiIt[timesteps2]*0.00123*volume/total_volume) #integrated activity * volume fraction = initial radon decays, and 0.00122 is an efficiency of hitting the detector that i got by averaging some cases
    n_ambientPo218s = int(((n_Rn222s/0.00123)*Po218AiIt[timesteps2]/Rn222AiIt[timesteps2])*0.00123) #0.0005 is efficiency of hitting the detector. different between Rn and Po because Rn has the advantage of being able to start in the field region which has a nice view of the PIPS
    #see comment in pipsMC6: poloniums aren't neutralised so they need to be considered from the field region too
    n_ambientPo214s = int(((n_Rn222s/0.00123)*Po214AiIt[timesteps2]/Rn222AiIt[timesteps2])*0.00123)
    n_monoPo218s = 0
    n_monoPo214s = int(n_monoPo218s*Po214AiIt[timesteps2]/Po218AiIt[timesteps2])
    t_monoPo210s = timesteps2
if conditions == 'deflection' or conditions == 'transfer':
    n_Rn222s = int(1*Rn222AiIt[timesteps2]*0.00123*volume/total_volume)
    n_ambientPo218s = int(1*((n_Rn222s/0.00123)*Po218AiIt[timesteps2]/Rn222AiIt[timesteps2])*0.00123)
    n_ambientPo214s = int(1*((n_Rn222s/0.00123)*Po214AiIt[timesteps2]/Rn222AiIt[timesteps2])*0.00123)
    n_monoPo218s = int(0.5*0.01*n_Rn222s/0.00123) #fielddecays/N = 0.01, and n_Rn222s = N*0.00122
    n_monoPo214s = int(n_monoPo218s*Po214AiIt[timesteps2]/Po218AiIt[timesteps2])
    t_monoPo210s = timesteps2

sigma_fudge = 1
if True: #further manual finetuning on top of the existing manual adjustments. i don't want to make another variable to easily switch this so i'll just put it in a block like this
    if conditions == 'reverse':
        n_Rn222s *= 1.0
        n_ambientPo218s *= 1.1
        n_ambientPo214s *= 1.1
        n_plated218 = 50
        n_plated214 = 50
        n_monoPo218s = 140
        n_monoPo214s = int(n_monoPo218s*Po214AiIt[timesteps2]/Po218AiIt[timesteps2])
        n_monoPo214s *= 0.5
        t_monoPo210s *= 0.5
        sigma_fudge *= 1.0
    elif conditions == 'deflection':
        n_Rn222s *= 1.0
        n_ambientPo218s *= 1.0
        n_ambientPo214s *= 1.0
        n_plated218 = 0
        n_plated214 = 0
        n_monoPo218s *= 0.4
        n_monoPo214s *= 0.5
        t_monoPo210s *= 0.5
        sigma_fudge *= 1.0
    elif conditions == 'transfer':
        n_Rn222s *= 1.0
        n_ambientPo218s *= 1.0
        n_ambientPo214s *= 1.0
        n_plated218 = 0
        n_plated214 = 0
        n_monoPo218s *= 0.55
        n_monoPo214s *= 0.75
        t_monoPo210s *= 0.5
        sigma_fudge *= 1.0
# '''
n_bins=100

n_Rn222s = int(n_Rn222s)
n_ambientPo218s = int(n_ambientPo218s)
n_ambientPo214s = int(n_ambientPo214s)
n_monoPo218s = int(n_monoPo218s)
n_monoPo214s = int(n_monoPo214s)
t_monoPo210s = int(t_monoPo210s)

#we have N, volume, decayDensity, num_successes, efficiency, successes, fielddecays, Normalizer, po218successes, po214successes
# Normalizer=1

#just checking what we have:
print(f"{conditions=}")
print(f"{manual_adjustments=}")
if manual_adjustments == False:
    print(f"{timesteps=}")
    print(f"{N=}")
    # print(f"{volume=}", 'mm^3')
    # print(f"{decayDensity=}", 'decays per mm^3')
    print(f"{num_successes=}")
    print(f"{fielddecays=}")
    # print(f"{Normalizer=}")
    print('po218 non-field successes= ', len(po218successes))
    print('po214 non-field successes= ', len(po214successes))
#print(f"{successes=}")
#print(successes[:,0][0]) #first successful decay position as xyz list
    # print(f"{addifreverse=}")
elif manual_adjustments == True:
    print('virtual timesteps= ', timesteps2)
    print('rn222 hits= ', n_Rn222s)
    print('po218 non-field hits= ', n_ambientPo218s)
    print('po214 non-field hits= ', n_ambientPo214s)
    print('po218 monoenergetic hits= ', n_monoPo218s)
    print('po214 monoenergetic hits= ', n_monoPo214s)
    if isPlated:
        print('po218 plated-out hits= ', n_plated218)
        print('po214 plated-out hits= ', n_plated214)
    

MC_distances=np.array(successes[:,2], dtype=float)*0.1 #converting mm to cm

# Normalizer = 1/(24*60) #converting counts to counts/min
# Normalizer *= 0.8/1.2 #pressures and hence densities of chambers affecting density of decays


astar=np.loadtxt('AR_apdata.pl.txt') #data table from ASTAR
energies=astar[:,0]
dEdxs_AR=astar[:,1]
if conditions == 'reverse': pressure_detector, pressure_source = 0.9, 1.4
if conditions == 'deflection': pressure_detector, pressure_source = 1.3, 1.4
if conditions == 'transfer': pressure_detector, pressure_source = 0.9, 1.4
densityAr_detector=0.0016448*pressure_detector #g/cc #at 19 deg C
densityAr_source=0.0016448*pressure_source
dEdxs_AR=densityAr_detector*dEdxs_AR #normalizing ASTAR stopping powers with density of the medium
lengths=np.linspace(0,13,2**10) #13cm for good measure, that's about how far Po214s reach
V_detector = 3 #litres, approximately
V_source = 0.5
density_fraction = (1+(V_source/V_detector))/(1+(V_source/V_detector)*(densityAr_source/densityAr_detector))

dEdxs_SI=2.33*np.loadtxt('SI_apdata.pl.txt')[:,1] #stopping power in silicon, multiplied with density

if shuffle == True:  np.random.shuffle(MC_distances)
if manual_adjustments == True:
    MC_distances = MC_distances[:n_Rn222s] #manually setting how many radons are seen, to try and match the experimental data
#MC_distances = MC_distances[:int(len(MC_distances)*density_fraction)] #less pressure (hence density) in the detector chamber so less activity there compared to system-wide average

Rn222Q=5.5904 #MeV
Po218Q=6.1
Po214Q=7.8
Po210Q=5.4
Rn222Q *= (222-4)/222 #changing q-value to alpha particle kinetic energy. not big difference but still
Po218Q *= (218-4)/218
Po214Q *= (214-4)/214
Po210Q *= (210-4)/210

#this function gives a list of what energy an alpha with a particular inital energy will have after travelling a certain length.
#the lengths of interest are just 0 to 13cm with a thousand divisions in between
def alpha_energies(init_E, medium='argon'):
    dEdxs = dEdxs_AR
    if medium=='silicon':
        dEdxs = dEdxs_SI
    dx=lengths[1]-lengths[0]
    alpha_energies=np.zeros(lengths.shape)
    alpha_energies[0]=init_E #MeV
    for i in range(1,len(alpha_energies)):
        alpha_energy=alpha_energies[i-1]
        dE=dx*np.interp(alpha_energy, energies, dEdxs)
        alpha_energy-=dE
        alpha_energies[i]=alpha_energy
    return alpha_energies

#'''
##temporary: make a plot of this
plt.plot(lengths[alpha_energies(Rn222Q)>0], alpha_energies(Rn222Q)[alpha_energies(Rn222Q)>0])
plt.show()
exit()
#'''

#the pipsMC#.py simulation gave a list of lengths the alphas had to go through, now we plug these into ASTAR data to get a list of energies the alphas will have when hitting the detector
MC_energies_detected=np.interp(MC_distances, lengths, alpha_energies(Rn222Q))


'''
###temporary: making a cdf of distances travelled by radons, just to see how far away most of them are coming from
#this is to judge the straight-line trajectory assumption by comparing to SRIM plots
def ecdf(x):
    xs = np.sort(x)
    ys = np.arange(1, len(xs)+1)/float(len(xs))
    return xs, ys
xs, ys = ecdf(MC_distances[MC_energies_detected>0])
plt.title('ecdf of travel distance of detected radon alphas in {} bar Ar'.format(pressure_detector))
plt.plot(xs,ys)
plt.xlabel('travel distance')
plt.ylabel('cumulative probability')
plt.show()
exit()
'''
#'''
###temporary: finding out where the successful radon alphas are coming from
range=np.max(MC_distances[MC_energies_detected>0])
print('range of radon alphas in {} bar Ar = '.format(pressure_detector), range)
Os = np.array(list(successes[:,0]))
xs = 0.1*Os[:,0]
ys = 0.1*Os[:,1]
zs = 0.1*Os[:,2]
rs=np.sqrt(xs**2+ys**2+zs**2)
xs=xs[rs<=range]
ys=ys[rs<=range]
zs=zs[rs<=range]
rs=np.sqrt(xs**2+ys**2+zs**2)
# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# ax.set_title('where successful radon alphas are coming from ({} bar Ar)'.format(pressure_detector))
# ax.set_xlabel('x (cm)')
# ax.set_ylabel('y (cm)')
# ax.set_zlabel('z (cm)')
# ax.scatter(xs, ys, zs, s=1)

plt.title('alpha distances travelled')
plt.hist(rs, bins=80)
plt.show()
exit()
#'''


#adding poloniums stuck to pips. the gross number is as many radons as initialised in field region, the net number is half of that
monopoloniums = np.array([])
po218s=Po218Q*np.ones(int(density_fraction*fielddecays/2))
po214s=Po214Q*np.ones(int(density_fraction*fielddecays/2))
if manual_adjustments == True or mono_adjustments == True:
    po218s = po218s[:n_monoPo218s]
    po214s = po214s[:n_monoPo214s]
    timesteps=t_monoPo210s #just for po210. now i am manually adjusting everything else to fit the data
po210s = Po210Q*np.ones(int(37.7*timesteps/3600)) #constant po210 background experimentally measured at 37.71+-3.66 counts/hour
print('po210 hits= ', len(po210s))

# simulating the monopoloniums actually going through a thin silicon dead layer
window = 50e-7 # dead layer thickness 50nm (converted to cm)
# angular distribution is isotropic over a hemisphere (pointed into the detector). for isotropy, colatitude must be chosen such that cos(theta) is uniform. see stackoverflow. then the thickness travelled is window/cos(theta)
po218s=np.interp(window/np.random.uniform(size=len(po218s)), lengths, alpha_energies(Po218Q, 'silicon'))
po214s=np.interp(window/np.random.uniform(size=len(po214s)), lengths, alpha_energies(Po214Q, 'silicon'))
po210s=np.interp(window/np.random.uniform(size=len(po210s)), lengths, alpha_energies(Po210Q, 'silicon'))

if (conditions!='reverse' or manual_adjustments==True or mono_adjustments==True):
    monopoloniums = np.append(monopoloniums, po218s)
    monopoloniums = np.append(monopoloniums, po214s)
monopoloniums = np.append(monopoloniums, po210s)

#adding poloniums from ambient, as found in pipsMC4.py
MC_distances_po218=[]
if(po218successes.size>0):
    MC_distances_po218=np.array(po218successes[:,2], dtype=float)*0.1
MC_distances_po214=[]
if(po214successes.size>0):
    MC_distances_po214=np.array(po214successes[:,2], dtype=float)*0.1
if isPlated:
    MC_distances_plated=[]
    if(poPlatesuccesses.size>0):
        MC_distances_plated=np.array(poPlatesuccesses[:,2], dtype=float)*0.1

#putting together the poloniums from ambient and from PIPS
Po218s=np.interp(MC_distances_po218, lengths, alpha_energies(Po218Q))
Po214s=np.interp(MC_distances_po214, lengths, alpha_energies(Po214Q))
if isPlated:
    Plated214s=np.interp(MC_distances_plated, lengths, alpha_energies(Po214Q))
    Plated218s=np.interp(MC_distances_plated, lengths, alpha_energies(Po218Q))
if manual_adjustments == True:
    if shuffle == True:  np.random.shuffle(Po218s)
    if shuffle == True:  np.random.shuffle(Po214s)
    if shuffle == True and isPlated == True:
        np.random.shuffle(Plated214s)
        np.random.shuffle(Plated218s)
    Po218s = Po218s[:n_ambientPo218s]
    Po214s = Po214s[:n_ambientPo214s]
    if isPlated:
        Plated214s = Plated214s[:n_plated214]
        Plated218s = Plated218s[:n_plated218]
    
poloniums=np.append(Po218s, Po214s)
if isPlated:
    poloniums=np.append(poloniums, Plated214s)
    poloniums=np.append(poloniums, Plated218s)
poloniums=np.append(poloniums, monopoloniums)
poloniums=poloniums[poloniums>0]

MC_energies_detected=np.append(MC_energies_detected, poloniums)

#the detector region i cut out is a bit too large, so some particles that would have been exhausted
#by the time they hit the detector are inadvertently included. cutting them out here with this mask
MC_energies_detected=MC_energies_detected[MC_energies_detected>0]



#loading experimental results

if conditions == 'transfer':
    expdata = np.loadtxt('transfer3.2_hist.txt')
if conditions == 'reverse':
    expdata = np.loadtxt('reverse2.2_hist.txt')
if conditions == 'deflection':
    expdata = np.loadtxt('deflection1.2_hist.txt')
calibrator = np.mean(expdata[expdata>160])/15.2 #calibration peak is centred at 170, which from diagram is about 15.3 MeV. so divide by about 11
expdata = expdata/calibrator
sigma = np.std(expdata[expdata>10]) #finding width of calibration peak to use to convolve simulation results appropriately
expdata = expdata[expdata<10] #cutting out calibration peak
print('number of experimentally observed pips hits: ', len(expdata))
print('number of experimentally observed pips hits (Po-214): ', len(expdata[expdata>6.6])) #visually chosen


#convolve simulated data with detector response
convolved = convolve_gaussian(MC_energies_detected, sigma_fudge*sigma)

def FDbins(v):  #Freedman-Diaconis bins rule
    try:
        return int(np.ptp(v)/(2*ss.iqr(v)*len(v)**(-1./3)))
    except:
        return 1

# np.random.seed(1)
B = np.random.normal(0,sigma_fudge*sigma,2**22)
save('gaussian_{}'.format(sigma_fudge*sigma), 'B')
# exit()
load('gaussian_{}'.format(sigma_fudge*sigma))
#plotting simulated results
C_edges, C_heights, step = hist_of_addition(MC_energies_detected, B, bins=n_bins, plot=False)
# plt.bar(C_edges[:-1], C_heights, step, color='orange', alpha=0.8, label='simulation')
#plotting experimental results
bins = C_edges-0.5*step
# save('bins_{}'.format(conditions),'bins')
# load('bins_{}'.format(conditions))
plt.hist(expdata, bins=bins, color='blue', alpha=0.3, label='experiment')
plt.hist(convolved, bins=bins, color='red', alpha=0.4, label='simulation 2')

plt.xlim((-0.5,8.5))
# if conditions == 'reverse': plt.ylim((0,44))
# elif conditions == 'deflection': plt.ylim((0,80))
# elif conditions == 'transfer': plt.ylim((0,100))
plt.xlabel('energy (MeV)')
plt.ylabel('counts')
plt.title('{}-field run'.format(conditions))
plt.legend()
# plt.savefig('{}_{}_detectedenergies_{}.png'.format(version,conditions,timesteps), bbox_inches='tight')
plt.show()
