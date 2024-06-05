import numpy as np
from stl import mesh
from matplotlib import pyplot as plt
import pickle
from scipy.interpolate import interp1d

#utilities

Epsilon=2**-20 #idk smth small

def filter_decays():
    global decayPos
    global decayDir
    global success_indices
    decayPos=decayPos[success_indices]
    decayDir=decayDir[success_indices]
def filter_decays_po():
    global poPos
    global poDir
    global Po218mask
    global success_indices
    poPos=poPos[success_indices]
    poDir=poDir[success_indices]
def filter_decays_poPlate():
    global poPlatePos
    global poPlateDir
    global success_indices
    poPlatePos=poPlatePos[success_indices]
    poPlateDir=poPlateDir[success_indices]

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



dt=1 #timestep in seconds
# timesteps=int(2*60*60/dt)
timesteps = 9500
print('timesteps = ',timesteps)

#geometry
xmin, xmax = -68.82, 68.82 #bounding box for volume under consideration
ymin, ymax = -25.42, 25.42
zmin, zmax = 0.65, 130
xrange=xmax-xmin
yrange=ymax-ymin
zrange=zmax-zmin
volume=xrange*yrange*zrange
detectR = 9.77 #radius of circular active region on detector. centered at origin, in xy plane.


#interpolate the given data to find out the activity of each species at every point in time
#these are all single-decay reactions, so just integrate activity of the poloniums to find out how many of their decays there are

times_given=np.array([0,10800,21600,86400,172800,259200,345600,432000,864000])

# A for Activity
Ra226A=np.array([5000.00000025, 4999.999258, 4999.998515, 4999.994062,4999.988124,4999.982185,4999.976247,4999.970309,4999.940618]) #t=0 value extrapolated
Rn222A=np.array([0, 112.7145822, 222.8882321, 833.6894579, 1528.370218	,2107.220312	,2589.553113	,2991.46172	,4193.130921])
Po218A=np.array([0,	109.944266, 220.1803673, 831.3278231	, 1526.402361	,2105.580574	,2588.186785	,2990.323214	,4192.673588])
Pb214A=np.array([0,	86.14118592, 196.6577366, 810.8106227	,1509.306184	,2091.335006	,2576.316521	,2980.4322	,4188.700407])
Bi214A=np.array([0	,68.84997429	, 179.1327122, 795.5202288	,1496.565299	,2080.718531	,2567.470235	,2973.060946	,4185.739404])
Po214A=np.array([0	,68.84997191, 179.1327097, 795.5202267	,1496.565297	,2080.71853	,2567.470234	,2973.060945	,4185.739404])
Pb210A=np.array([0	,0.000259254, 0.001578593, 0.033151067	,0.131646674	,0.284703018	,0.483216622	,0.719601905	,2.281773287])

#Aif for Activity interpolation function
Ra226Aif=interp1d(times_given, Ra226A, kind='cubic')
Rn222Aif=interp1d(times_given, Rn222A, kind='cubic')
Po218Aif=interp1d(times_given, Po218A, kind='cubic')
Pb214Aif=interp1d(times_given, Pb214A, kind='cubic')
Bi214Aif=interp1d(times_given, Bi214A, kind='cubic')
Po214Aif=interp1d(times_given, Po214A, kind='cubic')
Pb210Aif=interp1d(times_given, Pb210A, kind='cubic')

times_smooth=np.arange(0,timesteps,dt)

#Ai for Activity interpolated
Ra226Ai=Ra226Aif(times_smooth)
Rn222Ai=Rn222Aif(times_smooth)
Po218Ai=Po218Aif(times_smooth)
Pb214Ai=Pb214Aif(times_smooth)
Bi214Ai=Bi214Aif(times_smooth)
Po214Ai=Po214Aif(times_smooth)
Pb210Ai=Pb210Aif(times_smooth)

'''
#just checking how good the interpolations are (they're pretty good)
plt.plot(times_given, Ra226A, 'o', times_smooth, Ra226Ai, '-')
plt.plot(times_given, Rn222A, 'o', times_smooth, Rn222Ai, '-')
plt.plot(times_given, Po218A, 'o', times_smooth, Po218Ai, '-')
plt.plot(times_given, Pb214A, 'o', times_smooth, Pb214Ai, '-')
plt.plot(times_given, Bi214A, 'o', times_smooth, Bi214Ai, '-')
plt.plot(times_given, Po214A, 'o', times_smooth, Po214Ai, '-')
plt.plot(times_given, Pb210A, 'o', times_smooth, Pb210Ai, '-')
plt.show()
exit()
'''

#AiI for Activity interpolated Integrated

Ra226AiI=Ra226Ai.sum()*dt
Rn222AiI=Rn222Ai.sum()*dt
Po218AiI=Po218Ai.sum()*dt
Pb214AiI=Po218Ai.sum()*dt
Bi214AiI=Bi214Ai.sum()*dt
Po214AiI=Po214Ai.sum()*dt
Pb210AiI=Pb210Ai.sum()*dt

'''
#plot of integrated activities vs time intervals (over any given time interval, how many decay events of any species will occur?)
Rn222AiIt = np.zeros_like(times_smooth)
Po218AiIt = np.zeros_like(times_smooth)
Po214AiIt = np.zeros_like(times_smooth)
for i in range(len(times_smooth)):
    time_range_t = times_smooth[:i]
    Rn222AiIt[i] = Rn222Aif(time_range_t).sum()*dt #AiIt for Activity interpolated Integrated as a function of time
    Po218AiIt[i] = Po218Aif(time_range_t).sum()*dt
    Po214AiIt[i] = Po214Aif(time_range_t).sum()*dt
# plt.plot(times_smooth, Rn222AiIt)
# plt.plot(times_smooth, Po218AiIt)
# plt.plot(times_smooth, Po214AiIt)
# save('integrated_activities', 'times_smooth', 'Rn222AiIt', 'Po218AiIt', 'Po214AiIt')
plt.plot(times_smooth, Po218AiIt/Rn222AiIt)
plt.plot(times_smooth, Po214AiIt/Rn222AiIt)
plt.xlabel('time (sec)')
plt.ylabel('# of decay events over a time interval in the whole system\nnormalised to Rn222')
plt.show()
exit()
'''

#these are just the old variable names I was using.
#dividing by total volume to get density, then multiplying by my bounding box to get how many decays to consider
total_volume= 80e6 #in mm^3 #this was old volume with cryostat
total_volume = 3.3e6 #new volume with small source chamber
N=int(volume*Rn222AiI/total_volume)
Po218s=int(Po218AiI*volume/total_volume)
Po214s=int(Po214AiI*volume/total_volume)

print('decay calculations done\n')

'''start the radon hitting'''
# Load the STL files...
my_mesh = mesh.Mesh.from_file('boil-2-pipsMC.stl')
my_mesh=np.reshape(my_mesh, (len(my_mesh),3,3)) #before this, the whole triangle was in a (9,) array. I am splitting the vertices into their own arrays

print('starting with {} radon decays'.format(N))
#N=int(4000*volume) #24000 Rn/cc, over 1 day, means 4000 decays/cc
Normalizer=1 #This is used to scale the histogram later. In this file we start with the 'real' number of decays so no need for this

#generate random positions and direction rays for each decay.
#N rows, 3 columns. each row identifies a decay, and there is a column for each coordinate
decayPos=np.random.uniform([xmin, ymin, zmin], [xmax, ymax, zmax], (N,3))
decayPos=decayPos.astype(np.float16) #i don't need too much precision
decayDir=np.random.normal(0,1,(N,3))
decayDir=decayDir.astype(np.float16)
#roughly cutting out decays that start outside the cross
xydistsquared=np.power(decayPos[:,0], 2)+np.power(decayPos[:,1], 2)
success_indices=np.logical_not(np.logical_and.reduce([decayPos[:,2]>41, xydistsquared>645]))
filter_decays()
print(len(decayPos))

zs=decayPos[:,2]
rhosquareds=np.power(decayPos[:,0], 2)+np.power(decayPos[:,1], 2)
fielddecaymask=np.logical_and.reduce([zs<41.25, rhosquareds<67.24])
fielddecays=fielddecaymask.sum()

decayDir=decayDir/np.reshape(np.sqrt(np.einsum('ij...,ij->i...',decayDir,decayDir)), (len(decayDir),1))
#this einsum thing is to get the elementwise dot product between two lists of vectors (here to get a list of norms of a list of vectors), to normalise the directions

#stuff going upwards has no chance of hitting the detector
success_indices=decayDir[:,2]<0
filter_decays()

#determine if the generated direction ray falls within the cone of feasibility
#just find where the ray intersects the xy plane and see if it falls within the circle
#if this is confusing just work it out on pen-paper
floor_intersections=decayPos - np.divide(np.multiply(decayDir, np.reshape(decayPos[:,2], (len(decayPos),1))), np.reshape(decayDir[:,2], (len(decayDir),1)))
success_indices=np.einsum('ij...,ij->i...',floor_intersections,floor_intersections)<95.5 #(squared norm <300/pi)
filter_decays()

#the real fight begins. need to find a ray-casting algorithm
#the algorithm takes a point (where decay occurred), a direction (), and a triangle (surface of solid object).
#it returns the intersection of the ray with the triangle (and whether it even intersects)
#for each decay, iterate over all triangles in the assembly to get a list of intersection points
#check whether the intersection points are near the detector
#if any of the intersection points are far away, reject the decay
#if all intersection points are near the decay (technically there should only be 2 but whatever), then the decay succeeds
#do this for all decays parallely? use numpy magic to end up with a mask of success_indices that then filters the decays
#finally find length of decayPos, i.e. how many decays succeeded
#later get a list of lengths between successful decay points and their intersection points with the detector, to put in SRIM later

successes=[]
lendecayPos=len(decayPos)
for i in range(lendecayPos):
    if int(10000*i/lendecayPos)%1==0: print("radons: {:.2f}%".format(100*i/lendecayPos), end='\r')
    O=np.tile(decayPos[i], (len(my_mesh),1))
    D=np.tile(decayDir[i], (len(my_mesh),1))
    v1=my_mesh[:,0]
    v2=my_mesh[:,1]
    v3=my_mesh[:,2]
    #first check what triangles are parallel to the ray and exclude them from further calculations
    normals_dcs_dot=np.einsum('ij...,ij->i...',np.cross(v2-v1, v3-v1),D)
    mask=np.logical_or(normals_dcs_dot<-Epsilon, normals_dcs_dot>Epsilon)
    
    #Moeller-Trumbore
    MTmatrix=np.array([-D[mask].T, (v2-v1)[mask].T,(v3-v1)[mask].T]).T
    MTvector=O[mask]-v1[mask]
    #this gives us intersection of ray with the plane of the triangle:
    tuv=np.linalg.solve(MTmatrix, MTvector)
    #now to see if the intersection is actually within the triangle, and exclude if not
    mask2=np.logical_and.reduce([tuv[:,0]>Epsilon, tuv[:,1]>Epsilon, tuv[:,2]>Epsilon, tuv[:,1]<1+Epsilon, tuv[:,1]+tuv[:,2]<1+Epsilon])
    #print(mask2.sum())
    #print(O[mask][mask2][0])
    #print(tuv[mask2][:,0])
    #print(D[mask][mask2][0])
    
    #funnily enough we probably don't really need the exact intersection point or anything
    #mask2.sum()/N gives the geometric efficiency, and t will go into ASTAR
    #intersections=np.multiply(np.reshape(tuv[mask2][:,0],(mask2.sum(),1)), D[mask][mask2])+O[mask][mask2] #O+tD
    #print(intersections)
    if mask2.sum()==2: #this decay hits the detector and nothing else, so jot it down somewhere. List of O, D, t
        successes.append([list(O[mask][mask2][0]), list(D[mask][mask2][0]), tuv[mask2][:,0][0]])

successes=np.array(successes, dtype='object')
#print(successes)

print('\nradon done\nstarting with {}, {} Po218s, Po214s\n'.format(Po218s, Po214s))


'''done with radon decays. move to polonium.'''

#now to just see which of the accumulated polonium decays hit the detector
#yes i'm copying and pasting this code yet again. i hope no one has a problem with that

#generate random positions for each polonium.
#N rows, 3 columns. each row identifies an ion, and there is a column for each coordinate
poPos=np.random.uniform([xmin, ymin, zmin], [xmax, ymax, zmax], (Po218s+Po214s,3))
poPos=poPos.astype(np.float16) #i don't need too much precision
#roughly cutting out ions that start outside the cross
xydistsquared=np.power(poPos[:,0], 2)+np.power(poPos[:,1], 2)
success_indices=np.logical_not(np.logical_and.reduce([poPos[:,2]>41, xydistsquared>645]))
poPos=poPos[success_indices]
print((Po218s/(Po218s+Po214s))*len(poPos))

#cutting out stuff in field region. #see comments a few lines down!
#some ions that start here will stick to PIPS and are manually added later in pipsNRG3.py.
#ions that start outside will never go in there.
zs=poPos[:,2]
rhosquareds=np.power(poPos[:,0], 2)+np.power(poPos[:,1], 2)
fieldpomask=np.logical_and.reduce([zs<41.25, rhosquareds<67.24])
fieldpos=fieldpomask.sum()

#commenting this out because the experimental data suggests that not all poloniums are ions and thus they need to be considered even from inside the field region. for now assume none of them are ions and see how this assumption fares (mainly for the reverse field condition. in the deflection field condition things will be a bit different, since po218 especially would not have time to hit anything and gain an electron)

# success_indices=np.logical_not(fieldpomask)
# poPos=poPos[success_indices]

#in reverse field conditions, these poloniums are blown back into the rest of the chamber and thus cannot simply be cut out. thus in pipsNRG4, for reverse field conditions only, mutliply the number of successful poloniums by (1+addifreverse) to account for these that would otherwise have been lost
addifreverse = fieldpos/len(poPos)

poDir=np.random.normal(0,1,(len(poPos),3))
poDir=poDir.astype(np.float16)
poDir=poDir/np.reshape(np.sqrt(np.einsum('ij...,ij->i...',poDir,poDir)), (len(poDir),1))
xydistsquared=np.power(poPos[:,0], 2)+np.power(poPos[:,1], 2)
success_indices=np.logical_not(np.logical_and.reduce([poPos[:,2]>41, xydistsquared>645]))
filter_decays_po()
success_indices=poDir[:,2]<0
filter_decays_po()
floor_intersections=poPos - np.divide(np.multiply(poDir, np.reshape(poPos[:,2], (len(poPos),1))), np.reshape(poDir[:,2], (len(poDir),1)))
success_indices=np.einsum('ij...,ij->i...',floor_intersections,floor_intersections)<95.5 #(squared norm <300/pi)
filter_decays_po()

posuccesses=[]
po218successes=[]
po214successes=[]
i_successes=[]
lenpoPos=len(poPos)
for i in range(lenpoPos):
    if int(10000*i/lenpoPos)%1==0: print("poloniums: {:.2f}%".format(100*i/len(poPos)), end='\r')
    O=np.tile(poPos[i], (len(my_mesh),1))
    D=np.tile(poDir[i], (len(my_mesh),1))
    v1=my_mesh[:,0]
    v2=my_mesh[:,1]
    v3=my_mesh[:,2]
    #first check what triangles are parallel to the ray and exclude them from further calculations
    normals_dcs_dot=np.einsum('ij...,ij->i...',np.cross(v2-v1, v3-v1),D)
    mask=np.logical_or(normals_dcs_dot<-Epsilon, normals_dcs_dot>Epsilon)
    
    #Moeller-Trumbore
    MTmatrix=np.array([-D[mask].T, (v2-v1)[mask].T,(v3-v1)[mask].T]).T
    MTvector=O[mask]-v1[mask]
    #this gives us intersection of ray with the plane of the triangle:
    tuv=np.linalg.solve(MTmatrix, MTvector)
    #now to see if the intersection is actually within the triangle, and exclude if not
    mask2=np.logical_and.reduce([tuv[:,0]>Epsilon, tuv[:,1]>Epsilon, tuv[:,2]>Epsilon, tuv[:,1]<1+Epsilon, tuv[:,1]+tuv[:,2]<1+Epsilon])
    if mask2.sum()==2: #this po decay hits the detector and nothing else, so jot it down somewhere. List of O, D, t
        posuccesses.append([list(O[mask][mask2][0]), list(D[mask][mask2][0]), tuv[mask2][:,0][0]])

posuccesses=np.array(posuccesses, dtype='object')
print('\npoloniums done')
Po218f=int(len(posuccesses)*Po218s/(Po218s+Po214s))
Po214f=int(len(posuccesses)*Po214s/(Po218s+Po214s))
po218successes=posuccesses[:Po218f]
po214successes=posuccesses[-Po214f:]
po218successes=np.array(po218successes, dtype='object')
po214successes=np.array(po214successes, dtype='object')




'''adding some polonium to the underside of the bottom ring to see if plateout from that accurately describes the tail on the po214 we're seeing in the experimental data'''
poPlatePos=np.random.uniform([xmin, ymin, 5-Epsilon], [xmax, ymax, 5], (Po214s,3))
#dont take the starting number of po214s here literally. this number will need to be finetuned later anyways, so we just want a biggish number to get a statistically proper distribution of lengths
poPlatePos=poPlatePos.astype(np.float16)
xydistsquared=np.power(poPlatePos[:,0], 2)+np.power(poPlatePos[:,1], 2)
success_indices=np.logical_not(np.logical_and.reduce([poPlatePos[:,2]>41, xydistsquared>645]))
poPlatePos=poPlatePos[success_indices]
rhosquareds=np.power(poPlatePos[:,0], 2)+np.power(poPlatePos[:,1], 2)
success_indices=np.logical_and.reduce([rhosquareds<(35/2)**2, rhosquareds>(15.88/2)**2]) #restricting to the ring's surface
poPlatePos=poPlatePos[success_indices]
poPlateDir=np.random.normal(0,1,(len(poPlatePos),3))
poPlateDir=poPlateDir.astype(np.float16)
poPlateDir=poPlateDir/np.reshape(np.sqrt(np.einsum('ij...,ij->i...',poPlateDir,poPlateDir)), (len(poPlateDir),1))
success_indices=poPlateDir[:,2]<0
filter_decays_poPlate()
floor_intersections=poPlatePos - np.divide(np.multiply(poPlateDir, np.reshape(poPlatePos[:,2], (len(poPlatePos),1))), np.reshape(poPlateDir[:,2], (len(poPlateDir),1)))
success_indices=np.einsum('ij...,ij->i...',floor_intersections,floor_intersections)<95.5 #(squared norm <300/pi)
filter_decays_poPlate()
poPlatesuccesses=[]
i_successes=[]
lenpoPlatePos=len(poPlatePos)
for i in range(lenpoPlatePos):
    if int(10000*i/lenpoPlatePos)%1==0: print("poloniums: {:.2f}%".format(100*i/len(poPlatePos)), end='\r')
    O=np.tile(poPlatePos[i], (len(my_mesh),1))
    D=np.tile(poPlateDir[i], (len(my_mesh),1))
    v1=my_mesh[:,0]
    v2=my_mesh[:,1]
    v3=my_mesh[:,2]
    normals_dcs_dot=np.einsum('ij...,ij->i...',np.cross(v2-v1, v3-v1),D)
    mask=np.logical_or(normals_dcs_dot<-Epsilon, normals_dcs_dot>Epsilon)
    MTmatrix=np.array([-D[mask].T, (v2-v1)[mask].T,(v3-v1)[mask].T]).T
    MTvector=O[mask]-v1[mask]
    tuv=np.linalg.solve(MTmatrix, MTvector)
    mask2=np.logical_and.reduce([tuv[:,0]>Epsilon, tuv[:,1]>Epsilon, tuv[:,2]>Epsilon, tuv[:,1]<1+Epsilon, tuv[:,1]+tuv[:,2]<1+Epsilon])
    if mask2.sum()==2:
        poPlatesuccesses.append([list(O[mask][mask2][0]), list(D[mask][mask2][0]), tuv[mask2][:,0][0]])
poPlatesuccesses=np.array(poPlatesuccesses, dtype='object')
print('\nplated poloniums done')




'''saving stuff'''
decayDensity=N/volume
num_successes=len(successes)
efficiency=num_successes/N
#output N, volume, decayDensity, num_successes, efficiency, successes
conditions = 'reverse' #field conditions: transfer, deflection, reverse
save('pipsMC6_output_{}_noions_plated'.format(timesteps), 'N', 'volume', 'decayDensity', 'num_successes', 'efficiency', 'successes', 'fielddecays', 'Normalizer', 'Po218s', 'Po214s', 'po218successes', 'po214successes', 'addifreverse', 'poPlatesuccesses')
#load('pipsMC2_output')
