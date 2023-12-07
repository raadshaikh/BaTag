import numpy as np
from stl import mesh
from matplotlib import pyplot as plt
import pickle

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



# Load the STL files...
my_mesh = mesh.Mesh.from_file('boil-2-pipsMC.stl')
my_mesh=np.reshape(my_mesh, (len(my_mesh),3,3)) #before this, the whole triangle was in a (9,) array. I am splitting the vertices into their own arrays

xmin, xmax = -68.82, 68.82 #bounding box for volume under consideration
ymin, ymax = -25.42, 25.42
zmin, zmax = 0.65, 130
xrange=xmax-xmin
yrange=ymax-ymin
zrange=zmax-zmin
volume=xrange*yrange*zrange
detectR = 9.77 #radius of circular active region on detector. centered at origin, in xy plane.
alpha_energy_init=5.5904 #MeV. idk, why not just put it in here

N=int(2**21) #number of radon decays
print('\nstarting with {} radon decays'.format(N))
#N=int(4000*volume) #24000 Rn/cc, over 1 day, means 4000 decays/cc
Normalizer=4000*volume/N #This is used to scale the histogram later so it looks like we started out with 4000 decays/cc

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

zs=decayPos[:,2]
rhosquareds=np.power(decayPos[:,0], 2)+np.power(decayPos[:,1], 2)
fielddecaymask=np.logical_and.reduce([zs>5, zs<41.25, rhosquareds<67.24])
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

print('\nradon done\n')


'''done with radon decays. move to polonium.'''

#MFP=112.2e-6 #mean free path for argon at 700mb, in mm
#poVel=180e-3 #polonium ion velocity at room temp, mm/s. works for anything with mass 214u
#dt=MFP/poVel #time between collisions, seconds. do we need it this fine?
dt=2 #timestep in seconds
timesteps=int(24*60*60/dt) #one whole day
#timesteps=int(3*60*60/dt) #3 hours, start small
Po218L=np.log(2)/(3.05*60) #decay constants, inverse seconds
Pb214L=np.log(2)/(26.8*60)
Bi214L=np.log(2)/(19.7*60)
Po214L=np.log(2)/(0.16e-3)

#need to keep track of ion position, species
#iterate over many short time periods. at start of the time period, flip a coin for each ion and decay it or not based on species
#if a polonium decays, add it to a list so we know where the alpha is coming from
#once 24hrs have passed, we have a list of ion decay positions corresponding to poloniums and the iteration is over.
#now generate random directions for these decays and check if they hit detector or not

Po218s=0
Po214s=0
Po218mask=np.ones(N, dtype=bool)
Pb214mask=np.zeros(N, dtype=bool)
Bi214mask=np.zeros(N, dtype=bool)
Po214mask=np.zeros(N, dtype=bool)

#once you're done figuring out what happens in each time period, indent all of it and add a for loop over time right here.
#ignore that comment, i already did it
for j in range(timesteps):
    if int(100*j/timesteps)%1==0: print("\rion decays: {:.2f}%".format(100*j/timesteps), end='\r')
    #for all ions, draw a U(0,1). if this is smaller than lambda*dt, the ion decays.
    coin=np.random.uniform(size=N)
    coincompare=(Po218L*Po218mask + Pb214L*Pb214mask + Bi214L*Bi214mask + Po214L*Po214mask)*dt
    decaymask=coin<coincompare #list of indices pointing out which ions decay in this time step. based on this, update the species masks
    Po218s+=np.logical_and(Po218mask, decaymask).sum() #jotting down polonium decays
    Po214s+=np.logical_and(Po214mask, decaymask).sum()
    #updating species masks:
    Po214mask=np.logical_and(Po214mask, np.logical_not(decaymask))#removing Po214s that decay
    Po214mask=np.logical_or(Po214mask, np.logical_and(Bi214mask, decaymask))#adding to Po214s the decays from Bi214
    Bi214mask=np.logical_and(Bi214mask, np.logical_not(decaymask))
    Bi214mask=np.logical_or(Bi214mask, np.logical_and(Pb214mask, decaymask))
    Pb214mask=np.logical_and(Pb214mask, np.logical_not(decaymask))
    Pb214mask=np.logical_or(Pb214mask, np.logical_and(Po218mask, decaymask))
    Po218mask=np.logical_and(Po218mask, np.logical_not(decaymask))    
    #end the time loop. we have accumulated a bunch of polonium decays

print('\nions decayed\nstarting with {}, {} Po218s, Po214s\n'.format(Po218s, Po214s))

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
#cutting out stuff in field region.
#some ions that start here will stick to PIPS and are manually added later in pipsNRG3.py.
#ions that start outside will never go in there.
zs=poPos[:,2]
rhosquareds=np.power(poPos[:,0], 2)+np.power(poPos[:,1], 2)
fieldpomask=np.logical_and.reduce([zs>5, zs<41.25, rhosquareds<67.24])
fieldpos=fieldpomask.sum()
success_indices=np.logical_not(fieldpomask)
poPos=poPos[success_indices]

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


'''saving stuff'''
decayDensity=N/volume
num_successes=len(successes)
efficiency=num_successes/N
#output N, volume, decayDensity, num_successes, efficiency, successes
save('pipsMC4_output', 'N', 'volume', 'decayDensity', 'num_successes', 'efficiency', 'successes', 'fielddecays', 'Normalizer', 'po218successes', 'po214successes')
#load('pipsMC2_output')
