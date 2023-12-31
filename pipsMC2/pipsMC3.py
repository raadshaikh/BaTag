import numpy as np
from stl import mesh
from mpl_toolkits import mplot3d
from matplotlib import pyplot
import pickle

#utilities

Epsilon=2**-20 #idk smth small
#def normalize_each(vectors):
#    return vectors/np.reshape(np.sqrt(np.diag(np.inner(vectors,vectors))), (len(vectors),1))
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
    Po218mask=Po218mask[success_indices]

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
#print('volume:', xrange*yrange*zrange*1e-3, 'cc') #bounding box is 577 ml
detectR = 9.77 #radius of circular active region on detector. centered at origin, in xy plane.
alpha_energy_init=5.5904 #MeV. idk, why not just put it in here

N=int(2**20) #number of decays
#N=int(4000*volume) #24000 Rn/cc, over 1 day, means 4000 decays/cc
#assert(N<2e4)
Normalizer=4000*volume/N

#generate random positions and direction rays for each decay.
#N rows, 3 columns. each row identifies a decay, and there is a column for each coordinate
decayPos=np.random.uniform([xmin, ymin, zmin], [xmax, ymax, zmax], (N,3))
decayPos[N-1]=[0,0,0.7] #adding a decay right above the detector so during testing it doesn't all get washed out
decayPos[N-1]=[9, 0, 1.0]
decayDir=np.random.normal(0,1,(N,3))
decayDir[N-1]=[0,0,-1] #making sure that artificial point is guaranteed to hit the detector
decayPos=decayPos.astype(np.float16) #i don't need too much precision
decayDir=decayDir.astype(np.float16)
#roughly cutting out decays that start outside the cross
xydistsquared=np.power(decayPos[:,0], 2)+np.power(decayPos[:,1], 2)
success_indices=np.logical_not(np.logical_and.reduce([decayPos[:,2]>41, xydistsquared>2500]))
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
    print("{:.1f}%".format(100*i/lendecayPos))
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
    #mask2.sum()/N gives the geometric efficiency, and t will go into SRIM
    #intersections=np.multiply(np.reshape(tuv[mask2][:,0],(mask2.sum(),1)), D[mask][mask2])+O[mask][mask2]
    #print(intersections)
    if mask2.sum()==2: #this decay hits the detector and nothing else, so jot it down somewhere. List of O, D, t
        successes.append([list(O[mask][mask2][0]), list(D[mask][mask2][0]), tuv[mask2][:,0][0]])

successes=np.array(successes, dtype='object')
#print(successes)


print('radon done\n')

'''done with radon decays. move to polonium.'''

#generate random positions and direction rays for each ion.
#N rows, 3 columns. each row identifies an ion, and there is a column for each coordinate
ionPos=np.random.uniform([xmin, ymin, zmin], [xmax, ymax, zmax], (N,3))
ionDir=np.random.normal(0,1,(N,3))
ionPos=ionPos.astype(np.float16) #i don't need too much precision
ionDir=ionDir.astype(np.float16)
#roughly cutting out ions that start outside the cross
xydistsquared=np.power(ionPos[:,0], 2)+np.power(ionPos[:,1], 2)
success_indices=np.logical_not(np.logical_and.reduce([ionPos[:,2]>41, xydistsquared>645]))
ionPos=ionPos[success_indices]
ionDir=ionDir[success_indices]
#cutting out stuff in field region
zs=ionPos[:,2]
rhosquareds=np.power(ionPos[:,0], 2)+np.power(ionPos[:,1], 2)
fieldionmask=np.logical_and.reduce([zs>5, zs<41.25, rhosquareds<67.24])
fieldions=fieldionmask.sum()
success_indices=np.logical_not(fieldionmask)
ionPos=ionPos[success_indices]
ionDir=ionDir[success_indices]

ionDir=ionDir/np.reshape(np.sqrt(np.einsum('ij...,ij->i...',ionDir,ionDir)), (len(ionDir),1))

MFP=112.2e-6 #mean free path for argon at 700mb, in mm
poVel=180e-3 #polonium ion velocity at room temp, mm/s. works for anything with mass 214u
dt=MFP/poVel #time between collisions, seconds
#dt=1
Po218L=np.log(2)/(3.05*60) #decay constants, inverse seconds
Pb214L=np.log(2)/(26.8*60)
Bi214L=np.log(2)/(19.7*60)
Po214L=np.log(2)/(0.16e-3)

#need to keep track of ion position, direction, species, and whether or not it's stuck
#iterate over many short time periods. at start of the time period, flip a coin for each ion and decay it or not based on species
#if a polonium decays, add it to a list of decayPos so we know where the alpha is coming from
#then iterate over all non-stuck ions and update position based on direction and speed
#once 24hrs have passed, we have a list of decayPos corresponding to poloniums and the iteration is over.
#now generate random directions for these decays and check if they hit detector or not

lenionPos=len(ionPos)

Po218s=np.array([])
Po214s=np.array([])
stuck=np.zeros(lenionPos, dtype=bool)
Po218mask=np.ones(lenionPos, dtype=bool)
Pb214mask=np.zeros(lenionPos, dtype=bool)
Bi214mask=np.zeros(lenionPos, dtype=bool)
Po214mask=np.zeros(lenionPos, dtype=bool)

#once you're done figuring out what happens in each time period, indent all of it and add a for loop over time right here.
timesteps=int(24*60*60/dt) #one whole day
#timesteps=50 #start small i guess
timesteps=int(3*60*60/dt) #1 hour's simulation took 11 hours
for j in range(timesteps):
    print("{:.2f}%".format(100*j/timesteps))
    #at beginning of each time step the ion collided with an argon, so give it a new random direction.
    ionDir=np.random.normal(0,1,(lenionPos,3))
    ionDir=ionDir.astype(np.float16)
    ionDir=ionDir/np.reshape(np.sqrt(np.einsum('ij...,ij->i...',ionDir,ionDir)), (len(ionDir),1))

    #for all ions, draw a U(0,1). if this is smaller than lambda*dt, the ion decays.
    coin=np.random.uniform(size=lenionPos)
    coincompare=(Po218L*Po218mask + Pb214L*Pb214mask + Bi214L*Bi214mask + Po214L*Po214mask)*dt
    decaymask=coin<coincompare #list of indices pointing out which ions decay in this time step. based on this, update the species masks
    Po218s=np.append(Po218s, ionPos[np.logical_and(Po218mask, decaymask)]) #jotting down polonium decays
    Po214s=np.append(Po214s, ionPos[np.logical_and(Po214mask, decaymask)])
    #updating species masks:
    Po214mask=np.logical_and(Po214mask, np.logical_not(decaymask))#removing Po214s that decay
    Po214mask=np.logical_or(Po214mask, np.logical_and(Bi214mask, decaymask))#adding to Po214s the decays from Bi214
    Bi214mask=np.logical_and(Bi214mask, np.logical_not(decaymask))
    Bi214mask=np.logical_or(Bi214mask, np.logical_and(Pb214mask, decaymask))
    Pb214mask=np.logical_and(Pb214mask, np.logical_not(decaymask))
    Pb214mask=np.logical_or(Pb214mask, np.logical_and(Po218mask, decaymask))
    Po218mask=np.logical_and(Po218mask, np.logical_not(decaymask))
    
    #update non-stuck ions' positions and see if they stick:
    ionPosNS=ionPos[~stuck]
    ionDirNS=ionDir[~stuck]
    ionPosNS+=ionDirNS*MFP #update position based on direction and mean free path
    xydistsquared=np.power(ionPos[:,0], 2)+np.power(ionPos[:,1], 2)
    outside=np.logical_and.reduce([ionPos[:,2]>41, xydistsquared>564])
    stuck=np.logical_or(stuck, outside)
    '''#for i in range(np.logical_not(stuck).sum()):
        #if i%10==0: print(i)
        #this is taking wayyyy too long so I'm just gonna approximate the geometry. sorry.
        O=np.tile(ionPos[np.logical_not(stuck)][i], (len(my_mesh),1)) #if something goes wrong, investigate this. is it in place, is the indexing okay, etc.
        D=np.tile(ionDir[np.logical_not(stuck)][i], (len(my_mesh),1))
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
        if np.any(tuv[mask2][:,0]<MFP): #if any of the triangles are less than MFP away from the ion in the direction of its travel, it's gonna hit and stick
            stuck[i]=True
    '''
    #if it ended up inside the field region we gotta get it out cuz it's not allowed in there
    zs=ionPosNS[:,2]
    rhosquareds=np.power(ionPosNS[:,0], 2)+np.power(ionPosNS[:,1], 2)
    fieldionmask=np.logical_and.reduce([zs>5, zs<41.25, rhosquareds<17.5**2])
    #ionPosNS[~fieldionmask]-=2*ionDirNS[~fieldionmask]*MFP
    
    #end the time loop. we have accumulated a bunch of polonium decays while also letting ions stick when they hit a wall.


#now to just see which of the accumulated polonium decays hit the detector
#yes i'm copying and pasting this code yet again. i hope no one has a problem with that
print('ions decayed\n',len(Po218s), len(Po214s))
Po218mask=np.append(np.ones(int(len(Po218s)/3),dtype=bool),np.zeros(int(len(Po214s)/3),dtype=bool))
poPos=np.append(Po218s, Po214s)
poPos=np.reshape(poPos, (int(len(poPos)/3),3))
poDir=np.random.normal(0,1,(len(poPos),3))
poDir=poDir.astype(np.float16)
poDir=poDir/np.reshape(np.sqrt(np.einsum('ij...,ij->i...',poDir,poDir)), (len(poDir),1))
success_indices=poDir[:,2]<0
filter_decays_po()
floor_intersections=poPos - np.divide(np.multiply(poDir, np.reshape(poPos[:,2], (len(poPos),1))), np.reshape(poDir[:,2], (len(poDir),1)))
success_indices=np.einsum('ij...,ij->i...',floor_intersections,floor_intersections)<95.5 #(squared norm <300/pi)
filter_decays_po()
posuccesses=[]
po218successes=[]
po214successes=[]
i_successes=[]
for i in range(len(poPos)):
    print("{:.1f}%".format(100*i/len(poPos)))
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
        i_successes.append(Po218mask[i])
Po218mask=np.array(i_successes)
posuccesses=np.array(posuccesses, dtype='object')
print('poloniums done')
#print(Po218mask)
#print(posuccesses)
if Po218mask.size>0:
    po218successes=posuccesses[Po218mask]
    po214successes=posuccesses[np.logical_not(Po218mask)]
po218successes=np.array(po218successes, dtype='object')
po214successes=np.array(po214successes, dtype='object')


'''saving stuff'''
decayDensity=N/volume
num_successes=len(successes)
efficiency=num_successes/N
#output N, volume, decayDensity, num_successes, efficiency, successes
save('pipsMC3_output', 'N', 'volume', 'decayDensity', 'num_successes', 'efficiency', 'successes', 'fielddecays', 'Normalizer', 'po218successes', 'po214successes')
#load('pipsMC2_output')
