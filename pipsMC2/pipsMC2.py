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

N=int(2**9) #number of decays
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

for i in range(len(decayPos)):
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
    intersections=np.multiply(np.reshape(tuv[mask2][:,0],(mask2.sum(),1)), D[mask][mask2])+O[mask][mask2]
    print(intersections)
    print('\n')
    if mask2.sum()==2: #this decay hits the detector and nothing else, so jot it down somewhere. List of O, D, t
        successes.append([list(O[mask][mask2][0]), list(D[mask][mask2][0]), tuv[mask2][:,0][0]])

successes=np.array(successes, dtype='object')
#print(successes)

decayDensity=N/volume
num_successes=len(successes)
efficiency=num_successes/N
#output N, volume, decayDensity, num_successes, efficiency, successes
save('pipsMC2_output', 'N', 'volume', 'decayDensity', 'num_successes', 'efficiency', 'successes', 'fielddecays', 'Normalizer')
#load('pipsMC2_output')

'''
# Create a new plot
figure = pyplot.figure()
axes = figure.add_subplot(projection='3d')
# ...and add the vectors to the plot
axes.add_collection3d(mplot3d.art3d.Poly3DCollection(my_mesh.vectors))
# Auto scale to the mesh size
scale = my_mesh.points.flatten()
axes.auto_scale_xyz(scale, scale, scale)
# Show the plot to the screen
pyplot.show()
'''