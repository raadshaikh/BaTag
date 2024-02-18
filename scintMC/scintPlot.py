import numpy as np
import pickle
import matplotlib.pyplot as plt

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

N=556982
M=1
option=2
stlname='detector_chamber4_cut_noTefl.stl'
load('scintMC3_{}_output{}_{}x{}'.format(stlname[18:-4],option,N,M))
print(sipm_intersections)
x=sipm_intersections[:,0]
y=sipm_intersections[:,1]
z=sipm_intersections[:,2]
plt.subplot(311)
plt.hist(x)
plt.subplot(312)
plt.hist(y)
plt.subplot(313)
plt.hist(z)
plt.show()
exit()

#should have saved this from scintMC2 but i'm not running that again right now
big=16
load('scintNRG_output_{}'.format(big))
N=int(np.sum(np.rint(energies/np.min(energies))))
positions=lengths+dx/2
positions=10*positions #scintNRG dealt in cm, here i work with mm
weights=np.int_(np.rint(energies/np.min(energies)))
decayPosZ=np.repeat(positions, weights)

x=teflon_intersections[:,0]
y=teflon_intersections[:,1]
z=teflon_intersections[:,2]

r=np.sqrt(x**2+y**2)
theta=np.arctan2(y,x)
theta=theta*180/np.pi

plt.suptitle(stlname)
plt.subplot(411)
plt.xlim((0,67))
plt.hist(decayPosZ, label='z all photons (mm)')
plt.legend()
plt.subplot(412)
plt.xlim((0,67))
plt.hist(decayPos[:,2], label='z successful photons')
plt.legend()
plt.subplot(413)
plt.xlim((0,67))
plt.hist(z, label='z teflon hits')
plt.legend()
plt.subplot(414)
plt.hist(theta, label='theta teflon hits (deg)')
plt.legend()

'''plt.scatter(theta, z, alpha=0.7, s=1)
plt.xlabel('$theta$ (deg)')
plt.ylabel('z (mm)')
plt.show()'''
plt.savefig('scintPlot_{}_output{}_{}x{}.png'.format(stlname[18:-4],option,N,M), bbox_inches='tight')
plt.show()