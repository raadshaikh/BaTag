pipsMC#.py reads the stl of the assembly (boil-2-pipsMC2.stl), and based on that it generates decays and sees how many hit the detector. It needs the numpy-stl library.
It outputs a file (pipsMC#_output) that contains a list of positions and distances of all successfully detected decays.

pipsMC2 has only radon
pipsMC3 adds poloniums but using the slow, random decay thing
I will use the calculated activities to speed up the simulation in pipsMC4

pipsNRG#.py (use the latest) reads the previously mentioned output file, computes energies based on stopping power, and generates a plot of the expected energy spectrum histogram.