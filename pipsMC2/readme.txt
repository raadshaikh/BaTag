pipsMC#.py reads the stl of the assembly (boil-2-pipsMC2.stl), and based on that it generates decays and sees how many hit the detector. It needs the numpy-stl library.
It outputs a file (pipsMC#_output) that contains a list of positions and distances of all successfully detected decays.

pipsMC2 has only radon
pipsMC3 adds poloniums but using the slow, random decay thing and attempts to simulate ions sticking to walls (very slow)
pipsMC4 ignores the wall sticking but still uses the random decay
I will use the calculated activities to speed up the simulation in pipsMC5. (done)
pipsMC6: separating po218 and po214, and adding po210. this will go to pipsNRG4 which will read in existing data to compare so we can adjust starting amounts manually
actually i just realised that the poloniums don't need to be separated in the code. now that i think about it radon didn't need to be separated like that either but i guess it's just stuck like that now

pipsNRG#.py (use the latest) reads the previously mentioned output file, computes energies based on stopping power, and generates a plot of the expected energy spectrum histogram.


So basically run pipsMC5.py and then pipsNRG3.py. (or 6 and 4 now)
Before running pipsMC you may want to adjust the variables N, dt, timesteps.