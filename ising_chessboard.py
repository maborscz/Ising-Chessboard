import matplotlib.pyplot as plt
import scipy as sp
from scipy import interpolate
import numpy as np
from math import exp
from random import randrange, random, choice

rng = np.random.default_rng()

class Lattice:
    def __init__(self, N, T, B=0, start='Low'):
        
        self.N = N # size of lattice
        self.B = B # strength of magnetic field
        self.T = T # temperature

        if start == 'High':
            self.array = rng.choice([-1, 1], size=(self.N, self.N)) # random spins
        elif start == 'Low':
            self.array = np.ones((self.N, self.N), dtype=np.int64) # spins all up
        
        self.energy = self.latticeEnergy() # energies of each atom
        

    # calculates the energy of each atom according to Ising Model
    def latticeEnergy(self):
        up = np.roll(self.array, (-1, 0), (0, 1))
        down = np.roll(self.array, (1, 0), (0, 1))
        left = np.roll(self.array, (0, -1), (0, 1))
        right = np.roll(self.array, (0, 1), (0, 1))

        return (-1.0 * self.array * (up + down + left + right + self.B))


    # Modified the metropolis algorithm for faster convergence. Algorithm
    # probabilistically flips spins that lie on a 'chessboard' pattern, since they
    # do not interact according to the nearest-neighbour assumption in the Ising model

    def monteCarlo(self, steps):

        self.Mlist = [] # ensemble for magnetic susceptibility calcs

        for i in range(steps):
            prob = np.exp(2 * self.energy/self.T) # flipping probability for each atom
            mask1 = rng.random(size=(self.N, self.N)) < prob # randomly choose atoms to flip

            # create chessboard pattern, which alternates between 'black' and 'white' each
            # iteration
            mask2 = np.zeros((self.N, self.N)) + (i % 2)
            mask2[1::2, ::2] = 1 - (i % 2)
            mask2[::2, 1::2] = 1 - (i % 2)

            # flip spins that are in chessboard and have been randomly chosen
            self.array[np.where(mask1 * mask2)] *= -1
            self.energy = self.latticeEnergy() # calculate energies of new lattice

            # create list of magnetisations for last tenth of iterations
            if steps - i <= (steps // 10):
                self.Mlist.append(np.sum(self.array))
        
        self.Mlist = np.array(self.Mlist)


class Plots:
    def __init__(self, N=10, B=0, start='Low', t0=1, t1=5, inc=0.1, steps=100, label=None):
        self.N = N # size of lattice
        self.B = B # strength of magnetic field
        self.start = start # initialisation of lattice
        self.inc = inc # size of increments in plots
        self.t0 = t0 # initial plot temperature
        self.t1 = t1 # final plot temperature
        self.steps = steps # number of Monte Carlo iterations
        self.label = label # plot label

    def calcMagSus(self, array, T):
        av2 = np.mean(array)**2 # (average of M)^2
        av_2 = np.mean(array ** 2) # average of (M^2)
        return (av_2 - av2) / (self.N**2 * T) # magnetic susceptibility per atom
    
    
    def tempPlot(self): 
        plt.xlabel(self.labelx)
        plt.ylabel(self.labely)
        plt.title(f'atoms: {self.N}, steps: {self.steps},\nT inc. by {self.inc}')
        plt.scatter(np.arange(self.t0, self.t1, self.inc), self.data, label=self.label)
        plt.legend()

    def exp(self, var):
        self.data = []
        for T in np.arange(self.t0, self.t1, self.inc):
            L = Lattice(self.N, T, start=self.start, B=self.B)
            L.monteCarlo(self.steps) 

            if var == 'mag':
                self.data.append(np.abs(np.mean(L.array)))
                self.labely = 'Magnetization per atom'
            elif var == 'energy':
                self.data.append(np.mean(L.energy))
                self.labely = 'Energy per atom'
            elif var == 'mag sus':
                self.data.append(self.calcMagSus(L.Mlist, T))
                self.labely='Magnetic susceptibility per atom' 

        self.labelx = 'Temperature in units of kB/J'
        self.tempPlot()


    def latticePlot(self, T):
        L = Lattice(self.N, T, start=self.start, B=self.B)
        L.monteCarlo(self.steps)
        plt.imshow(L.array)
        plt.title(f'atoms: {self.N}, steps: {self.steps}, M field: {self.B},\n' \
                  f'T: {T}, init. at a {self.start} temperature')
        plt.show()
