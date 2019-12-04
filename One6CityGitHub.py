# Originated by Richard H. Warren, October 2019, email: rhw3@psu.edu
# Finds optimal tours classically and on D-Wave for one 6-city asymmentric traveling salesman problem (TSP)
# Classically places all optimal tours in a set T and then finds lowest energy solutions on D-Wave.
# Purpose is to determine if D-Wave lowest energy solutions are optimal tours.  Comparison of classical optimal tours and D-Wave so;utions is not automated.

# Results from executing this script are reported in "Optimal Solutions of 6-City TSPs" which is a chapter in the book
# "A Closer Look at the Travelling Salesman Problem" to be published in 2020 by Nova Science Publishers, Inc.

# Classicaly finds all optimal tours for an asymmetric 6-city TSP. Distances are 0, except for 1 distance = -50
# Writes to screen classical results: distance matrix, length of optimal tour, all optimal tours, and number of optimal tours
# Places optimal tours in a set called T. Each tour has 1 in first position.

# Variables that are hard coded: N = 6, positions in a tour x2,...,x6 (x1 = 1 and xi is the number of a city in position i of a tour)
# Print statements with 1 to 6 are hard coded

N = 6 # N is number of cities. Need to loop on N + 1 = 7, as pointed out page 66 "Fundamentals of Python" by Lambert

# Create distance matrix D with all 0s as integers, according to pages 370-373 in Lubanovic "Introducing Python"
import numpy as np
D = np.zeros((N + 1,N + 1), dtype=int) #See pages 5-7 in Numpy User Guide

# Using array for D because arrays are easier to handle than dictionaries or lists.
# To overcome that arrays begin with index 0, will create arrays of size one
# larger than needed and use elements indexed 1 to 6, and not use any elements with index 0.

# Change distance matrix D to an example for which I know the optimal tours

D[1,6] = -50  # There are 24 optimal tours with distance -50 and 96 tours with distance 0

# Print distance matrix

for k in range (1,N + 1):
    print(D[k,1], D[k,2], D[k,3], D[k,4], D[k,5], D[k,6])

# Next, find length of optimal tour. Number of cities is hard coded at 6.

# Initialize 3 variables

Length = 9999  # length of a tour
ShortestLength = 10000  # shortest length found so far
NumberOfOptimalTours = 0 # number of shortest length tours, which at the end is the number of optimal tours

# Find a tour. It will be (1,x2,x3,x4,x5,x6)

for x2 in range (2, N + 1):
    for x3 in range (2, N + 1):
        if (x2 == x3): continue
        for x4 in range (2, N + 1):
            if (x2 == x4): continue
            if (x3 == x4): continue
            for x5 in range (2, N + 1):
                if (x2 == x5): continue
                if (x3 == x5): continue
                if (x4 == x5): continue
                for x6 in range (2, N + 1):
                    if (x2 == x6): continue
                    if (x3 == x6): continue
                    if (x4 == x6): continue
                    if (x5 == x6): continue
                    Length = D[1,x2] + D[x2,x3] + D[x3,x4] + D[x4,x5] + D[x5,x6] + D[x6,1] # Find length of tour that was found
                    if (Length < ShortestLength): # Compare length of tour to shortest length
                        ShortestLength = Length
                        NumberOfOptimalTours = 0 # Re-initialize. Number will be incremented for current tour in next if statement
                        print("New shortest length", Length, "tour is (1", x2, x3, x4, x5, x6,")")
                    if (Length == ShortestLength):
                        NumberOfOptimalTours = NumberOfOptimalTours + 1

print("Number of optimal tours =", NumberOfOptimalTours, "Last tour shown above is optimal")

T = set()  # create an empty set to hold optimal tours found classically, each having city 1 in position 1
for x2 in range (2, N + 1):
    for x3 in range (2, N + 1):
        if (x2 == x3): continue
        for x4 in range (2, N + 1):
            if (x2 == x4): continue
            if (x3 == x4): continue
            for x5 in range (2, N + 1):
                if (x2 == x5): continue
                if (x3 == x5): continue
                if (x4 == x5): continue
                for x6 in range (2, N + 1):
                    if (x2 == x6): continue
                    if (x3 == x6): continue
                    if (x4 == x6): continue
                    if (x5 == x6): continue
                    Length = D[1,x2] + D[x2,x3] + D[x3,x4] + D[x4,x5] + D[x5,x6] + D[x6,1]
                    if (Length == ShortestLength):
                        print("an optimal tour is 1", x2, x3, x4, x5, x6)
                        T.add((1, x2, x3, x4, x5, x6))
print(T)  # print the tours in T. Each is optimal classiically

# END OF CLASSICAL WORK.  BEGIN QUANTUM COMPUTATION

from dwave.system.samplers import DWaveSampler    # copied from "Getting started with the D-Wave System" 8.1.2
from dwave.system.composites import EmbeddingComposite  # copied from "Getting started with the D-Wave System" 8.1.2
my_sampler = EmbeddingComposite(DWaveSampler())

# Set tunable parameter gamma. See notes in D-Wave 48-city TSP
gamma = 250
sampler = DWaveSampler(solver='DW_2000Q_5')  # set solver to lower noise. It is alco set in config file.
# chain strength is set in response as directed by David Johnson
# number of reads is set in response
# auto-scaling is on unless it is explicity turned off. Getting Started with the D-Wave System -> Defining a Simple Problem -> Scaling (It is needed)
# optimization post processing is off unless it is explicity turned on. Postprocessing Methods -> Chapter Two line 4 (Do not seem to need it because scaling factor in Figure 4.2 is .1 or larger)

# The Boolean varialbes are y(i,j) designating city i in position j of a tour. Cites and positions in a tour are numbered 1, 2, ..., 6
# City 1 is in position 1.
# Each Boolena variable y(i,j) has a row and column in Q.
Q = {}  # D-Wave requires Q to be a dictionary
for i in range (N*N): # range for i is 0 to (6*6)-1 = {0, 1, ..., 35}.
    for j in range (N*N):
        Q[(i,j)] = 20  # reaised background of Q from 0 to 20
# print("Number of entries in Q is", len(Q), "Number of entries in D is", D.size)   # len(Q) = 1296 = 36 x 36, D.size = 7 x 7 = 49

# Print portion of Q
#for k in range (1,N + 1):
#    print(Q[(k,1)], Q[(k,2)], Q[(k,3)], Q[(k,4)], Q[(k,5)], Q[(k,6)])
 
# Since our Boolean variables have two indices, a helper function is provided to assign y(a,b) to a specific
# row/column index in Q. The function to compute the index in Q for a variable y(a,b):
def y(a,b):  # a is a city 1, 2, 3, ..., 6.  b is a location in a tour 1, 2, 3, ..., 6.
    return ((a-1)*N) + (b-1) # (1,1) -> 0  (1,2) -> 1 (1,3) -> 2 (1,4) -> 3 (1,5) -> 4 (1,6) -> 5
                             # (2,1) -> 6  (2,2) -> 7 (2,3) -> 8 ... (2,6) -> 11
                             # ...
                             # (5,1) -> 24 (5,2) -> 25 (5,3) -> 26 (5,4) -> 27 (5,5) -> 28 (5,6) -> 29
                             # (6,1) -> 30 (6,2) -> 31 (6,3) -> 32 (6,4) -> 33 (6,5) -> 34 (6,6) -> 35

# OBJECTIVE FUNCTION based on Warren, Adapting the traveling salesman problem to an adiabatic quantum computer, Quantum Information Processing 2013, 12, 1781–1785
# y(1,1) is fixed. City 1 is in position 1
# first term: j is a city in position 2. Thus, 2 <= j <= 6
# first term accounts for distance from city 1 to the city in position 2
for j in range (2,N+1):
    Q[(y(1,1),y(j,2))] = D[1,j]

# last term: i is a city in position 6. Thus, 2 <= i <= 6
# last term accounts for distance from city in position 6 to city 1
for i in range (2,N+1):
    Q[(y(i,6),y(1,1))] = D[i,1]

#print(Q[(11,0)], Q[(17,0)], Q[(23,0)], Q[(29,0)], Q[(35,0)])

# terms between first and last. u and v are cities. j is a position in a tour
for u in range (2,N+1):    # range for a city is 2, 3, 4, 5, 6
    for v in range (2,N+1): 
        if u != v:
            for j in range (2,N):   # range for a positon in a tour is 2, 3, 4, 5, since 6 = 5 + 1 in next line for j = 5
                Q[y(u,j),y(v,j+1)] = D[u,v]  # changed from +=

#for v in range (2,N+1): 
#    if v != 3:
#        print(Q[y(3,2),y(v,3)], Q[y(3,3),y(v,4)], Q[y(3,4),y(v,5)], Q[y(3,5),y(v,6)])

# PENALTY FUNCTIONS, squared because each is a sum = 1, based on Warren, Adapting the traveling salesman problem to an adiabatic quantum computer. Quantum Information Processing
# For each city v, except city 1, ensure it occurs in exactly one position of a tour
for v in range (2,N+1):
    for j in range (2,N+1):
        Q[(y(v,j),y(v,j))] = -1*gamma
for v in range (2,N+1):
    for j in range (2,N+1):
        for k in range (2,N+1):
            if j != k:
                Q[(y(v,j),y(v,k))] = 2*gamma
Q[(y(1,1),y(1,1))] = -1*gamma
for v in range (2,N+1):
    Q[(y(v,1),y(v,1))] = 2*gamma

# For each positon j in a tour, except first position, ensure that it has exactly one city
for j in range (2,N+1):
    for v in range (2,N+1):
        Q[(y(v,j),y(v,j))] = -1*gamma
for j in range (2,N+1):
    for v in range (2,N+1):
        for w in range (2,N+1):
            if v != w:
                Q[(y(v,j),y(w,j))] = 2*gamma  

# Ensure position 1 has city 1
#Q[(y(1,1),y(1,1))] += -1*gamma  # this line is deleted because it duplicates a line above for city 1 in position 1
# Ensure positions 2 - 6 do not have city 1
for j in range (2,N+1):
    Q[(y(1,j),y(1,j))] = 2*gamma

# Minor-embed and sample up to 1000 times on a default D-Wave system
response = EmbeddingComposite(DWaveSampler()).sample_qubo(Q, num_reads=1000, chain_strength=1000)
for sample, energy, num_occurrences, chain_break_fraction in response.data():
    print(sample, "Energy: ", energy, "Occurrences: ", num_occurrences, "Chain Break Fraction: ", chain_break_fraction)
# last 3 lines are from "Getting Started8_1_2.py" which are a correction to box in Section 8.1.2 "Getting Started with the D-Wave Sustem"
# Recommend 2 enhancements: put lowest energy tours in a set Z, compare T and Z as indicated below
#                           loop on 100 executions of the TSP and compute statistics
# print(T - Z) # pirnt the classical optimal tours that are not lowest energy tours.
# print(Z - T) # pirnt the lowest energy tours that are not optimal classiically.
