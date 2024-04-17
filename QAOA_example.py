import numpy as np
from docplex.mp.model import Model
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit_optimization.translators import from_docplex_mp
from qiskit.primitives import Sampler

# Define the problem
cars = np.array([1,1,1,0,0,0,0,1,1,1,1,1,0,0])

# Define the model
mdl = Model(name='Optimize Car Paint Shop')

# Define the variables
x = mdl.binary_var_list(len(cars), name='x')

# Define the objective function
mdl.minimize(mdl.sum(x[i] * (1 - x[i+1]) + (1 - x[i]) * x[i+1] for i in range(len(cars)-1)))

# Convert the problem to a QUBO problem
qp = from_docplex_mp(mdl)
qubo = QuadraticProgramToQubo().convert(qp)


#Solve the Problem using QAOA
sampler = Sampler()
sampler.set_options(shots=2^15,seed=42)
    
cobyla = COBYLA()
qaoa_mes = QAOA(sampler,optimizer=cobyla, reps=3)

qaoa = MinimumEigenOptimizer(qaoa_mes)
result = qaoa.solve(qubo)

# Print the result
print('Optimal solution:', result.x)
print('Optimal solution:', result)