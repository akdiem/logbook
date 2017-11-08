# -*- coding: utf-8 -*-

from fenics import *
from dolfin import *
from mshr import *

from matplotlib import pyplot as plt
import numpy as np

parameters["plotting_backend"] = "matplotlib"
prm = parameters['krylov_solver']
prm['absolute_tolerance'] = 1e-7
prm['relative_tolerance'] = 1e-4
prm['maximum_iterations'] = 1000

# Units
cm = 1e-2
um = 1e-4 * cm
dyn = 1
pa = 10 * dyn/cm**2
s = 1

# Scaled variables
r0 = 20*um
R = r0/r0
We = 0.2*R
w_ast = 10*um/r0
gap = 1*um/r0
Le = 10*R + 5*w_ast + 4*gap
tf = 50*s

print(Le*r0/um)


asta1 = Le/2 - 5*w_ast
asta2 = asta1+w_ast
astb1 = asta2+gap
astb2 = astb1+w_ast
astc1 = astb2+gap
astc2 = astc1+w_ast
astd1 = astc2+gap
astd2 = astd1+w_ast
aste1 = astd2+gap
aste2 = aste1+w_ast

Y = 1.0e6 * pa
nu = 0.49
lam = Y*nu/((1+nu)*(1-2*nu))
mu = Y/(2*(1+nu))
Mu = mu/mu
Lam = lam/mu

disp = np.loadtxt('./data/disp.csv', delimiter=',')/r0
nt = disp.shape[0]
dt = tf/nt

# Create mesh and define function space
N = 512
deg = 2
elem = "Lagrange"
sim = "rx5_5"
geom = Rectangle(Point(0, 0), Point(We, Le))
mesh = generate_mesh(geom, N)

V = VectorFunctionSpace(mesh, elem, deg)
W = FunctionSpace(mesh, elem, deg)
WW = TensorFunctionSpace(mesh, elem, deg)

# Define boundaries
def astr_a(x, on_boundary):
    return near(x[0], We) and (x[1] < asta2 and x[1] > asta1)

def astr_b(x, on_boundary):
    return near(x[0], We) and (x[1] < astb2 and x[1] > astb1)

def astr_c(x, on_boundary):
    return near(x[0], We) and (x[1] < astc2 and x[1] > astc1)

def astr_d(x, on_boundary):
    return near(x[0], We) and (x[1] < astd2 and x[1] > astd1)

def astr_e(x, on_boundary):
    return near(x[0], We) and (x[1] < aste2 and x[1] > aste1)

def fixed_boundary(x, on_boundary):
    return near(x[1], 0) or near(x[1], Le)

disps = Expression(('d', '0'), d=disp[0], degree=deg)
bc0 = DirichletBC(V, Constant((0, 0)), fixed_boundary)
bc1 = DirichletBC(V, disps, astr_a, method='pointwise')
bc2 = DirichletBC(V, disps, astr_b, method='pointwise')
bc3 = DirichletBC(V, disps, astr_c, method='pointwise')
bc4 = DirichletBC(V, disps, astr_d, method='pointwise')
bc5 = DirichletBC(V, disps, astr_e, method='pointwise')
bcs = [bc0, bc1, bc2, bc3, bc4, bc5]

# Stress and strain
def epsilon(u):
    return 0.5*(nabla_grad(u) + nabla_grad(u).T)

def sigma(u):
    return Lam*nabla_div(u)*Identity(d) + 2*Mu*epsilon(u)

# Define variational problem
u = TrialFunction(V)
d = u.geometric_dimension()
v = TestFunction(V)
f = Constant((0, 0))
a = inner(sigma(u), epsilon(v))*dx
L = dot(f, v)*dx

# Create VTK file for saving solution
ufile = File('data/%s/u_%d.pvd' % (sim, N))
sfile = File('data/%s/s_%d.pvd' % (sim, N))

# Compute solution
u = Function(V)
t = 0.0
for i in range(nt):
    disps.d = disp[i]
    solve(a == L, u, bcs)
    u.rename('u (um/s)', 'u (um/s)')
    
    # Calculate stress
    sigma_w = project(sigma(u)*Identity(d), WW)
    sigma_r = project(sigma_w[0, 0], W)
    sigma_r.rename('sigma (Pa)', 'sigma (Pa)')
        
    # Save to file and plot solution
    ufile << (u, t)
    sfile << (sigma_r, t)
    
    t += dt
