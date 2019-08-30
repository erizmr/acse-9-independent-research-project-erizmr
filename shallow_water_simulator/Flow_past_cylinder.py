import thetis as th
from thetis import *
import pylab as plt
import time
import datetime
import numpy as np
import firedrake as fire
import math
import os


def update_forcings_hydrodynamics(t_new):
        old_bathymetry_2d.assign(bathymetry_2d)
        uv1, elev1 = solver_obj.fields.solution_2d.split()
        depth.interpolate(elev1 + old_bathymetry_2d)


def export_final_state(inputdir, uv, elev,):
        th.print_output("Exporting fields for subsequent simulation")

        chk = th.DumbCheckpoint(inputdir + "/velocity", mode=th.FILE_CREATE)
        chk.store(uv, name="velocity")
        th.File(inputdir + '/velocityout.pvd').write(uv)
        chk.close()
        chk = th.DumbCheckpoint(inputdir + "/elevation", mode=th.FILE_CREATE)
        chk.store(elev, name="elevation")
        th.File(inputdir + '/elevationout.pvd').write(elev)
        chk.close()




#-------------------------
# Set time steps
#-------------------------
dt_2 = 0.01
# export interval in seconds
t_export = 1
# final time of simulation
t_end = 150



#--------------------------------
# Define mesh and function space
#--------------------------------

mesh2d = th.Mesh("shallow_water4.0.msh")

P1_2d = th.FunctionSpace(mesh2d, 'DG', 1)
vectorP1_2d = th.VectorFunctionSpace(mesh2d, 'DG', 1)
V = th.FunctionSpace(mesh2d, 'CG', 1)
bathymetry_2d = th.Function(V, name='Bathymetry')

# Show the mesh
fig = plt.figure(figsize=(12, 4))
ax1 = fig.add_subplot(111)
fire.plot(mesh2d, axes=ax1)


#-----------------------------------
# Define field
#-----------------------------------

x,y = th.SpatialCoordinate(mesh2d)

#--------------------------------
# Define the physical parameters
#--------------------------------
lx = 1.5
ly = 0.6
D = 0.1
initialdepth_value = 0.04
initialdepth = th.Constant(initialdepth_value)
depth_riv = th.Constant(initialdepth - 0.0)
bathymetry_2d.interpolate(depth_riv)

viscosity_value = 10**(-6)
viscosity = th.Constant(viscosity_value)


# ---------------------------------
# Define the profile for velocities and depth
# ---------------------------------
vx_val = 0.0023*10
vy_val = 0.0
vx_value = th.Constant(vx_val)
vy_value = th.Constant(vy_val)
elev_init = initialdepth


horizontal_velocity = th.Function(P1_2d).interpolate(vx_value)
vertical_velocity = th.Function(P1_2d).interpolate(vy_value)

# Initialize the velocity field
uv_init = th.Function(vectorP1_2d).interpolate(th.as_vector((horizontal_velocity, vertical_velocity)))
# Initialize the depth field
depth = th.Function(V).interpolate(elev_init + bathymetry_2d)



qfc = th.Constant(0.0025) # Start from a constant drag coefficients
qfc_value = 0.0025
old_bathymetry_2d = th.Function(V).interpolate(bathymetry_2d)

#------------------------------------
# Define the output file name
#------------------------------------

# choose directory to output results
ts = time.time()
st = datetime.datetime.fromtimestamp(ts).strftime('%Y_%m_%d_%H_%M_%S')
outputdir = 'outputs'+ st

H = depth
Re = vx_val * D / viscosity_value
DH = D / initialdepth_value
S = qfc_value * DH
outputdir = './Re%.0f_DH%.0f_' %(Re, DH) + outputdir
th.print_output('Exporting to '+outputdir)
print("Re:", Re)
print("D/H:", DH)
print("Stability number:", S)
print("velocity: ", vx_val)
print("viscosity: ", viscosity_value)



#----------------------------------
# Solve
#----------------------------------
# set up solver 
solver_obj = th.solver2d.FlowSolver2d(mesh2d, bathymetry_2d)
options = solver_obj.options
options.simulation_export_time = t_export
options.simulation_end_time = t_end
options.output_directory = outputdir

options.check_volume_conservation_2d = True
options.fields_to_export = ['uv_2d', 'elev_2d']
options.solve_tracer = False
options.use_lax_friedrichs_tracer = False
options.quadratic_drag_coefficient = qfc
# set horizontal diffusivity parameter
options.horizontal_diffusivity = th.Constant(0.2)
options.horizontal_viscosity = viscosity

# crank-nicholson used to integrate in time system of ODEs resulting from application of galerkin FEM
options.timestepper_type = 'CrankNicolson'
options.timestepper_options.implicitness_theta = 1.0

if not hasattr(options.timestepper_options, 'use_automatic_timestep'):
        options.timestep = dt_2

    
# Define the boundary conditions
left_bnd_id = 1
right_bnd_id = 2
flux_constant = th.Constant(initialdepth * ly * vx_value)

swe_bnd = {}

uv_vector = th.as_vector((vx_value, vy_value))

swe_bnd[left_bnd_id] = {'flux': flux_constant}
swe_bnd[left_bnd_id] = {'uv': uv_vector}
swe_bnd[right_bnd_id] = {'elev': initialdepth,'flux':flux_constant}


# Non-slip boundary condition for cylinder
swe_bnd[4] = {'un': th.Constant(0.0)}
swe_bnd[4] = {'uv': th.as_vector((0.0, 0.0))}


solver_obj.bnd_functions['shallow_water'] = swe_bnd
solver_obj.assign_initial_conditions(uv=uv_init, elev = elev_init)



solver_obj.iterate(update_forcings = update_forcings_hydrodynamics)
uv, elev = solver_obj.fields.solution_2d.split()


# Save some key parameters
with open(outputdir+"/Parameters.txt",'w') as file:
    file.write("Re: "+ str(Re)+"\n"+
               "D/H: "+ str(DH)+"\n"+
               "Stability number: "+ str(S)+"\n"+
              "velocity: "+ str(vx_val)+"\n"+
              "viscosity: "+ str(viscosity_value)+"\n")

export_final_state("hydrodynamics_mono_stokes_fine", uv, elev)









