"""Example of passive Lagrangian particles in a channel."""

import sys
import numpy

## Uncomment this to profile code
#from fluidity_tools import Profile
#try:
#    import builtins
#except ImportError:
#    import __builtin__ as builtins
#builtins.__dict__['profile'] = Profile.make_line_profiler()

import particle_model as pm

## Edit this section to change behaviour

N = 100  # number of particles initially in domain
Re = 1725
DH = 2
particle_insert_number = [50, 80, 120]

for insert_number in particle_insert_number:

    DATA_NAME = 'Raw_data/Re'+str(Re)+'_DH'+str(DH)+'/particles'+str(insert_number)+'ps/Velocity2d.pvd' # base name for .vtus
    MESH_NAME = 'shallow_water2.0.msh'

    INLET_IDS = [1]  # Mesh physical ids to insert particles
    INLET_VELOCITY = (0.46/20, 0, 0) # particle speed at inlet
    OUTLET_IDS = [2] # Mesh physical ids where particles leave

    INSERTION_RATE= insert_number * 1.0 # Particles per time unit inserted
    OUTPUT_TIMESTEP = 1.0 # time in seconds between vtus
    PARTICLE_TIMESTEP = 0.01 # time in seconds for particle time integration

    X = numpy.zeros((N, 3))
    V = numpy.zeros((N, 3)) 

    TEMP_CACHE = pm.TemporalCache.TemporalCache(DATA_NAME, 
                                                velocity_name="Depth averaged velocity", 
                                                pressure_name=None,
                                                timescale_factor=OUTPUT_TIMESTEP,
                                                online=False)

    for k, x in enumerate(X):
        V[k, :] = TEMP_CACHE.get_velocity(x, 0.0)

    print('????')

    MESH = pm.IO.GmshMesh()
    MESH.read(MESH_NAME)
    BOUNDARY_MESH = pm.IO.make_boundary_from_msh(MESH)
    INLET = pm.Options.Inlet(surface_ids=INLET_IDS, insertion_rate=INSERTION_RATE,
                             velocity=lambda X, t,: INLET_VELOCITY, # setup inlet, inlet velocity
                             pdf=None)
    BOUNDARY = pm.IO.BoundaryData(BOUNDARY_MESH, inlets=[INLET])
    SYSTEM = pm.System.System(BOUNDARY, temporal_cache=TEMP_CACHE, outlet_ids=INLET_IDS+OUTLET_IDS, inlets=[INLET]) #outlets allow particles to leave

    PAR = pm.ParticleBase.PhysicalParticle(diameter=0.0)

    FIELDS = {"InsertionTime": numpy.zeros((N, 1))}

    # buckets hold collections of particles
    # drive the system from there

    PB = pm.Particles.ParticleBucket(X, V, time=0.0, delta_t=PARTICLE_TIMESTEP,
                                     system=SYSTEM,
                                     parameters=PAR,
                                     field_data=FIELDS,
                                     online=False)
    #time is start time, delta_t is timestep. Setting online only matters in parallel


    PD = pm.IO.PolyData(DATA_NAME+'_trajectories', 
                        {}) # This holds trajectory information
    # output format is dictionary, key is name, value is length

    PD.append_data(PB) # Store initial particle positions

    for i, cache in enumerate(TEMP_CACHE):

        print('time', cache[0])

        # call which updates the particles
        PB.run(time=cache[0], write=False, method="AdamsBashforth3")

        xx = numpy.array((0.5, 5.0, 0.0)) # Space is always 3d: in 2d z component is zero
        vv = numpy.array((1.0, 0.0, 0.0)) # Same for velocity.
        part = pm.Particles.Particle((xx, vv, PB.time, PB.delta_t),
                                     system=SYSTEM,
                                     parameters=PAR) # Make a new particle
    #    PB.particles.append(part) # Stick it in the bucket
        
        pm.IO.write_level_to_polydata(bucket=PB, level=i, basename=DATA_NAME,
                                      field_data={}) # Dump just this timelevel
        PD.append_data(PB)

        print(len(PB))
        if len(PB):
            print('min, max: pos_x', PB.pos_as_array()[:, 0].min(), PB.pos_as_array()[:, 0].max())
            print('min, max: pos_y', PB.pos_as_array()[:, 1].min(), PB.pos_as_array()[:, 1].max())
            print('min, max: vel_x', PB.vel_as_array()[:, 0].min(), PB.vel_as_array()[:, 0].max())
            print('min, max: vel_y', PB.vel_as_array()[:, 1].min(), PB.vel_as_array()[:, 1].max())

    PD.write() # write trajectories

    with open(outputdir+"/Parameters.txt",'w') as file:
        file.write("Re: "+ str(Re)+"\n"+
                   "D/H: "+ str(DH)+"\n"+
                   "INSERTION_RATE: "+ str(INSERTION_RATE)+"\n"+
                   "Particle Density: "+ str(1.5/INLET_VELOCITY[0]*INSERTION_RATE/(450*180))+"\n")
