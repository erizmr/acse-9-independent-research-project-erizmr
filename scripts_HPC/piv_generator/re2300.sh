#PBS -N piv_gene_2300
#PBS -l walltime=12:00:00
#PBS -l select=1:ncpus=2:mpiprocs=1:mem=1gb

# Place this file WITHIN the Fluidity source code directory and then qsub

# module load ese-software
# module load ese-gcc
# module load ese-vtk


#module load anaconda3/personal
module load ese-software
module load ese-gcc
module load ese-vtk

# Move back to the current directory.
cd $PBS_O_WORKDIR
cd ..

#export PYTHONPATH=$HOME/FlowML/ParticleModule:$PYTHONPATH 
python2.7 ./run_particles_2300.py