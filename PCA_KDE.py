# let's start by loading all the necessary libraries
import mdtraj as md
import os
import numpy as np
import sys
import matplotlib.pyplot as plt
%matplotlib inline
import time
from sklearn.decomposition import PCA
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns

def _atom_selection(topology, residue_selection, selection, verbose):
    """
    Function takes a topology file and residue selection and verifies if
    the lattter is possible. It also returns the C-alpha atom selection
    Input:
        topology: 
            mdtraj.Topology or string
            Either mdtraj.Topology object or path to trajectory
            to be loaded
        residue_selection:
            string or list with integers
            String will be interpreted with the Mdtraj atom
            selection language. The list will be treated as
            atom number
        selection (by default True):
            boolean
            if true the function will try to return the residue/
            atom selection
    Output:
        atom_subset
            numpy.array 
            array with all the atom numbers corresponding to selection
        md_topology
            mdtraj.core.topology.Topology object of protein 
    """
    ## First have to load all the inputs if they are defined by a path
    if isinstance(topology, str):
        if os.path.exists(topology):
            try:
                md_topology = md.load_topology(topology)
            except:
                sys.exit('Make sure you have provided a valid path to topology file!')
            else:
                if verbose > 0:
                    print 'The following topology file was succesfully loaded: \n %s \n' %(md_topology)

    elif isinstance(topology, md.core.topology.Topology):
        md_topology = topology
        if verbose > 0:
            print 'The following topology file was succesfully loaded: \n %s \n' %(md_topology)
    else:
        sys.exit('Invalid input! Must be a valid path to topology file or mdtraj.Topology object')
    
    ## if selection is True the function will try to obtain the specified atoms/residues
    ## if residue name is specified it will by default look for C-alpha atoms
    if selection:
        if isinstance(residue_selection, list):
            try:
                atom_subset = md_topology.select(residue_selection)
            except:
                sys.exit('Invalid atom selection in list!')
            else:
                if verbose > 1:
                    print 'Your selection includes the following atom(s): \n %s \n' %(atom_subset)
                    print 'Your selection includes the following residues: \n'
                    for residue in md_topology.subset(atom_subset).residues:
                        print residue
        elif isinstance(residue_selection, str):
            try:
                atom_subset = md_topology.select('name CA and ' + residue_selection)
            except:
                sys.exit('Check if your atom selection command is recognized by the Mdtraj atom selection language!')
            else:
                if verbose > 1:
                    print 'Your selection includes the following atom(s): \n %s \n' %(atom_subset)
                    print 'Your selection includes the following residues: \n'
                    for residue in md_topology.subset(atom_subset).residues:
                        print residue
        else:
            sys.exit('Make sure you provided a valid residue selection!')
    else:
        atom_subset=md_topology.select('not name H and all')
    
    # now that we are sure that both topology and selection are valid we can return atom_subset
    # and use the loaded topology file
    
    
    return atom_subset, md_topology

def _load_traj_xyz(md_topology, trajectory, atom_subset, verbose, chunk):
    """
    Returns xyz coordinates of all requested trajectories
    """
    
    # first create a list with all the paths that are needed
    try:
        trajectory_path = os.listdir(trajectory)
    except:
        sys.exit('Make sure you have provided a string for a valid path to a trajectory file!')
    else:
        if verbose > 0:
            print 'Loading trajectories from the following files: '
            for trajectory_i in trajectory_path:
                print trajectory_i
                
    # get first frame for superpositioning
    first_frame = md.load(trajectory + trajectory_path[0], frame=0, top=md_topology, atom_indices=atom_subset)
    
    # initiate some variables
    all_coordinates = []
    number_of_frames = 0
    sim_time = []
    
    # now we need to load each trajectory file as a chunk
    try:
        for file_i in trajectory_path:
            for chunk_i in md.iterload(trajectory + file_i, chunk, top=md_topology, atom_indices = atom_subset):
                        
                sim_time.append(chunk_i.time)
                
                # superpose each chunk to first frame
                chunk_i.superpose(first_frame, 0)

                if verbose > 1:
                    print 'Successfully loaded trajectory: \n %s' %(chunk_i)

                all_coordinates.append(chunk_i.xyz.reshape(chunk_i.n_frames, chunk_i.n_atoms * 3))
                        
                        
        all_coordinates_np = np.concatenate(all_coordinates)
    except:
        sys.exit('Make sure you provided a valid path to a folder with trajectory files!')
    else:
        print '\nSuccesfully loaded coordinates for %s atoms in %s frames!' %(all_coordinates_np.shape[1] / 3, all_coordinates_np.shape[0])
    
    time_frame = all_coordinates_np.shape[0] / len(trajectory_path)
    
    sim_time = np.concatenate(sim_time)
    
    return all_coordinates_np, time_frame, sim_time

def main(topology, trajectory, residue_selection, verbose = 1, selection = True, chunk = 100):
    """
    Function to perform PCA on trajectory coordinates
    and returns a plot of the explained variance
    for each PC1
    """
    
    if verbose > 0:
        # timer for the whole function
        start = time.time()
        
    # now load topology and verify selection by calling _atom_selection()
    atom_subset, md_topology = _atom_selection(topology, residue_selection, selection, verbose)
    
    # now we need to load the trajectory coordinates for each file in folder
    all_coordinates, time_frame, sim_time = _load_traj_xyz(md_topology, trajectory, atom_subset, verbose, chunk)
    
    if verbose > 0:
        print 'Loading time: %.2f seconds \n' %(time.time() - start)
        
        
    if verbose > 0:
        # timer for the PCA
        start_PCA = time.time()
    
    pca1=PCA(n_components=1)

    # calculate PCA and convert data to A
    reduced_cartesian = pca1.fit_transform(all_coordinates * 10)
    
    var = pca1.explained_variance_ratio_[0]
    
    plt.figure(figsize = (12,12)) 
    ax = sns.distplot(reduced_cartesian[:,0])
    ax.set_xlabel('PC1 (${\AA}$)', size =16)
    ax.set_ylabel('Density', size = 16)
    ax.set_title('PC1 distribution - explained variance: %.2f %%' %(var * 100), size =24)

    
    if verbose > 0:
        print 'PCA execution time: %.2f seconds \n' %(time.time() - start_PCA)
        print 'Total execution time: %.2f seconds \n' %(time.time() - start)

# Author: Gil Ferreira Hoben