import mdtraj as md
import os
import numpy as np
import sys
import matplotlib.pyplot as plt
%matplotlib inline
import time
import seaborn as sns
from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA
from scipy.stats import entropy
from tqdm import tqdm

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
                atom_subset = md_topology.select(residue_selection)
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
        atom_subset=md_topology.select('backbone and all')
    
    # now that we are sure that both topology and selection are valid we can return atom_subset
    # and use the loaded topology file
    
    return atom_subset, md_topology

def _load_traj_xyz(md_topology, trajectory, atom_subset, verbose, chunk, stride):
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
            
            for chunk_i in md.iterload(trajectory + file_i, chunk, top=md_topology, atom_indices = atom_subset, stride = stride):
                        
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
        print '\nSuccesfully loaded coordinates for %s atoms from %s out of %s frames!' %(all_coordinates_np.shape[1] / 3, all_coordinates_np.shape[0], all_coordinates_np.shape[0] * stride)
    
    
    sim_time = np.concatenate(sim_time)
    
    return all_coordinates_np, sim_time

def main(topology, trajectory1, trajectory2, residue_selection, verbose = 1, selection = True, chunk = 100, stride = 1, plot_KLD = True):
    """
    Function to perform PCA on 2 runs and 
    return KLD of distribution over time
    """
    
    if verbose > 0:
        # timer for the whole function and run 1
        start_1 = time.time()
        
    # now load topology and verify selection by calling _atom_selection()
    atom_subset, md_topology = _atom_selection(topology, residue_selection, selection, verbose)
    
    # RUN 1
    
    # now we need to load the trajectory coordinates for each file in folder
    all_coordinates_run1, sim_time_run1 = _load_traj_xyz(md_topology, trajectory1, atom_subset, verbose, chunk, stride)
    
    if verbose > 0:
        print 'Loading time: %.2f seconds \n' %(time.time() - start_1)
        
        
    if verbose > 0:
        # timer for the PCA
        start_PCA_run1 = time.time()
    
    pca1=PCA(n_components=1)
    
    
    # calculate PCA and convert data to A
    
    pca_run1_reduced_cartesian = pca1.fit_transform(all_coordinates_run1 * 10)
    
    # gaussian kde of PCA 
    
    kde_run1 = gaussian_kde(pca_run1_reduced_cartesian[:,0])
    
    if verbose:
        print 'Run 1 PCA and KDE calculation time: %.2f seconds \n' %(time.time() - start_PCA_run1)

    # RUN 2
    
    if verbose > 0:
        # timer for run 2
        start_2 = time.time()
    
    # now we need to load the trajectory coordinates for each file in folder
    all_coordinates_run2, sim_time_run2 = _load_traj_xyz(md_topology, trajectory2, atom_subset, verbose, chunk, stride)
    
    if verbose > 0:
        print 'Loading time: %.2f seconds \n' %(time.time() - start_2)
        
        
    if verbose > 0:
        # timer for the PCA
        start_PCA_run2 = time.time()
    
    pca1=PCA(n_components=1)
    
    
    # calculate PCA and convert data to A
    
    pca_run2_reduced_cartesian = pca1.fit_transform(all_coordinates_run2 * 10)
    
    # gaussian kde of PCA for plot
    
    kde_run2 = gaussian_kde(pca_run2_reduced_cartesian[:,0])
    
    if verbose:
        print 'Run 2 PCA and KDE calculation time: %.2f seconds \n' %(time.time() - start_PCA_run2)
        
    print 'Starting KLD calculation \n'
    
    if verbose:
        KLD_start = time.time()
    
    KLD_PCA = []
    
    x = range(min(np.append(pca_run1_reduced_cartesian, pca_run1_reduced_cartesian)),
                   max(np.append(pca_run1_reduced_cartesian, pca_run1_reduced_cartesian)))
    
    total = pca_run2_reduced_cartesian.shape[0] * stride
    pbar = tqdm(total=total, unit= 'Frame')
    
    for frame in range(1, pca_run2_reduced_cartesian.shape[0] + 1):

        time.sleep(0.01)

        #if verbose and frame * stride % 1000 == 0:

        #   print 'Calculating KLD for frame %s/%s (KLD calculation time %.2f seconds)' %(frame * stride, pca_run2_reduced_cartesian.shape[0] * stride, time.time() - KLD_start)

        kde_n = gaussian_kde(pca_run2_reduced_cartesian[:frame + 1,0])

        KLD_PCA.append(entropy(kde_n(x), kde_run1(x), base = 10.0))
            
        pbar.update(stride)

    if verbose:
        print '\nKLD calculation time: %.2f seconds \n' %(time.time() - KLD_start)
    
    
    if plot_KLD:
        
        plt.figure(figsize=(12,12))
        plt.plot(range(1, len(KLD_PCA * stride) + 1, stride), KLD_PCA)
        plt.xlabel('Frame number', size = 16)
        plt.ylabel('KLD', size = 16)
        plt.title('Kullback Leibler Divergence of PCA over time', size = 22)
    
    if verbose > 0:
        
        total_trajectory_time_1 = (sim_time_run1[-1] - sim_time_run1[0])
        total_trajectory_time_2 = (sim_time_run2[-1] - sim_time_run2[0])
        print 'Total execution time: %.2f seconds \n' %(time.time() - start_1)
        print 'Run 1 simulation time: %s ns \n' %(total_trajectory_time_1 / 1000)
        print 'Run 2 simulation time: %s ns \n' %(total_trajectory_time_2 / 1000)
        
        
# Author: Gil Ferreira Hoben