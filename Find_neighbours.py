import mdtraj as md
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import time
import sys
import os
import pandas as pd
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
        atom_subset=md_topology.select('all')
    
    # now that we are sure that both topology and selection are valid we can return atom_subset
    # and use the loaded topology file
    
    
    return atom_subset, md_topology

def _neighbouring_atoms(md_topology, trajectory, atom_subset, atom_number, verbose, unpythonize, chunk, cutoff):    
    
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

    # initiate some variables
    neighbour_atoms = []
    sim_time=[]
    number_of_frames = 0
    
    
    # now we need to load each trajectory file as a chunk
    try:

        pbar = tqdm(total=len(trajectory_path), unit= 'File')
        
        for file_i in trajectory_path:
            for chunk_i in md.iterload(trajectory + file_i, chunk, top=md_topology, atom_indices = atom_subset):

                sim_time.append(chunk_i.time)
                number_of_frames += chunk_i.n_frames

                if verbose > 1:
                    print 'Successfully loaded trajectory: \n %s' %(chunk_i)

                neighbour_atoms.append(md.compute_neighbors(chunk_i, cutoff, np.array([atom_number])))
                
            neighbour_atoms_np =np.concatenate(neighbour_atoms)
            
            pbar.update(1)
                    
    except:
        sys.exit('Make sure you provided a valid path to a folder with trajectory files!')
    else:
        print '\nSuccesfully loaded coordinates for %s atoms in %s frames!' %(len(atom_subset), number_of_frames)

    all_neighbour_atoms_np = np.concatenate(neighbour_atoms_np)
    
    sim_time = np.concatenate(sim_time)
    
    return all_neighbour_atoms_np, sim_time

def main(topology, trajectory, atom_number, residue_selection, cutoff = 5.0, unpythonize = True, verbose = 1, selection = True, chunk=100, threshold = 5):
    """
    Find neighbouring atoms
    """
    
    # convert to nanometers (MDtraj unit) 
    cutoff = cutoff/10
    
    if verbose > 0:
    # timer for the whole function
        start = time.clock()
        
    # now load topology and atom_subset of interest
    atom_subset, md_topology = _atom_selection(topology, residue_selection, selection, verbose)
    
    # load trajectories and find neighbours
    neighbour_atoms, sim_time = _neighbouring_atoms(md_topology, trajectory, atom_subset, atom_number, verbose, unpythonize, chunk, cutoff)

    
   
    
    # histogram
    # need to make a small detour because matplotlib
    # histogram has problems counting frequency
    # when using small bins
    
    y = np.bincount(neighbour_atoms)
    dist = np.nonzero(y)[0]
    zip(dist,y[dist])
    neighbour_atoms_freq_count = pd.DataFrame(zip(dist,y[dist]))
    neighbour_atoms_freq_count = neighbour_atoms_freq_count.astype(float)
    neighbour_atoms_freq_count[1] = neighbour_atoms_freq_count[1].values/max(neighbour_atoms_freq_count[1].values)*100
    neighbour_atoms_freq = neighbour_atoms_freq_count[neighbour_atoms_freq_count[1].values > threshold]
    
    if unpythonize:
        # rename residues (still testing)
        final_resid_name = []
        for i in md_topology.subset(np.append(np.array(neighbour_atoms_freq[0].values), atom_number)).atoms:
            residues = str(i)
            # position to substitute
            pos = []
            num = []
            for i in range(len(residues)):    
                # skip first three letters
                if i<6 and residues[i].isdigit():
                    pos.append(i)
                    num.append(int(residues[i]))
            if len(pos) == 1:
                if num[-1] < 9:
                    num[-1] += 1
                else:
                    num[-1] = 10
            if len(pos) == 2:
                if num[-1] < 9:
                    num[-1] += 1
                else:
                    num[-1] = 0
                    num[0] += 1
            if len(pos) == 3:
                if num[-1] < 9:
                    num[-1] += 1
                else:
                    num[1] += 1
                    num[-1] = 0
            z = 0
            residues = list(residues)
            for j in pos:
                residues[j] = str(num[z])
                z+=1
            final_resid_name.append("".join(residues)) 
        
        x = np.array([range(len(final_resid_name) - 1)])

        plt.figure(figsize=(15,15))
        plt.xticks(x.T[:,0], final_resid_name[:-1], rotation= 60 )
        plt.bar(x.T[:,0], neighbour_atoms_freq[1].values)
        plt.xlabel('Residue name', size = 16)
        plt.ylabel('Frequency (%)', size =16)
        plt.title('Atoms within %s ${\AA}$ of %s' %(cutoff*10, final_resid_name[-1]), size = 24)
        plt.show()



    else:
            
        final_resid_name = []
        for i in md_topology.subset(np.append(np.array(neighbour_atoms_freq[0].values), atom_number)).atoms:
            final_resid_name.append(i)
            
        x = np.array([range(len(final_resid_name) -1)])
            
        plt.figure(figsize=(15,15))
        plt.xticks(x.T[:,0], final_resid_name[:-1], rotation= 60 )
        plt.bar(x.T[:,0], neighbour_atoms_freq[1].values)
        plt.xlabel('Residue name', size = 16)
        plt.ylabel('Frequency (%)', size =16)
        plt.title('Atoms within %s ${\AA}$ of %s' %(cutoff*10, final_resid_name[-1]), size = 24)
        plt.show()
            
    if verbose > 0:
        print '\nFound the following atoms within the cutoff:'
        for residues in final_resid_name[:-1]:
            print residues