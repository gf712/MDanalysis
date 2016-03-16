import mdtraj as md
import os
import numpy as np
import sys
import matplotlib.pyplot as plt
%matplotlib inline
import time
import seaborn as sns
from scipy.stats import gaussian_kde

def _dist_atom_selection(topology, atom1, atom2, verbose, unpythonize):
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
    
    ## check chosen atoms
    if isinstance(atom1, str):
        try:
            atom_subset1 = md_topology.select(atom1)
        except:
            sys.exit('Invalid atom selection!')
        else:
            if verbose > 0:
                print 'Atom1 number: %s' %(atom_subset1[0] + 1)
    
    
    elif isinstance(atom1, int):
        atom1 = atom1 - 1
        try:
            atom_subset1 = []
            if atom1 < md_topology.n_atoms:
                atom_subset1.append(atom1)
        except:
            sys.exit('Atom selection invalid for given topology!')
        else:
            if verbose > 0:
                print 'Atom1 number: %s' %(atom_subset1[0] + 1)
                
    else:
        sys.exit('Invalid atom selection, you need to provide a string or an integer')

    if isinstance(atom2, str):
        try:
            atom_subset2 = md_topology.select(atom2)
        except:
            sys.exit('Invalid atom selection!')
        else:
            if verbose > 0:
                print 'Atom2 number: %s' %(atom_subset2[0] + 1)                   
                
    elif isinstance(atom2, int):
        atom2 = atom2 - 1
        try:
            atom_subset2 = []
            if atom2 < md_topology.n_atoms:
                atom_subset2.append(atom2)
        except:
            sys.exit('Invalid atom selection!')
        else:
            if verbose > 0:
                print 'Atom2 number: %s' %(atom_subset2[0] + 1)
                
    else:
        sys.exit('Invalid atom selection, you need to provide a string or an integer')

    pair = np.append(atom_subset1, atom_subset2).reshape(1,2)
    
    # should work until 1109
    if unpythonize:
        # rename residues (still testing)
        final_resid_name = []
        for i in md_topology.subset(pair).atoms:
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
        final_resid = " - ".join(final_resid_name)
    
    if verbose > 0:
        print 'Calculating distance between following atoms: %s \n' %(final_resid)
    
    
    return md_topology, pair, final_resid

def _dist_atom_load(md_topology, trajectory, pair, verbose, chunk):
    
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
    
    all_distances = []
    sim_time = []
    
    try:
        for file_i in trajectory_path:
                for chunk_i in md.iterload(trajectory + file_i, chunk, top=md_topology):
                            
                    sim_time.append(chunk_i.time)

                    if verbose > 1:
                        print 'Successfully loaded trajectory: \n %s' %(chunk_i)

                    all_distances.append(md.compute_distances(chunk_i, pair))
                            
                            
        all_distances_np = np.concatenate(all_distances) * 10
    except:
        sys.exit('Make sure you provided a valid path to a folder with trajectory files!')
    else:
        print '\nSuccesfully calculated atom distances in %s frames!' %(all_distances_np.shape[0])
 
    sim_time = np.concatenate(sim_time) / 1000
    
    return all_distances_np.reshape(all_distances_np.shape[0]), sim_time

def main(topology, trajectory, atom1, atom2, verbose = 1, chunk = 100, unpythonize = True, integrate = None):
    """
    Function to calculate distance between two
    atoms, plot pdf and integrate
        Input:
            topology: 
                str
                path to topology file
            trajectory:
                str
                path to folder with trajectory files
            atom1 and atom2:
                str
                MDTraj atom selection language
                to select atoms for distance calculation
            verbose (optional (Default 1)):
                int (0,1,2)
                level of information shown
            chunk (optional (Default 100)):
                int
                number of frames loaded at a time
            unpythonize (optional (Default True)):
                boolean
                convert residue number from python 0-index
            integrate (optional (Default None))
                list
                list with lists of integer corresponding lower
                and upper bound values for integral calculation
    """
    
    if verbose > 0:
    # timer for the whole function
        start = time.clock()
        
    # now load topology and verify selection by calling _dist_atom_selection()
    md_topology, pair, final_residues = _dist_atom_selection(topology, atom1, atom2, verbose, unpythonize)
    
    # load trajectories and calculate distances
    distance, sim_time  = _dist_atom_load(md_topology, trajectory, pair, verbose, chunk)
    
    kde= gaussian_kde(distance)
        
    x = np.linspace(0,max(distance), distance.shape[0])
    
    plt.figure(figsize=(12,12))
    plt.plot(x, kde(x))
    plt.title('Gaussian Kernel Density Estimation of %s distance' %(final_residues), size = 20)
    plt.xlabel('Distance (${\AA}$)', size = 16)
    plt.ylabel('Density', size =16)
    plt.show()
    
    if integrate != None:
        print ''
        for x in integrate:
            print 'Integral values between %s and %s: %s' %(x[0], x[1], kde.integrate_box(x[0], x[1]))

    
    if verbose > 0:
        total_trajectory_time = sim_time[-1] - sim_time[0]
        print '\nTotal execution time: %.2f seconds \n' %(time.clock() - start)
        print 'Total simulation time of all frames: %.2f nanoseconds' %(total_trajectory_time)
