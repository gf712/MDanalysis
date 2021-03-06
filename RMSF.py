import mdtraj as md
import os
import numpy as np
import sys
import matplotlib.pyplot as plt
%matplotlib inline
import time

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
        atom_subset=md_topology.select('name CA and ' + 'all')
    
    # now that we are sure that both topology and selection are valid we can return atom_subset
    # and use the loaded topology file
    
    
    return atom_subset, md_topology

def _trajectory_mean(trajectories_path, chunk, top, atom_subset, first_frame, verbose):
    """
    Function takes in a trajectory and calculates the
    mean position of each atom after superimposing to reference
    Input:
        trajectories_path:
            str
            path of trajectories file of interest
        chunk:
            int
            number of frames to be loaded at a time.
            Note that this value can be defined in the main
            function.
        top:
            mdtraj.core.topology.Topology
        atom_subset:
            numpy.array
            array with all the atom numbers corresponding to selection
        first_frame:
            mdtraj.core.trajectory.Trajectory
            trajectory of first frame
        verbose:
            boolean
            if True shows more information
            Note that this value can be defined in the main
            function.
    """
    # initiating some variables...
    
    traj_sum_list=[]
    number_of_frames = 0
    
    # just a quick check to verify if path exists
    try:
        os.path.exists(trajectories_path)
    except:
        sys.exit('Make sure you have provided a string for a valid path to a trajectory file!')
    else:
        if verbose > 0:
            print 'Loading trajectories...'
    
    try:
        # now let's calculate the native conformation which
        # is just the mean position of each atom in the 
        # whole trajectory file
        for chunk_i in md.iterload(trajectories_path, chunk, top=top, atom_indices = atom_subset):
            
            # just adding the time length of chunk
            # to calculate the total simulation time
            # (not needed in calculation, just for display)
               
            
            if verbose > 1:
                print 'Successfully loaded trajectory: \n %s' %(chunk_i)

            # will use lists in this part because we don't know size
            # of matrices at this point, room for future optimization
            # first we get the sum of all atom coordinates
            # this will be achieved by column wise summation of a coordinate
            # matrix called by xyz trajectory attribute
            
            all_atom_sum =[]
            for atom in range(len(atom_subset)):
                all_atom_sum.append(chunk_i.xyz[:,atom,:].sum(axis=0))

            traj_sum_list.append(all_atom_sum)

            number_of_frames += chunk_i.xyz.shape[0]
            
    except:
        sys.exit('Error while loading trajectories! Make sure you provided a valid trajectory file!')
    
    else:
        print '\nSuccessfully loaded trajectory file!'
        if verbose > 0:
            print '\nTotal number of frames loaded: %s \n' %(number_of_frames)
    
    traj_sum_all = np.concatenate(traj_sum_list)
        
    # then we need to sum all the coordinates of all chunks together
    # we want the result to be a matrix of shape = (len(atom_subset), 3)
    all_atoms_sum_list=[]
    for atom in range(len(atom_subset)):
        all_atom_sum = traj_sum_all[atom::len(atom_subset), :].sum(axis=0)
        all_atoms_sum_list.append(all_atom_sum)
    
    # we just have to put all together
    reference_conformation_array = np.concatenate(all_atoms_sum_list)
    reference_conformation = np.reshape(reference_conformation_array, (len(atom_subset), 3))
    
    # and now we can calculate the average outside of the loop
    reference_conformation = (reference_conformation / number_of_frames) * 10
    
    # the function returns the numpy array with all coordinates
    # and the trajectory time contains the simulation time length
    return reference_conformation

def _fluctuation_matrix(reference_frame, trajectories_path, atom_subset, topology, chunk, first_frame):
    """
    This function computes the residual sum of squares of
    the reference frame and all the corresponding atoms
    in the provided frames
    
    Input:
        reference_frame: 
            numpy.array 
            array with the coordinates of reference frame/ 
            average conformation/ native conformation
        trajectories_path:
            str
            path of trajectories file of interest
        atom_subset:
            numpy.array
            array with all the atom numbers corresponding to selection
        topology:
            mdtraj.core.topology.Topology
        chunk:
            int
            number of frames to be loaded at a time.
            Note that this value can be defined in the main
            function.
        number_frames:
            int
            total number of frames of trajectories
        first_frame:
            mdtraj.core.trajectory.Trajectory
            trajectory of first frame        
    """
    residual_sum_squares = np.zeros((len(atom_subset)))
    
    ## now can compute the difference between the trajectory and its reference
    ## ri(t) - riref Using the mdtraj trajectory attribute xyz to extract
    ## the cartesian coordinates of trajectory and reference in a numpy array
    ## chunk.xyz.shape = (frames, atom, coordinate dimensions)
    
    
    number_of_frames=0                                
    trajectory_time=[]
    for chunk_i in md.iterload(trajectories_path, chunk = chunk, top=topology, atom_indices = atom_subset):
        trajectory_time.append(chunk_i.time)
        for atom in range(len(atom_subset)):
            diff = np.subtract(chunk_i.xyz[:, atom, :] * 10, reference_frame[atom])
            residual_sum_squares[atom] = residual_sum_squares[atom] + ((diff ** 2).sum(axis = 1).sum(axis=0))
        number_of_frames += chunk_i.xyz.shape[0]
    ## the result is a matrix with all fluctuations squared
    ## shape(number of frames * atom numbers, 3)
    ## from 0 to number of frames we have information of first atom
    ## then from number of frames to number of frames * 2 second atoms
    ## and so forth
    
    return residual_sum_squares, number_of_frames, trajectory_time

def rmsf_main(topology, trajectories_path, residue_selection, subunits = [], native_conformation=True, reference_frame = 0, selection=True, plot_result = True, verbose=1, chunk=100):
    """
    Compute RMSF of large trajectory files
    Input:
        topology: 
            mdtraj.Topology or string
            Either mdtraj.Topology object or path to trajectory
            to be loaded
        trajectories_path:
            trajectory file path 
            trajectories to be used for rmsf calculation
        residue_selection:
            string or list of integers
            String will be interpreted with the Mdtraj atom
            selection language. The list will be treated as
            atom number to be used for rmsf calculation
        subunits (optional):
            list
            list of tuples corresponding to start and end
            of each subunit. Needed to create subplot of 
            RMSF vs subunit
        native_conformation (optional):
            boolean (default True)
            if true it will take the mean of all coordinates 
            to calculate the 'native conformation'
            if false the reference_frame_assigned will be used
            to assign reference frame
        selection (optional, by default True)
            boolean
            if True only selection of residues will be used for RMSF,
            else all residues of mdtraj.Topology object will be used
        reference_frame (optional, default is 0):
            integer
            frame number to be used if not using native conformation method
            by default uses first frame and only is taking into account if
            native_conformation = False
        selection (optional, by default True):
            boolean
            if true the function will try to return the residue/
            atom selection 
        plot_result (optional, by default True):
            boolean
            if true the function will also return a plot of RMSF
            as a function of residue
        verbose (optional):
            interger from 0-2 (by default 1)
            level of information displayed
        chunk (optional):
            integer
            Number of frames from Trajectory file to be loaded at a time
            100 is recommended
    Output: 
        RMSF values 
            numpy.array
                each row corresponds to residue RMSF and the matrix is ordered
                by first to last residue
        RMSF plot
            matplotlib plot
                if plot_result is true a plot of the result will also be shown
        
    """
    if verbose > 0:
        # timer for the whole function
        start = time.clock()

    ## Let's start by loading the topology file
    ## Atom_subset contains information about the atom numbers
    ## we are interested in
    atom_subset, md_topology = _atom_selection(topology, residue_selection, selection, verbose)
    

    # also will use the first frame to superpose all structures
    first_frame = md.load(trajectories_path, frame=0, top=md_topology, atom_indices=atom_subset)
    
    # first have to assign the reference structure:
    # if native_conformation is True we need to calculate the the native conformation
    
    if native_conformation:
        
        try:
            # timer for reference conformation calculation and assignment
            load_start = time.clock()
            # calculate the reference conformation which is defined as the mean position of the carbon alpha atoms
            reference_conformation = _trajectory_mean(trajectories_path, chunk, md_topology, atom_subset, first_frame, verbose)
            load_end = time.clock()

        except:
            sys.exit('Did not manage to create native conformation, try using first frame as reference structure!')
        else:
            print 'Successfully created native conformation and assigned it as reference frame in %.2f seconds! \n' %(load_end - load_start)
            if verbose > 1:
                print 'C-alpha native conformation coordinates (in Å): \n%s' %(reference_conformation)
                
    # else we will just use a given frame as a reference
    # by default the first one will picked
    else:
        if reference_frame == 0:
            try:
                reference_conformation = first_frame.xyz * 10
                reference_conformation = np.reshape(reference_conformation, (reference_conformation.shape[1], 3))    
            except:
                print sys.exit('Error occured while loading first frame!')
            else:
                print 'Successfully assigned first frame as reference conformation!'
        else:
            try:
                reference_traj = md.load(trajectories_path, frame=reference_frame, top=md_topology, atom_indices=atom_subset)
                reference_conformation = reference_traj.xyz * 10
                reference_conformation = np.reshape(reference_conformation, (reference_conformation.shape[1], 3))
            except:
                print sys.exit('Error occured while assigning reference conformation!')
            else:
                print 'Successfully assigned frame %s as reference conformation!' %(reference_frame)
              
    # now we just need to subtract the native conformation to the corresponding atoms
    rmsf_matrix, number_of_frames, trajectory_time = _fluctuation_matrix(reference_conformation, trajectories_path, atom_subset, md_topology, chunk, first_frame)
    # and finally calculate the average fluctuation of each atom from
    # rmsf matrix and convert it from nm to angstrom
    # to save computing power the division and sqrt will be computed 
    # for the whole matrix outside the loop

    rmsf = (rmsf_matrix / number_of_frames) ** 0.5   
    
    if plot_result:
        max_value = max(rmsf) + 2
        # subplotting each subunit
        if len(subunits) > 0:
            f, ax = plt.subplots(figsize=(10,10), nrows= len(subunits))
            f.tight_layout()
            for i in range(len(subunits)):
                ax[i].plot(range(subunits[i][0], subunits[i][1]), rmsf[subunits[i][0]-1:subunits[i][1]-1])
                ax[i].set_title('RMSF of residues %s to %s' %(subunits[i][0], subunits[i][1]), size=15)
                ax[i].set_ylabel('RMSF (${\AA}$)')
                ax[i].set_xlabel('Residue number')
                ax[i].set_ylim([0, max_value])
            f.subplots_adjust(hspace=0.3)
        
        else:
            plt.figure(figsize=(10,10))
            plt.plot(range(1, len(atom_subset) + 1), rmsf)
            plt.title('Root mean square fluctuation', size=24)
            plt.xlabel('Residue number', size=16)
            plt.ylabel('RMSF (${\AA}$)', size =16)
            plt.ylim([0, max_value])
    
    if verbose > 0:
        total_trajectory_time = (trajectory_time[-1][-1] - trajectory_time[0][0])
        stop = time.clock()
        print '\nThe RMSF calculation took %.2f seconds.\nNumber of residues: %s \nNumber of frames: %s' %(stop - start, len(atom_subset), number_of_frames)
        print 'RMSF calculation speed: %.2f microseconds per residue per frame' %((stop - start)/len(atom_subset)/number_of_frames * 1e6)
        print 'Total simulation time of all frames: %s nanoseconds' %(total_trajectory_time / 1000)
        print 'Simulation time per frame: %s picoseconds' %(trajectory_time[0][1] - trajectory_time[0][0])
        
    return rmsf


# Author: Gil Ferreira Hoben