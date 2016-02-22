def multi_plot(RMSF_complexes, RMSF_complex_names, subunits, max_value):
    """
    Function takes in arrays with RMSF values
    to be plotted together according to subunit
    specifications
        Input:
            RMSF
                list
                list of numpy arrays with RMSF values
            RMSF_complex_names
                list
                list of strings to be used in legend
            subunits
                list
                list of tuples that define boundaries
                of each protein subunit
            max_values
                int
                maximum value of y axis
                (avoid finding maximum of all arrays)
        Output
            maplotlib plot
    """
    # create subplots for each subunit
    f, ax = plt.subplots(figsize=(10,12), nrows= len(subunits))
    f.tight_layout()
    # plot RMSF of each subunit in same graph
    for j in range(len(RMSF_complexes)):
        for i in range(len(subunits)):
                ax[i].plot(range(subunits[i][0], subunits[i][1]), RMSF_complexes[j][subunits[i][0]-1:subunits[i][1]-1], label=RMSF_complex_names[j])
                ax[i].set_title('RMSF of residues %s to %s' %(subunits[i][0], subunits[i][1]), size=18)
                ax[i].set_ylabel('RMSF (${\AA}$)')
                ax[i].set_xlabel('Residue number')
                ax[i].set_ylim([0, max_value])
                # the following line was copied from the pyplot manual
                # and positions the legend box on the right side of the figure
                ax[i].legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., prop={'size':15})
        # adjust the spacing between graphs
        f.subplots_adjust(hspace=0.3)