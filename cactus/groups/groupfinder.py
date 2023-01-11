def order_edge_indices(edge_ind):
    """Order edges so first index is always smaller, and order by first index.

    Parameters
    ----------
    edge_ind : 2darray
        Edge indexes.

    Returns
    -------
    sort_edge_ind : 2darray
        Return sorted edge indexes.
    """
    e1 = np.copy(edge_ind[:,0])
    e2 = np.copy(edge_ind[:,1])
    cond = np.where(e1 > e2)[0]
    edge_ind[cond,0] = e2[cond]
    edge_ind[cond,1] = e1[cond]
    del(e1,e2)
    idsort = np.argsort(edge_ind[:,0])
    sort_edge_ind = np.array([edge_ind[i] for i in idsort])
    return sort_edge_ind


def findgroups3D(binmap):
    """Returns group IDs for adjoining regions on a binary map.

    Parameters
    ----------
    binmap : 3darray
        Binary map (ie. 0 or 1) 3D array.

    Returns
    -------
    groupID : 3darray
        groupIDs on a 3D array map (same dimensions as binmap). Regions
        with no groups given a -1.
    """
    shape = np.shape(binmap)
    assert len(shape) == 3, "Binmap must be a 3D array."
    xgrid, ygrid, zgrid = shape[0], shape[1], shape[2]
    # Flatten arrays
    binmap = binmap.flatten()
    # Grouping strategy: go through pixel by pixel and assign a groupID if binmap at pixel = 1,
    # check three neighbours +1 in each axis, assign same groupID if they are unassigned and also
    # have a binmap = 1. We note clashes, where a neighbour already has a groupID and will reassign
    # clashing groupIDs later.
    # Create groupID array all set to unassigned (ie. = -1)
    groupID = -np.ones(len(binmap))
    # Create list to note down clashes in ID assignments.
    clashID = []
    # Start assigning groupID
    ngroup = -1
    mask = np.where(binmap == 1.)[0]
    for ii in range(0, len(mask)):
        i = mask[ii]
        # Unassigned but a group, given a new groupID
        if binmap[i] == 1. and groupID[i] == -1:
            ngroup += 1
            groupID[i] = ngroup
        # check neighbours and reassign or note clashes
        if binmap[i] == 1.:
            whichgroup = groupID[i]
            xpixID, ypixID, zpixID = magpie.src.pix_id_3dto1d_scalar(pix_id=i, ygrid=ygrid, zgrid=zgrid)
            xpixID_plus = xpixID+1
            if xpixID_plus == xgrid:
                xpixID_plus = 0
            ypixID_plus = ypixID+1
            if ypixID_plus == ygrid:
                ypixID_plus = 0
            zpixID_plus = zpixID+1
            if zpixID_plus == zgrid:
                zpixID_plus = 0
            pixID_plus_x = magpie.src.pix_id_1dto3d_scalar(xpix_id=xpixID_plus, ypix_id=ypixID, zpix_id=zpixID, ygrid=ygrid, zgrid=zgrid)
            pixID_plus_y = magpie.src.pix_id_1dto3d_scalar(xpix_id=xpixID, ypix_id=ypixID_plus, zpix_id=zpixID, ygrid=ygrid, zgrid=zgrid)
            pixID_plus_z = magpie.src.pix_id_1dto3d_scalar(xpix_id=xpixID, ypix_id=ypixID, zpix_id=zpixID_plus, ygrid=ygrid, zgrid=zgrid)
            if binmap[pixID_plus_x] == 1.:
                if groupID[pixID_plus_x] == -1:
                    groupID[pixID_plus_x] = whichgroup
                elif whichgroup != groupID[pixID_plus_x]:
                    clashID.append(sorted([whichgroup, groupID[pixID_plus_x]]))
            if binmap[pixID_plus_y] == 1.:
                if groupID[pixID_plus_y] == -1:
                    groupID[pixID_plus_y] = whichgroup
                elif whichgroup != groupID[pixID_plus_y]:
                    clashID.append(sorted([whichgroup, groupID[pixID_plus_y]]))
            if binmap[pixID_plus_z] == 1.:
                if groupID[pixID_plus_z] == -1:
                    groupID[pixID_plus_z] = whichgroup
                elif whichgroup != groupID[pixID_plus_z]:
                    clashID.append(sorted([whichgroup, groupID[pixID_plus_z]]))
        knpy.utils.progress_bar(ii, len(mask), explanation='Scanning group finder')
    ngroup += 1
    # Deal with clashes, first ordering and keeping unique clashes only.
    clashID = np.array(clashID).astype('int')
    clashID = order_edge_indices(clashID)
    clashID_pi = mistv2.index.cantor_pair(clashID[:,0], clashID[:,1])
    clashID_pi = np.unique(clashID_pi)
    clashID_k1, clashID_k2 = mistv2.index.uncantor_pair(clashID_pi)
    clashID = np.array([clashID_k1, clashID_k2]).T
    clashID = order_edge_indices(clashID)
    # Figure out which groupIDs need to be switched
    switchgroup = []
    for i in range(ngroup):
        cond1 = np.where(clashID[:,0] == i)[0]
        cond2 = np.where(clashID[:,1] == i)[0]
        switchgroup.append(sorted(clashID[cond1,1].tolist() + clashID[cond2,0].tolist()))
        #knpy.utils.progress_bar(i, ngroup, explanation='Processing GroupID clashes')
    for i in range(0, len(switchgroup)):
        j = 0
        while j < len(switchgroup[i]):
            if len(switchgroup[switchgroup[i][j]]) != 0:
                if switchgroup[i][j] != i:
                    switchgroup[i] = switchgroup[i] + switchgroup[switchgroup[i][j]]
                    switchgroup[switchgroup[i][j]] = []
            j += 1
        knpy.utils.progress_bar(i, len(switchgroup), explanation='Collecting GroupID clashes')
    for i in range(0, len(switchgroup)):
        if len(switchgroup[i]) != 0:
            switchgroup[i] = np.unique(sorted(switchgroup[i]))[1:].tolist()
    switchID = np.arange(len(switchgroup))
    for i in range(0, len(switchgroup)):
        if len(switchgroup[i]) != 0:
            switchID[switchgroup[i]] = i
    uninewID = np.unique(switchID)
    newID = np.arange(len(switchID))
    for i in range(0,len(uninewID)):
        cond = np.where(switchID == uninewID[i])[0]
        newID[cond] = i
    groupID = np.copy(groupID.astype('int'))
    for i in range(0, len(groupID)):
        if groupID[i] != -1:
            groupID[i] = newID[groupID[i]]
        knpy.utils.progress_bar(i, len(groupID), explanation='Correcting groupID clashes')
    groupID = groupID.reshape(shape)
    return groupID
