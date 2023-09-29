import numpy as np

from .. import fortran_src as fsrc


def groupfinder(binmap, periodic=True):
    """Returns group IDs for adjoining regions on a binary map.

    Parameters
    ----------
    binmap : 3darray
        Binary map (ie. 0 or 1) 3D array.
    periodic : bool, optional
        Periodic boundary.

    Returns
    -------
    groupID : 2darray or 3darray
        groupIDs on a 2D/3d array map (same dimensions as binmap). Regions
        with no groups given a 0.
    """
    shape = np.shape(binmap)
    assert len(shape) == 2 or len(shape) == 3, "Only 2D or 3D shapes allowed."
    if np.isscalar(periodic):
        periodx = periodic
        periody = periodic
        if len(shape) == 3:
            periodz = periodic
    else:
        periodx = periodic[0]
        periody = periodic[1]
        if len(shape) == 3:
            periodz = periodic[2]
    nxgrid = shape[0]
    nygrid = shape[1]
    if len(shape) == 3:
        nzgrid = shape[2]
    if len(shape) == 2:
        maxlabel, groupID = fsrc.hoshen_kopelman_2d(binmap=binmap.flatten(),
            binlen=nxgrid*nygrid, nxgrid=nxgrid, nygrid=nygrid, periodx=periodx, periody=periody)
        groupID = groupID.reshape(nxgrid, nygrid)
    else:
        maxlabel, groupID = fsrc.hoshen_kopelman_3d(binmap=binmap.flatten(),
            binlen=nxgrid*nygrid*nzgrid, nxgrid=nxgrid, nygrid=nygrid, nzgrid=nzgrid,
            periodx=periodx, periody=periody, periodz=periodz)
        groupID = groupID.reshape(nxgrid, nygrid, nzgrid)
    return groupID


def mpi_groupfinder(binmap, MPI, periodic=True):
    """Returns group IDs for adjoining regions on a binary map.

    Parameters
    ----------
    binmap : 3darray
        Binary map (ie. 0 or 1) 3D array.
    MPI : object
        MPI object.
    periodic : bool, optional
        Periodic boundary.

    Returns
    -------
    groupID : 2darray or 3darray
        groupIDs on a 2D/3d array map (same dimensions as binmap). Regions
        with no groups given a 0.
    """
    shape = np.shape(binmap)
    assert len(shape) == 2 or len(shape) == 3, "Only 2D or 3D shapes allowed."
    if np.isscalar(periodic):
        periodx = periodic
        periody = periodic
        if len(shape) == 3:
            periodz = periodic
    else:
        periodx = periodic[0]
        periody = periodic[1]
        if len(shape) == 3:
            periodz = periodic[2]
    nxgrid = shape[0]
    nygrid = shape[1]
    if len(shape) == 3:
        nzgrid = shape[2]
    # Send bottom row of binmap down.
    binmap_send_down = MPI.send_down(binmap[0])
    if MPI.rank != MPI.size - 1  or periodx is True:
        binmap = np.concatenate([binmap, np.array([binmap_send_down])])
        nxgrid += 1
    # For the first run we compute the group finder for each individual section
    # making sure to state that the x-axis is not periodic because this is the
    # axis that is split by the MPI.
    if len(shape) == 2:
        maxlabel, groupID = fsrc.hoshen_kopelman_2d(binmap=binmap.flatten(),
            nxgrid=nxgrid, nygrid=nygrid, periodx=False, periody=periody)
        groupID = groupID.reshape(nxgrid, nygrid)
    else:
        maxlabel, groupID = fsrc.hoshen_kopelman_3d(binmap=binmap.flatten(),
            nxgrid=nxgrid, nygrid=nygrid, nzgrid=nzgrid, periodx=False,
            periody=periody, periodz=periodz)
        groupID = groupID.reshape(nxgrid, nygrid, nzgrid)
    # push groupIDs so there are no clashes between nodes.
    maxlabels = MPI.collect(np.array([maxlabel]))
    if MPI.rank == 0:
        summaxlabels = np.cumsum(maxlabels)
        MPI.send(summaxlabels, tag=11)
    else:
        summaxlabels = MPI.recv(0, tag=11)
    MPI.wait()
    cond = np.where(groupID != 0)
    if MPI.rank != 0:
        groupID[cond] += summaxlabels[MPI.rank-1]
    if len(shape) == 2:
        groupID = groupID.reshape(nxgrid, nygrid).astype('int')
    else:
        groupID = groupID.reshape(nxgrid, nygrid, nzgrid).astype('int')
    # work out groupID clashes between nodes
    groupID_send_up = MPI.send_up(groupID[-1])
    labelsout = np.arange(summaxlabels[-1]) + 1
    if MPI.rank != 0 or periodx is True:
        labelsout = fsrc.resolve_clashes(group1=groupID[0].flatten(),
            group2=groupID_send_up.flatten(), lengroup=len(groupID_send_up.flatten()),
            labels=labelsout, lenlabels=len(labelsout))
        labelsout = fsrc.cascade_all(maxlabel=summaxlabels[-1], lenlabels=len(labelsout),
            labels=labelsout)
    # collect labels and work out collective clashes.
    labelsall = MPI.collect(np.array([labelsout]))
    if MPI.rank == 0:
        labelsout = np.copy(labelsall[0])
        for i in range(1, MPI.size):
            labelsout = fsrc.resolve_clashes(group1=labelsout, group2=labelsall[i],
                lengroup=len(labelsall[i]), labels=labelsout, lenlabels=len(labelsout))
            labelsout = fsrc.cascade_all(maxlabel=summaxlabels[-1],
                lenlabels=len(labelsout), labels=labelsout)
        newmaxlabel, labelsout = fsrc.shuffle_down(maxlabel=summaxlabels[-1],
            lenlabels=len(labelsout), labels=labelsout)
        MPI.send(labelsout, tag=11)
    else:
        labelsout = MPI.recv(0, tag=11)
    MPI.wait()
    # relabel all groupIDs.
    if MPI.rank != MPI.size - 1  or periodx is True:
        groupID = groupID[:-1]
        nxgrid -= 1
    groupID = fsrc.relabel(group=groupID.flatten(), lengroup=len(groupID.flatten()),
                          labels=labelsout, lenlabels=len(labelsout))
    if len(shape) == 2:
        groupID = groupID.reshape(nxgrid, nygrid)
    else:
        groupID = groupID.reshape(nxgrid, nygrid, nzgrid)
    return groupID


def get_ngroup(groupID):
    """Get number of members for each group.

    Parameters
    ----------
    group : int array
        Group IDs.

    Returns
    -------
    Ngroup : int array
        Number of members in each group.
    """
    Ngroup = fsrc.get_nlabels(maxlabel=np.max(groupID), lenlabels=len(groupID.flatten()),
        labels=groupID.flatten())
    return Ngroup


def mpi_get_ngroup(groupID, MPI):
    """Get number of members for each group.

    Parameters
    ----------
    group : int array
        Group IDs.
    MPI : object
        MPI object.

    Returns
    -------
    Ngroup : int array
        Number of members in each group.
    """
    maxID = MPI.max(groupID.flatten())
    Ngroup = fsrc.get_nlabels(maxlabel=maxID, lenlabels=len(groupID.flatten()),
        labels=groupID.flatten())
    Ngroup = MPI.sum(Ngroup)
    return Ngroup


def sum4group(groupID, param):
    """Sum parameters for a given group.

    Parameters
    ----------
    group : int array
        Group IDs.
    param : float array
        Parameter values for each point in the grid.

    Returns
    -------
    sumparam : int array
        Sum parameter values for each group.
    """
    sumparam = fsrc.sum4group(group=groupID.flatten(), param=param.flatten(),
        lengroup=len(groupID.flatten()), maxlabel=np.max(groupID))
    return sumparam


def mpi_sum4group(groupID, param, MPI):
    """Sum parameters for a given group.

    Parameters
    ----------
    group : int array
        Group IDs.
    param : float array
        Parameter values for each point in the grid.
    MPI : object
        MPI object.

    Returns
    -------
    sumparam : int array
        Sum parameter values for each group.
    """
    maxID = MPI.max(groupID.flatten())
    sumparam = fsrc.sum4group(group=groupID.flatten(), param=param.flatten(),
        lengroup=len(groupID.flatten()), maxlabel=maxID)
    sumparam = MPI.sum(sumparam)
    return sumparam
