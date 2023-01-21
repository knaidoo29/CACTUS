import sys
import os.path
import numpy as np

import cactus
import shift
#import fiesta
import mpiutils

import matplotlib.pylab as plt
from cactus import src

MPI = mpiutils.MPI()

ngrid = 100
if MPI.rank == 0:
    path = '/Users/krishna/Science/Projects/2022/cosmic_web_classification/analysis1/'
    fname = path + 'data/COCO/node_maxResponse.MMF'
    data = np.fromfile(fname, dtype='single')[264:]
    data = data.reshape(640,640,640)
    data = data[:ngrid,:ngrid,:ngrid][0]
    split1, split2 = MPI.split(len(data))
    for i in range(1, len(split1)):
        MPI.send(data[split1[i]:split2[i]], to_rank=i, tag=10+i)
    data = data[split1[0]:split2[0]]
else:
    data = MPI.recv(0, tag=10+MPI.rank)
MPI.wait()

binmap = np.zeros(np.shape(data))
cond = np.where(data > 0.)
binmap[cond] = 1.
binmap = binmap.astype('int')


#-------------------------------------------------

# val = MPI.send_up(MPI.rank)
# MPI.mpi_print(MPI.rank, val)
#
# # val = MPI.send_down(MPI.rank)
# # MPI.mpi_print(MPI.rank, val)

groupID = cactus.groups.mpi_groupfinder(binmap, MPI, periodic=False)

MPI.mpi_print(MPI.rank, np.max(groupID))
# groupIDs = MPI.collect(np.array([groupID]), outlist=True)
# if MPI.rank == 0:
#     groupIDs = np.hstack(groupIDs)
#     MPI.mpi_print(np.shape(groupIDs))
#     plt.figure()
#     plt.imshow(np.log10(groupIDs.T), origin='lower')
#     plt.show()
#
# #
# # periodic = True
# #
# # shape = np.shape(binmap)
# # assert len(shape) == 2 or len(shape) == 3, "Only 2D or 3D shapes allowed."
# #
# # if np.isscalar(periodic):
# #     periodx = periodic
# #     periody = periodic
# #     if len(shape) == 3:
# #         periodz = periodic
# # else:
# #     periodx = periodic[0]
# #     periody = periodic[1]
# #     if len(shape) == 3:
# #         periodz = periodic[2]
# #
# # nxgrid = shape[0]
# # nygrid = shape[1]
# # if len(shape) == 3:
# #     nzgrid = shape[2]
#
# #
# # # plt.imshow(binmap.T, origin='lower', extent=[25*MPI.rank, 25*(MPI.rank+1), 0, 100])
# # # plt.show()
# #
# # binmap_up = MPI.send_down(binmap[0])
# #
# # if len(shape) == 2:
# #     binmap = np.concatenate([binmap, np.array([binmap_up])])
# # else:
# #     binmap = np.concatenate([binmap, np.array([binmap_up])])
# #
# # # For the first run we compute the group finder for each individual section
# # # making sure to state that the x-axis is not periodic because this is the
# # # axis that is split by the MPI.
# # if len(shape) == 2:
# #     maxlabel, groupID = src.hoshen_kopelman_2d(binmap=binmap.flatten(),
# #                                                nxgrid=nxgrid+1, nygrid=nygrid,
# #                                                periodx=False, periody=periody)
# # else:
# #     maxlabel, groupID = src.hoshen_kopelman_3d(binmap=binmap.flatten(),
# #                                                nxgrid=nxgrid+1, nygrid=nygrid, nzgrid=nzgrid,
# #                                                periodx=False, periody=periody, periodz=periodz)
# #
# # # push groupIDs so there are no clashes between nodes.
# # maxlabels = MPI.collect(np.array([maxlabel]))
# # if MPI.rank == 0:
# #     summax = np.cumsum(maxlabels)
# #     MPI.send(summax, tag=11)
# # else:
# #     summax = MPI.recv(0, tag=11)
# # MPI.wait()
# #
# # cond = np.where(groupID != 0)[0]
# # if MPI.rank != 0:
# #     groupID[cond] += summax[MPI.rank-1]
# #
# # if len(shape) == 2:
# #     groupID = groupID.reshape(nxgrid+1, nygrid)
# # else:
# #     groupID = groupID.reshape(nxgrid+1, nygrid, nzgrid)
# #
# # # work out groupID clashes between nodes
# # groupID_down = MPI.send_up(groupID[-1])
# # groupID = groupID[:-1]
# #
# # labels = np.arange(summax[-1]) + 1
# # # if MPI.rank == MPI.size - 1 and periodx is False:
# # #     labelsout = labels
# # # else:
# # labelsout = src.resolve_clashes(group1=groupID[0].flatten(), group2=groupID_down.flatten(),
# #                                 lengroup=len(groupID_down.flatten()), labels=labels,
# #                                 lenlabels=len(labels))
# #
# # labelsout = src.cascade_all(maxlabel=len(labelsout), lenlabels=len(labelsout), labels=labelsout)
# # groupID = src.relabel(group=groupID.flatten(), lengroup=len(groupID.flatten()), labels=labelsout, lenlabels=len(labelsout))
# #
# # # collect labels and work out collective clashes.
# # labelsall = MPI.collect(np.array([labelsout]))
# # if MPI.rank == 0:
# #     labelsout = labelsall[0]
# #     for i in range(1, len(labelsall)):
# #         labelsout = src.resolve_labels(labels1=labelsout, labels2=labelsall[i], lenlabels=len(labelsout))
# #     MPI.send(labelsout, tag=11)
# # else:
# #     labelsout = MPI.recv(0, tag=11)
# # MPI.wait()
# #
# # labelsout = src.cascade_all(maxlabel=len(labelsout), lenlabels=len(labelsout), labels=labelsout)
# # groupID = src.relabel(group=groupID.flatten(), lengroup=len(groupID.flatten()), labels=labelsout, lenlabels=len(labelsout))
# #
# # if MPI.rank == 0:
# #     newmaxlabel, labelsout = src.shuffle_down(maxlabel=len(labelsout), lenlabels=len(labelsout), labels=labelsout)
# #     MPI.send(labelsout, tag=11)
# #     MPI.send(newmaxlabel, tag=12)
# # else:
# #     labelsout = MPI.recv(0, tag=11)
# #     newmaxlabel = MPI.recv(0, tag=12)
# # MPI.wait()
# # MPI.mpi_print(newmaxlabel)
# #
# # groupID = src.relabel(group=groupID.flatten(), lengroup=len(groupID.flatten()), labels=labelsout, lenlabels=len(labelsout))
# #
# # if len(shape) == 2:
# #     groupID = groupID.reshape(nxgrid, nygrid)
# # else:
# #     groupID = groupID.reshape(nxgrid, nygrid, nzgrid)
