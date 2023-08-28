from typing import Tuple, List

import os

import numpy as np

import pyvista as pv

from morphomatics.manifold.util import align
from morphomatics.geom import Surface


def mean_vertex_dist(V: np.array, W: np.array) -> Tuple[np.array, np.array]:
    """Mean distance between corresponding vertices in V and W"""
    V = align(V, W)
    d = np.linalg.norm(V - W, axis=1)
    
    return np.mean(d), np.std(d)


def load_surfaces_dfaust() -> Tuple[List[List[Surface]], List[List[pv.DataSet]]]:
    """ Load Dynamic Faust data as Surface files"""
    T = []
    surf = []  # list of lists: each entry will be a list containing the three surfaces that belong to the same subject

    type = '/data/dynamic_faust/'

    foldernames = [f for f in os.listdir(os.getcwd() + type)]
    foldernames.sort()

    # read
    for foldername in foldernames:
        subfolders = [f for f in os.listdir(os.getcwd() + type + foldername)]
        subfolders.sort()
        subject = []  # list containing surfaces of a single subject ordered by time
        T_subject = []
        for m in range(len(subfolders)):
            filename = os.getcwd() + type + foldername + "/" + subfolders[m]

            pyT = pv.read(filename)
            v = np.array(pyT.points)
            f = pyT.faces.reshape(-1, 4)[:, 1:]
            subject.append(Surface(v, f))
            T_subject.append(pyT)

        if len(subject):
            surf.append(subject)
            T.append(T_subject)

    return surf[::-1], T
