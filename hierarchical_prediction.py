from typing import List, Tuple
import os

import numpy as np

import pyvista as pv

from joblib import Parallel, delayed
from joblib import parallel_backend

from morphomatics.manifold.Bezierfold import Bezierfold
from morphomatics.geom import BezierSpline, Surface
from morphomatics.manifold import ShapeSpace
from morphomatics.stats import RiemannianRegression, StatisticalShapeModel
from morphomatics.manifold import DifferentialCoords, FundamentalCoords, PointDistributionModel
from morphomatics.manifold.util import align

from util import mean_vertex_dist

from sklearn.model_selection import KFold


def create_shape_space(surfaces: List[List[Surface]], space: str = 'DCM') -> ShapeSpace:
    """Create shape space from longitudinal surface data.

    :param surf: list of subject-wise lists of surfaces that are sorted by time.
    :param space: string indicating the shape space. "DCM" = differential coordinate space, "FCM" = fundamental
    coordinate space, and "PDM" = point distribution model are possible.
    :return: shape space object
    """
    # space may be "DCM", "FCM", "PDM"

    if space == "DCM":
        SSM = StatisticalShapeModel(lambda ref: DifferentialCoords(ref))
    elif space == "FCM":
        SSM = StatisticalShapeModel(lambda ref: FundamentalCoords(ref))
    elif space == "PDM":
        SSM = StatisticalShapeModel(lambda ref: PointDistributionModel(ref))
    else:
        raise ValueError("No shape space was given.")

    # use the intrinsic mean as the reference
    SSM.construct([x for xs in surfaces for x in xs])
    ref = SSM.mean
    # ref = surfaces[0][0]

    # create shape space
    if space == "DCM":
        M = DifferentialCoords(ref)
    elif space == "FCM":
        os.environ['FCM_INIT_FACE'] = '12312'
        os.environ['FCM_INIT_VERT'] = '331'
        M = FundamentalCoords(ref)
    else:
        M = PointDistributionModel(ref)

    return M


def cross_validation_hierarchical_prediction(M: ShapeSpace, surfaces: List[List[Surface]], t: np.array,) \
        -> Tuple[List[np.array], List[np.array]]:
    """Perform cross-validation to approximate the accuracy of the shape prediction.

    :param M: Shape space
    :param surfaces: list of subject-wise lists of surfaces that are sorted by time.
    :param t: array of time labels (must be the same for each subject).
    :return: lists of vertex-wise shape-distance errors.
    """

    C = []  # list of encoded shapes
    for subject in surfaces:
        subject_coord = []  # differential coordinates of a single subject at every time point
        for S in subject:
            subject_coord.append(M.to_coords(S.v))

        C.append(np.array(subject_coord))

    # uncomment the following to perform regression

    # def reg(_c):
    #     regression = RiemannianRegression(M, _c, t, 1)  # geodesic regression for each training subject independently
    #     return regression.trend
    #
    # # regress geodesics for every subject (avoids re-computations in cross-validation)
    # with parallel_backend('multiprocessing'):
    #     gams = Parallel(n_jobs=-1, prefer='threads', require='sharedmem', verbose=10)(delayed(reg)(c) for c in C)
    #
    # control_points = np.array([g.control_points[0] for g in gams])
    # np.save('data/regressed_geodesics.npy', control_points)

    control_points = np.load('data/regressed_geodesics.npy')
    gams = [BezierSpline(M, [cp]) for cp in control_points]

    distances = []  # list to keep distances
    errors = []  # list to keep vertex-wise errors

    # do cross-validation
    kf = KFold(n_splits=3, shuffle=True)
    for train_index, test_index in kf.split(surfaces):
        # print("TRAIN:", train_index, "TEST:", test_index)
        surf_ad, surf_test = surfaces[train_index[0]:train_index[-1]], surfaces[test_index[0]:test_index[-1]]
        C_ad, C_test = C[train_index[0]:train_index[-1]], C[test_index[0]:test_index[-1]]
        gams_train = gams[train_index[0]:train_index[-1]]

        # manifold of geodesics through M
        B = Bezierfold(M, 1)  # 1 in second entry for geodesics
        # instantiate structure
        # B.initFunctionalBasedStructure()

        # compute mean only for training
        mean = B.mean(gams_train, n=3, delta=1e-5, nsteps=5, eps=1e-5, n_stepsGeo=5, verbosity=2)[0]

        distances_fold = []
        errors_fold = []

        for i, Y in enumerate(C_test):
            predicted_shape, predicted_vertices = predict_from_mean_geodesic(M, mean, Y[0])
            aligned_v = align(predicted_vertices, surf_test[i][-1].v)  # align

            dist = M.metric.dist(Y[-1], predicted_shape)  # calculate geodesic distance between predicted shape and ground truth
            distances_fold.append(dist)  # add distance to list

            e, _ = mean_vertex_dist(aligned_v, surf_test[i][-1].v)
            errors_fold.append(e)  # add error to list

        distances.append(np.mean(np.array(distances_fold)))
        errors.append(np.mean(np.array(errors_fold)))

    return distances, errors


def predict_from_mean_geodesic(M: ShapeSpace, m: BezierSpline, test_subject: np.array):
    """Predict the development of a shape from the mean trajectory.

    :param M: underlying shape space
    :param m: mean geodesic of the training subjects
    :param test_subject: shape space coordinates of the shape whose development we want to predict
    :return: predicted coordinates and the vertices of the corresponding triangular mesh
    """
    start_m = m.control_points[0][0]  # start point of the mean
    end_m = m.control_points[0][1]  # end point of the mean
    tangent_m = M.connec.log(start_m, end_m)  # compute tangent

    start_test = test_subject  # start of new data point whose development we want to predict
    tangent_test = M.connec.transp(start_m, start_test, tangent_m)  # parallel transport
    predicted_coords = M.connec.exp(start_test, tangent_test)  # exponential map

    new_v = M.from_coords(predicted_coords)  # find vertex-wise coordinates

    return predicted_coords, new_v
