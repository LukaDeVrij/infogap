import pytest
import numpy as np

def test_margin_adjustment():
    # create a random matrix of shape 41, 57, with positive numbers between 0 to 1
    cos_matrix = np.random.rand(41, 57)
    # create a random vector of shape 41
    en_hubness_measure = np.random.rand(41)
    # create a random vector of shape 57
    fr_hubness_measure = np.random.rand(57)

    # assert that cos_matrix + en_hubness_measure[:, None]
    # make sure that each row of the matrix is added to the corresponding element of the vector
    # and that the resulting matrix is of shape 41, 57
    assert np.allclose((cos_matrix + en_hubness_measure[:, None]).shape, (41, 57))
    assert cos_matrix[0][0] + en_hubness_measure[0] == (cos_matrix + en_hubness_measure[:, None])[0][0]
    assert cos_matrix[0][1] + en_hubness_measure[0] == (cos_matrix + en_hubness_measure[:, None])[0][1]

    # do the same with the fr hubness measure, but checking along the columns
    assert np.allclose((cos_matrix + fr_hubness_measure[None, :]).shape, (41, 57))
    assert cos_matrix[0][0] + fr_hubness_measure[0] == (cos_matrix + fr_hubness_measure[None, :])[0][0]
    assert cos_matrix[1][0] + fr_hubness_measure[0] == (cos_matrix + fr_hubness_measure[None, :])[1][0]