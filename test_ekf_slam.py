import pytest
import numpy as np

from ekf_slam import EKFSlam

@pytest.fixture
def slam():
    return EKFSlam()

# Test to/from Frame methods
# TODO: JACOBIANS

@pytest.mark.parametrize("F,p,expected", [
    (np.array([0,0,0]), np.array([1,2]), np.array([1,2])),
    (np.array([1,1,0]), np.array([2,3]), np.array([1,2])),
])
def test_toFrame(slam, F, p, expected):
    pf, PF_f, PF_p = slam.toFrame(F, p, jacobians=True)
    
    assert np.allclose(pf, expected)
    
    assert PF_f.shape == (p.shape[0], F.shape[0])
    
@pytest.mark.parametrize("F,pf,expected", [
    (np.array([0,0,0]), np.array([1,2]), np.array([1,2])),
    (np.array([1,1,0]), np.array([1,2]), np.array([2,3])),
])
def test_fromFrame(slam, F, pf, expected):
    pf, PF_f, PF_p = slam.fromFrame(F, pf, jacobians=True)
    
    assert np.allclose(pf, expected)
    
    assert PF_f.shape == (pf.shape[0], F.shape[0])
    
@pytest.mark.parametrize("F,p,expected", [
    (np.array([0,0,0]), np.array([1,2]), np.array([1,2])),
    (np.array([1,1,0]), np.array([2,3]), np.array([1,2])),
])
def test_ToFromFrame(slam, F, p, expected):
    pf = slam.toFrame(F, p)
    
    assert np.allclose(pf, expected)
    
    pfp = slam.fromFrame(F, pf)
    
    assert np.allclose(pfp, p)
    
@pytest.mark.parametrize("r,p,expected", [
    (np.array([0.1,-2,0.05]), np.array([-2.66,4]), np.array([1,2])),
])
def test_observe(slam, r, p, expected):
    print(slam.observe(r, p))