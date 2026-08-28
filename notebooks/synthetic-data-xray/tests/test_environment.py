# tests/test_environment.py
import numpy as np
import scipy
import skimage
import synth_xray


def test_dependencies_import():
    assert np.__version__
    assert scipy.__version__
    assert skimage.__version__


def test_package_importable():
    assert synth_xray is not None
