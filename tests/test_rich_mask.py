from lsst_astropy_bridge import RichMask
import numpy as np


def make_array():
    return np.array([1, 2, 4, 8]*4).reshape(4, 4)


def test_data():
    array = make_array()
    mask = RichMask(array)
    assert((mask.data == array).all())


def test_init_planes():
    array = make_array()
    planes = {'a': 1, 'b': 2, 'c': 4, 'd': 8}
    mask = RichMask(array, planes.keys())
    assert(mask.planes == planes)


def test_add_planes():
    array = make_array()
    planes = {'a': 1, 'b': 2, 'c': 4, 'd': 8}
    mask = RichMask(array)
    mask.add_mask_plane('a')
    mask.add_mask_plane('b')
    mask.add_mask_plane('c')
    mask.add_mask_plane('d')
    assert(mask.planes == planes)


def test_bad_planes():
    # Test bad plane by name
    array = make_array()
    planes = {'a': 1, 'b': 2, 'c': 4, 'd': 8}
    mask = RichMask(array, planes.keys())
    mask.bad_planes = 'b'
    assert(mask.bad_planes == {'b': 2})

    # Test with list of names
    array = make_array()
    planes = {'a': 1, 'b': 2, 'c': 4, 'd': 8}
    mask = RichMask(array, planes.keys())
    mask.bad_planes = ['b', 'd']
    assert(mask.bad_planes == {'b': 2, 'd': 8})

    # Test with bit pattern
    array = make_array()
    planes = {'a': 1, 'b': 2, 'c': 4, 'd': 8}
    mask = RichMask(array, planes.keys())
    mask.bad_planes = 10
    assert(mask.bad_planes == {'b': 2, 'd': 8})

    # Test the output bad mask
    assert((mask.get_bad_mask() ==
            np.array([True, False, True, False]*4).reshape(4, 4)).all())


def test_bool_mask():
    array = make_array()
    planes = {'a': 1, 'b': 2, 'c': 4, 'd': 8}
    mask = RichMask(array, planes)

    # Test with number
    assert((mask.get_bool_mask(1) ==
            np.array([True, False, False, False]*4).reshape(4, 4)).all())

    # Test with name
    assert((mask.get_bool_mask('a') ==
            np.array([True, False, False, False]*4).reshape(4, 4)).all())

    # Test with List
    assert((mask.get_bool_mask(['a', 'b']) ==
            np.array([True, True, False, False]*4).reshape(4, 4)).all())
