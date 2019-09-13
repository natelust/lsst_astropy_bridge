import numpy as np
import typing
from collections import OrderedDict
from astropy.nddata import bitmask


class RichMask:
    """Class to represent named mask planes with a bit pattern in an array

    Parameters
    ----------
    data : `numpy.array`
        A numpy array used to represent a named mask with a defined bit
        pattern
    planes : Iterable of `str`
        Names used to associate with mask planes
    """
    def __init__(self, data: np.array, planes: typing.Iterable[str]=None):
        self._array = data
        if planes is not None:
            self._planes = OrderedDict({k: i for i, k in enumerate(planes)})
        else:
            self._planes = OrderedDict()
        self._planes_set = set(self._planes.keys())
        self._bad_planes = set()

    def add_mask_plane(self, name: str):
        """Add a new mask plane named according to input string

        Parameters
        ----------
        name : `str`
            The name to use for the new mask plane
        """
        if name in self.planes:
            raise ValueError(f"A mask plane of name {name} already exists")
        if len(self._planes) == 0:
            last_plane_num = -1
        else:
            last_plane_num = next(reversed(self._planes.values()))
        self._planes[name] = last_plane_num + 1
        self._planes_set.add(name)

    @property
    def bad_planes(self) -> typing.Dict[str, int]:
        return {name: 2**self._planes[name] for name in self._bad_planes}

    @bad_planes.setter
    def bad_planes(self, planes: typing.Union[str, typing.Iterable[str],
                                              int]):
        if isinstance(planes, str):
            planes = tuple(planes)

        if isinstance(planes, int):
            tmp = []
            for name, number in self._planes.items():
                if planes & 2**number:
                    tmp.append(name)
            planes = tmp

        planes = set(planes)

        self._check_planes(planes)

        self._bad_planes = self._bad_planes.union(planes)

    @property
    def data(self)->np.array:
        return self._array

    def get_bad_mask(self)->np.array:
        bad_planes = self._args_to_bits(self.bad_planes)
        return self.get_bool_mask(~bad_planes)

    @property
    def planes(self) -> dict:
        return {name: 2**num for name, num in self._planes.items()}

    def clear(self, planes: typing.Union[None, str, typing.List[str],
                                         int]=None):
        if planes is None:
            self._array = np.zeros(self._array.shape)
            return
        bits = self.args_to_bits(planes)
        self._array = self._array & ~bits

    def get_bool_mask(self, planes: typing.Union[str, typing.List[str],
                                                 int]) -> np.array:
        bits = self._args_to_bits(planes)
        return bitmask.bitfield_to_boolean_mask(self._array,
                                                ignore_flags=~bits)

    def _check_planes(self, planes: set):
        extra_planes = planes - self._planes_set
        if extra_planes:
            raise ValueError(f"Planes {extra_planes} are not in the rich mask")

    def _args_to_bits(self, planes: typing.Union[None, str, typing.List[str],
                                                 int]) -> int:
        if not isinstance(planes, int):
            if isinstance(planes, str):
                planes = tuple(planes)
            self._check_planes(set(planes))
            bits = 0
            for name in planes:
                bits ^= 2**self._planes[name]
        else:
            bits = planes
        return bits
