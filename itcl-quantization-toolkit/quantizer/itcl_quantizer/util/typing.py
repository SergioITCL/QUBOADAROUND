from typing import TypeVar

import numpy as np
import numpy.typing as npt

NDArray = npt.NDArray

NPFP64 = npt.NDArray[np.float64]
NPFP32 = npt.NDArray[np.float32]
NPFP16 = npt.NDArray[np.float16]

NPFP = NPFP64 | NPFP32 | NPFP16

NPI64 = npt.NDArray[np.int64]
NPI32 = npt.NDArray[np.int32]
NPI16 = npt.NDArray[np.int16]
NPI8 = npt.NDArray[np.int8]

NPINT = NPI64 | NPI32 | NPI16 | NPI8

NPU64 = npt.NDArray[np.uint64]
NPU32 = npt.NDArray[np.uint32]
NPU16 = npt.NDArray[np.uint16]
NPU8 = npt.NDArray[np.uint8]

NPUINT = NPU64 | NPU32 | NPU16 | NPU8

E = TypeVar("E", bound=np.generic, covariant=True)


NPGeneric = TypeVar("NPGeneric", bound=np.generic, covariant=True)
