import os
import numpy as np
import struct
from typing import List

def index_file_path(prefix_path: str) -> str:
    return prefix_path + ".idx"

def data_file_path(prefix_path: str) -> str:
    return prefix_path + ".bin"

class MMapIndexedDataset:
    class Index:
        _HDR_MAGIC = b"MMIDIDX\x00\x00"

        @staticmethod
        def _dtype_code(dtype):
            if dtype == np.uint8:
                return 1
            elif dtype == np.int8:
                return 2
            elif dtype == np.int16:
                return 3
            elif dtype == np.int32:
                return 4
            elif dtype == np.int64:
                return 5
            elif dtype == np.float32:
                return 6
            elif dtype == np.float64:
                return 7
            elif dtype == np.uint16:
                return 8
            else:
                raise ValueError(f"Unsupported dtype: {dtype}")

        @classmethod
        def writer(cls, path: str, dtype):
            class _Writer:
                def __enter__(self):
                    self._file = open(path, "wb")
                    self._file.write(cls._HDR_MAGIC)
                    self._file.write(struct.pack("<Q", 1))
                    self._file.write(struct.pack("<B", cls._dtype_code(dtype)))
                    return self

                @staticmethod
                def _get_pointers(sizes: List[int]) -> List[int]:
                    dtype_size = np.dtype(np.int32).itemsize
                    address = 0
                    pointers = []

                    for size in sizes:
                        pointers.append(address)
                        address += size * dtype_size

                    return pointers

                def write(self, sizes: List[int], doc_idx: List[int]):
                    pointers = self._get_pointers(sizes)

                    self._file.write(struct.pack("<Q", len(sizes)))
                    self._file.write(struct.pack("<Q", len(doc_idx)))

                    sizes_array = np.array(sizes, dtype=np.int32)
                    self._file.write(sizes_array.tobytes(order="C"))

                    pointers_array = np.array(pointers, dtype=np.int64)
                    self._file.write(pointers_array.tobytes(order="C"))

                    doc_idx_array = np.array(doc_idx, dtype=np.int64)
                    self._file.write(doc_idx_array.tobytes(order="C"))

                def __exit__(self, exc_type, exc_val, exc_tb):
                    self._file.close()

            return _Writer()

        def __init__(self, path: str):
            with open(path, "rb") as stream:
                magic_test = stream.read(9)
                assert self._HDR_MAGIC == magic_test, "Index file doesn't match expected format."
                
                version = struct.unpack("<Q", stream.read(8))
                assert (1,) == version

                (dtype_code,) = struct.unpack("<B", stream.read(1))
                self._dtype = self._code_to_dtype(dtype_code)
                self._dtype_size = self._dtype().itemsize

                self._len = struct.unpack("<Q", stream.read(8))[0]
                self._doc_count = struct.unpack("<Q", stream.read(8))[0]
                offset = stream.tell()

            self._bin_buffer_mmap = np.memmap(path, mode="r", order="C")
            self._bin_buffer = memoryview(self._bin_buffer_mmap)
            self._sizes = np.frombuffer(self._bin_buffer, dtype=np.int32, count=self._len, offset=offset)
            self._pointers = np.frombuffer(self._bin_buffer, dtype=np.int64, count=self._len,
                                           offset=offset + self._sizes.nbytes)
            self._doc_idx = np.frombuffer(self._bin_buffer, dtype=np.int64, count=self._doc_count,
                                          offset=offset + self._sizes.nbytes + self._pointers.nbytes)

        @staticmethod
        def _code_to_dtype(code: int):
            if code == 1:
                return np.uint8
            elif code == 2:
                return np.int8
            elif code == 3:
                return np.int16
            elif code == 4:
                return np.int32
            elif code == 5:
                return np.int64
            elif code == 6:
                return np.float32
            elif code == 7:
                return np.float64
            elif code == 8:
                return np.uint16
            else:
                raise ValueError(f"Unknown dtype code: {code}")

        def __del__(self):
            self._bin_buffer_mmap._mmap.close()
            del self._bin_buffer_mmap

        @property
        def dtype(self):
            return self._dtype

        @property
        def sizes(self):
            return self._sizes

        @property
        def doc_idx(self):
            return self._doc_idx

        def __getitem__(self, i):
            return self._pointers[i], self._sizes[i]

        def __len__(self):
            return self._len

    def __init__(self, path: str):
        self._path = path
        self._index = self.Index(index_file_path(self._path))
        self._bin_buffer_mmap = np.memmap(data_file_path(self._path), mode="r", order="C")
        self._bin_buffer = memoryview(self._bin_buffer_mmap)

    def __del__(self):
        self._bin_buffer_mmap._mmap.close()
        del self._bin_buffer_mmap
        del self._index

    def __len__(self):
        return len(self._index)

    def __getitem__(self, idx):
        if isinstance(idx, int):
            ptr, size = self._index[idx]
            data = np.frombuffer(self._bin_buffer, dtype=self._index.dtype, count=size, offset=ptr)
            print(f"Retrieved item {idx}, type: {type(data)}, shape: {data.shape}")
            return data
        elif isinstance(idx, slice):
            start, stop, step = idx.indices(len(self))
            if step != 1:
                raise ValueError("Slices into indexed_dataset must be contiguous")
            ptr = self._index._pointers[start]
            sizes = self._index._sizes[idx]
            offsets = np.cumsum(sizes)
            total_size = np.sum(sizes)
            np_array = np.frombuffer(self._bin_buffer, dtype=self._index.dtype, count=total_size, offset=ptr)
            return np.split(np_array, offsets[:-1])

    def get(self, idx: int, offset: int = 0, length: int = None):
        ptr, size = self._index[idx]
        if length is None:
            length = size - offset
        ptr += offset * np.dtype(self._index.dtype).itemsize
        return np.frombuffer(self._bin_buffer, dtype=self._index.dtype, count=length, offset=ptr)

    @property
    def sizes(self):
        return self._index.sizes

    @property
    def doc_idx(self):
        return self._index.doc_idx

    def get_doc_idx(self):
        return self._index._doc_idx

    def set_doc_idx(self, doc_idx_: np.ndarray):
        self._index._doc_idx = doc_idx_

    @staticmethod
    def exists(path: str) -> bool:
        return os.path.exists(index_file_path(path)) and os.path.exists(data_file_path(path))