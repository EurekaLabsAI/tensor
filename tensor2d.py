# mostly similar to tensor1d.py with some adjustements to make it work with tensor2d.c

import cffi

# -----------------------------------------------------------------------------
ffi = cffi.FFI()
ffi.cdef(
    """
typedef struct {
    float *data;
    int data_size;
    int ref_count;
} Storage;

// The equivalent of tensor in PyTorch
typedef struct {
    Storage *storage;
    int offset[2];
    int size;
    int nrows;
    int ncols;
    int stride[2];
    char *repr; // holds the text representation of the tensor
} Tensor;

Tensor *tensor_empty(int nrows, int ncols);
float tensor_getitem(Tensor *t, int row, int col);
void tensor_setitem(Tensor *t, int row, int col, float val);
char *tensor_to_string(Tensor *t);
Tensor *tensor_addf(Tensor *t, float val);
Tensor *tensor_add(Tensor *t1, Tensor *t2);
Tensor *tensor_mulf(Tensor *t, float val);
Tensor *tensor_mul(Tensor *t1, Tensor *t2);
Tensor *tensor_dot(Tensor *t1, Tensor *t2);
Tensor *reshape(Tensor *t, int nrows, int ncols);
Tensor *tensor_arange(int size);
Tensor *tensor_slice(Tensor *t, int rstart, int rend, int rstep, int cstart,
                     int cend, int cstep);
"""
)
lib = ffi.dlopen(
    "./libtensor2d.so"
)  # Make sure to compile the C code into a shared library
# -----------------------------------------------------------------------------


class Tensor:
    def __init__(self, size_or_data=None, c_tensor=None):
        # let's ensure only one of size_or_data and c_tensor is passed
        assert (size_or_data is not None) ^ (
            c_tensor is not None
        ), "Either size_or_data or c_tensor must be passed"
        # let's initialize the tensor
        if c_tensor is not None:
            self.tensor = c_tensor
        elif isinstance(size_or_data, tuple):
            self.tensor = lib.tensor_empty(size_or_data[0], size_or_data[1])
        elif isinstance(size_or_data, (list, range)):
            self.tensor = lib.tensor_empty(len(size_or_data), len(size_or_data[0]))
            for i, val in enumerate(size_or_data):
                for j, val2 in enumerate(val):
                    lib.tensor_setitem(self.tensor, i, j, float(val2))
        else:
            raise TypeError("Input must be an integer size or a list/range of values")
        self.shape = (self.tensor.nrows, self.tensor.ncols)

    def __del__(self):
        # TODO: when Python intepreter is shutting down, lib can become None
        # I'm not 100% sure how to do cleanup in cffi here properly
        if lib is not None:
            if hasattr(self, "tensor"):
                lib.tensor_free(self.tensor)

    def __getitem__(self, key):
        # same as tensor.item() in pytorch
        if isinstance(key, tuple):
            value = lib.tensor_getitem(self.tensor, key[0], key[1])
            return value
        elif isinstance(key, list(slice)):
            # assign default values to start, stop, and step
            rstart = key[0].start if key[0].start is not None else 0
            rstop = key[0].stop if key[0].stop is None else self.tensor.size
            rstep = key[0].step if key[0].step is None else 1
            cstart = key[1].start if key[1].start is not None else 0
            cstop = key[1].stop if key[1].stop is None else self.tensor.size
            cstep = key[1].step if key[1].step is None else 1

            # call the C function to slice the tensor
            sliced_tensor = lib.tensor_slice(
                self.tensor, rstart, rstop, rstep, cstart, cstop, cstep
            )
            return Tensor(c_tensor=sliced_tensor)  # Pass the C tensor directly
        else:
            raise TypeError("Invalid index type")

    def __setitem__(self, key, value):
        if isinstance(key, tuple):
            lib.tensor_setitem(self.tensor, key[0], key[1], float(value))
        else:
            raise TypeError("Invalid index type")

    def __add__(self, other):
        if isinstance(other, (int, float)):
            c_tensor = lib.tensor_addf(self.tensor, float(other))
        elif isinstance(other, Tensor):
            c_tensor = lib.tensor_add(self.tensor, other.tensor)
        else:
            raise TypeError("Invalid type for addition")
        if c_tensor == ffi.NULL:
            raise ValueError("RuntimeError: tensor add returned NULL")
        return Tensor(c_tensor=c_tensor)

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            c_tensor = lib.tensor_mulf(self.tensor, float(other))
        elif isinstance(other, Tensor):
            c_tensor = lib.tensor_mul(self.tensor, other.tensor)
        else:
            raise TypeError(
                "Invalid type of multiplication."
            )
        if c_tensor == ffi.NULL:
            raise ValueError("RuntimeError: tensor mul returned NULL")

        return Tensor(c_tensor=c_tensor)

    def __len__(self):
        return self.tensor.size

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        c_str = lib.tensor_to_string(self.tensor)
        py_str = ffi.string(c_str).decode("utf-8")
        return py_str

    def dot(self, other):
        assert isinstance(other, Tensor)
        c_tensor = lib.tensor_dot(self.tensor, other.tensor)
        if c_tensor == ffi.NULL:
            raise ValueError("RuntimeError: tensor dot returned NULL")
        return Tensor(c_tensor=c_tensor)

    def reshape(self, shape=(None, None)):
        assert shape != (None, None), "shape cannot be empty"
        c_tensor = lib.reshape(self.tensor, shape[0], shape[1])
        if c_tensor == ffi.NULL:
            raise ValueError("RuntimeError: tensor reshape returned NULL")
        return Tensor(c_tensor=c_tensor)

    # returns a linear list; just as it is stored in memory
    def tolist(self):
        return [
            lib.tensor_getitem(self.tensor, i, j)
            for i in range(self.tensor.nrows)
            for j in range(self.tensor.ncols)
        ]


def empty(shape=(None, None)):
    return Tensor(size_or_data=shape)


def arange(size=(None, None)):
    c_tensor = lib.tensor_arange(size[0] * size[1])
    c_tensor = lib.reshape(c_tensor, size[0], size[1])
    return Tensor(c_tensor=c_tensor)


def tensor(data=[]):
    return Tensor(size_or_data=data)
