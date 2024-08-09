import cffi

# -----------------------------------------------------------------------------
ffi = cffi.FFI()
ffi.cdef("""
typedef struct {
    float* data;
    int data_size;
    int ref_count;
} Storage;

// The equivalent of tensor in PyTorch
typedef struct {
    Storage* storage;
    int offset;
    int size;
    int stride;
    char* repr; // holds the text representation of the tensor
} Tensor;

Tensor* tensor_empty(int size);
Tensor* tensor_zeros(int size);
int logical_to_physical(Tensor *t, int ix);
float tensor_getitem(Tensor* t, int ix);
Tensor* tensor_getitem_astensor(Tensor* t, int ix);
float tensor_item(Tensor* t);
void tensor_setitem(Tensor* t, int ix, float val);
Tensor* tensor_arange(int size);
char* tensor_to_string(Tensor* t);
void tensor_print(Tensor* t);
Tensor* tensor_slice(Tensor* t, int start, int end, int step);
Tensor* tensor_addf(Tensor* t, float val);
Tensor* tensor_add(Tensor* t1, Tensor* t2);
void tensor_incref(Tensor* t);
void tensor_decref(Tensor* t);
void tensor_free(Tensor* t);
""")
lib = ffi.dlopen("./libtensor1d.so")  # Make sure to compile the C code into a shared library
# -----------------------------------------------------------------------------

class Tensor:
    def __init__(self, size_or_data=None, c_tensor=None):
        # let's ensure only one of size_or_data and c_tensor is passed
        assert (size_or_data is not None) ^ (c_tensor is not None), "Either size_or_data or c_tensor must be passed"
        # let's initialize the tensor
        if c_tensor is not None:
            self.tensor = c_tensor
        elif isinstance(size_or_data, int):
            self.tensor = lib.tensor_empty(size_or_data)
        elif isinstance(size_or_data, (list, range)):
            self.tensor = lib.tensor_arange(len(size_or_data))
            for i, val in enumerate(size_or_data):
                lib.tensor_setitem(self.tensor, i, float(val))
        else:
            raise TypeError("Input must be an integer size or a list/range of values")

    def __del__(self):
        # TODO: when Python intepreter is shutting down, lib can become None
        # I'm not 100% sure how to do cleanup in cffi here properly
        if lib is not None:
            if hasattr(self, 'tensor'):
                lib.tensor_free(self.tensor)

    def __getitem__(self, key):
        if isinstance(key, int):
            c_tensor = lib.tensor_getitem_astensor(self.tensor, key)
            return Tensor(c_tensor=c_tensor)
        elif isinstance(key, slice):
            # assign default values to start, stop, and step
            start = key.start if key.start is not None else 0
            stop = self.tensor.size if key.stop is None else key.stop
            step = 1 if key.step is None else key.step
            # call the C function to slice the tensor
            sliced_tensor = lib.tensor_slice(self.tensor, start, stop, step)
            return Tensor(c_tensor=sliced_tensor)  # Pass the C tensor directly
        else:
            raise TypeError("Invalid index type")

    def __setitem__(self, key, value):
        if isinstance(key, int):
            lib.tensor_setitem(self.tensor, key, float(value))
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

    def __len__(self):
        return self.tensor.size

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        c_str = lib.tensor_to_string(self.tensor)
        py_str = ffi.string(c_str).decode('utf-8')
        return py_str

    def tolist(self):
        return [lib.tensor_getitem(self.tensor, i) for i in range(len(self))]

    def item(self):
        return lib.tensor_item(self.tensor)

def empty(size):
    return Tensor(size)

def arange(size):
    c_tensor = lib.tensor_arange(size)
    return Tensor(c_tensor=c_tensor)

def zeros(size):
    c_tensor = lib.tensor_zeros(size)
    return Tensor(c_tensor=c_tensor)

def tensor(data):
    return Tensor(data)