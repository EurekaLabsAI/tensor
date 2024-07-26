/*
Implements a 1-dimensional Tensor, similar to torch.Tensor.

Compile and run like:
gcc -Wall -O3 tensor1d.c -o tensor1d && ./tensor1d

Or create .so for use with cffi:
gcc -O3 -shared -fPIC -o libtensor1d.so tensor1d.c
*/

#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include "tensor1d.h"

// ----------------------------------------------------------------------------
// memory allocation

void *malloc_check(size_t size, const char *file, int line) {
    void *ptr = malloc(size);
    if (ptr == NULL) {
        fprintf(stderr, "Error: Memory allocation failed at %s:%d\n", file, line);
        exit(EXIT_FAILURE);
    }
    return ptr;
}

// ----------------------------------------------------------------------------
// tensor 1D

inline int ceil_div(int a, int b) {
    return (a + b - 1) / b;
}

// torch.empty(size)
Tensor* tensor_empty(int size) {
    // create the storage
    Storage* storage = malloc(sizeof(Storage));
    storage->data = malloc(size * sizeof(float));
    storage->data_size = size;
    storage->ref_count = 1;
    // create the tensor
    Tensor* t = malloc(sizeof(Tensor));
    t->storage = storage;
    t->size = size;
    t->offset = 0;
    t->stride = 1;
    t->repr = NULL;
    return t;
}

// torch.arange(size)
Tensor* tensor_arange(int size) {
    Tensor* t = tensor_empty(size);
    for (int i = 0; i < t->size; i++) {
        tensor_setitem(t, i, (float) i);
    }
    return t;
}

inline int logical_to_physical(Tensor *t, int ix) {
    return t->offset + ix * t->stride;;
}

// val = t[ix]
inline float tensor_getitem(Tensor* t, int ix) {
    // handle negative indices by wrapping around
    if (ix < 0) { ix = t->size + ix; }
    int idx = logical_to_physical(t, ix);
    // handle out of bounds indices
    if(idx >= t->storage->data_size) {
        fprintf(stderr, "Error: Index %d out of bounds of %d\n", ix, t->storage->data_size);
        return -1.0f;
    }
    float val = t->storage->data[idx];
    return val;
}

// t[ix] = val
inline void tensor_setitem(Tensor* t, int ix, float val) {
    // handle negative indices by wrapping around
    if (ix < 0) { ix = t->size + ix; }
    int idx = logical_to_physical(t, ix);
    // handle out of bounds indices
    if(idx >= t->storage->data_size) {
        fprintf(stderr, "Error: Index %d out of bounds of %d\n", ix, t->storage->data_size);
    }
    t->storage->data[idx] = val;
}

// PyTorch (and numpy) actually return a size-1 Tensor when you index like:
// val = t[ix]
// so in this version, we do the same by creating a size-1 slice
Tensor* tensor_getitem_astensor(Tensor* t, int ix) {
    // wrap around negative indices so we can do +1 below with confidence
    if (ix < 0) { ix = t->size + ix; }
    Tensor* slice = tensor_slice(t, ix, ix + 1, 1);
    return slice;
}

// same as torch.Tensor .item() function that strips 1-element Tensor to simple scalar
float tensor_item(Tensor* t) {
    if (t->size != 1) {
        fprintf(stderr, "ValueError: can only convert an array of size 1 to a Python scalar\n");
        return -1.0f;
    }
    return tensor_getitem(t, 0);
}

Tensor* tensor_slice(Tensor* t, int start, int end, int step) {
    // return a new Tensor with a new view, but same Storage
    // 1) handle negative indices by wrapping around
    if (start < 0) { start = t->size + start; }
    if (end < 0) { end = t->size + end; }
    // 2) handle out-of-bounds indices: clip to 0 and t->size
    if (start < 0) { start = 0; }
    if (end < 0) { end = 0; }
    if (start > t->size) { start = t->size; }
    if (end > t->size) { end = t->size; }
    // 3) handle step
    if (step == 0) {
        fprintf(stderr, "ValueError: slice step cannot be zero\n");
        return tensor_empty(0);
    }
    if (step < 0) {
        // TODO possibly support negative step
        // PyTorch does not support negative step (numpy does)
        fprintf(stderr, "ValueError: slice step cannot be negative\n");
        return tensor_empty(0);
    }
    // create the new Tensor: same Storage but new View
    Tensor* s = malloc(sizeof(Tensor));
    s->storage = t->storage; // inherit the underlying storage!
    s->size = ceil_div(end - start, step);
    s->offset = t->offset + start * t->stride;
    s->stride = t->stride * step;
    tensor_incref(s);
    return s;
}

char* tensor_to_string(Tensor* t) {
    // if we already have a string representation, return it
    if (t->repr != NULL) {
        return t->repr;
    }
    // otherwise create a new string representation
    int max_size = t->size * 20 + 3; // 20 chars/number, brackets and commas
    t->repr = malloc(max_size);
    char* current = t->repr;

    current += sprintf(current, "[");
    for (int i = 0; i < t->size; i++) {
        float val = tensor_getitem(t, i);
        current += sprintf(current, "%.1f", val);
        if (i < t->size - 1) {
            current += sprintf(current, ", ");
        }
    }
    current += sprintf(current, "]");
    // check that we didn't write past the end of the buffer
    assert(current - t->repr < max_size);
    return t->repr;
}

void tensor_print(Tensor* t) {
    char* str = tensor_to_string(t);
    printf("%s\n", str);
    free(str);
}

// reference counting functions so we know when to deallocate Storage
// (there can be multiple Tensors with different View that share the same Storage)
void tensor_incref(Tensor* t) {
    t->storage->ref_count++;
}

void tensor_decref(Tensor* t) {
    t->storage->ref_count--;
    if (t->storage->ref_count == 0) {
        free(t->storage->data);
        free(t->storage);
    }
}

void tensor_free(Tensor* t) {
    tensor_decref(t); // storage-related cleanups
    free(t->repr);
    free(t);
}

// ----------------------------------------------------------------------------

int main(int argc, char *argv[]) {
    // create a tensor with 20 elements
    Tensor* t = tensor_arange(20);
    tensor_print(t);
    // slice the tensor as t[5:15:1]
    Tensor* s = tensor_slice(t, 5, 15, 1);
    tensor_print(s);
    // slice that tensor as s[2:7]
    Tensor* ss = tensor_slice(s, 2, 7, 2);
    tensor_print(ss);
    // print element -1
    printf("ss[-1] = %.1f\n", tensor_getitem(ss, -1));
    return 0;
}