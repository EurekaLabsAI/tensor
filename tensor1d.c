/*
Implements a 1-dimensional Tensor, similar to torch.Tensor.

Compile and run like:
gcc -Wall -O3 tensor1d.c -o tensor1d && ./tensor1d

Or create .so for use with cffi:
gcc -O3 -shared -fPIC -o libtensor1d.so tensor1d.c
*/

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
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
#define mallocCheck(size) malloc_check(size, __FILE__, __LINE__)

// ----------------------------------------------------------------------------
// utils

int ceil_div(int a, int b) {
    // integer division that rounds up, i.e. ceil(a / b)
    return (a + b - 1) / b;
}

int min(int a, int b) {
    return (a < b) ? a : b;
}

int max(int a, int b) {
    return (a > b) ? a : b;
}

// ----------------------------------------------------------------------------
// Storage: simple array of floats, defensive on index access, reference-counted
// The reference counting allows multiple Tensors sharing the same Storage.
// similar to torch.Storage

Storage* storage_new(int size) {
    assert(size >= 0);
    Storage* storage = mallocCheck(sizeof(Storage));
    storage->data = mallocCheck(size * sizeof(float));
    storage->data_size = size;
    storage->ref_count = 1;
    return storage;
}

float storage_getitem(Storage* s, int idx) {
    assert(idx >= 0 && idx < s->data_size);
    return s->data[idx];
}

void storage_setitem(Storage* s, int idx, float val) {
    assert(idx >= 0 && idx < s->data_size);
    s->data[idx] = val;
}

void storage_incref(Storage* s) {
    s->ref_count++;
}

void storage_decref(Storage* s) {
    s->ref_count--;
    if (s->ref_count == 0) {
        free(s->data);
        free(s);
    }
}

// ----------------------------------------------------------------------------
// Tensor class functions

// torch.empty(size)
Tensor* tensor_empty(int size) {
    Tensor* t = mallocCheck(sizeof(Tensor));
    t->storage = storage_new(size);
    // at init we cover the whole storage, i.e. range(start=0, stop=size, step=1)
    t->offset = 0;
    t->size = size;
    t->stride = 1;
    // holds the text representation of the tensor
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

int logical_to_physical(Tensor *t, int ix) {
    int idx = t->offset + ix * t->stride;
    return idx;
}

// Index into the tensor.
// Note that both PyTorch and numpy actually return a 1-element Tensor when you index like:
// val = t[ix]
// This particular function returns the actual float, i.e.:
// val = t[ix].item()
float tensor_getitem(Tensor* t, int ix) {
    // handle negative indices by wrapping around
    if (ix < 0) { ix = t->size + ix; }
    // oob indices raise IndexError (and we return NaN)
    if (ix >= t->size) {
        fprintf(stderr, "IndexError: index %d is out of bounds of %d\n", ix, t->size);
        return NAN;
    }
    // get the physical index into the storage and return the value
    int idx = logical_to_physical(t, ix);
    float val = storage_getitem(t->storage, idx);
    return val;
}

// The _astensor version of getitem:
// val = t[ix]
// i.e. consistent with PyTorch/numpy create a 1-element Tensor and return it
Tensor* tensor_getitem_astensor(Tensor* t, int ix) {
    // wrap around negative indices so we can do +1 below with confidence
    if (ix < 0) { ix = t->size + ix; }
    // effectively: t[ix:ix+1:1] <=> t[ix:ix+1] <=> t[ix]
    Tensor* slice = tensor_slice(t, ix, ix + 1, 1);
    return slice;
}

// t[ix] = val
void tensor_setitem(Tensor* t, int ix, float val) {
    // handle negative indices by wrapping around
    if (ix < 0) { ix = t->size + ix; }
    if (ix >= t->size) {
        fprintf(stderr, "IndexError: index %d is out of bounds of %d\n", ix, t->size);
        return;
    }
    int idx = logical_to_physical(t, ix);
    storage_setitem(t->storage, idx, val);
}

// same as .item() on a torch.Tensor: strips 1-element Tensor to simple scalar
float tensor_item(Tensor* t) {
    if (t->size != 1) {
        fprintf(stderr, "ValueError: can only convert an array of size 1 to a Python scalar\n");
        return NAN;
    }
    return tensor_getitem(t, 0);
}

// return a new Tensor with a new view, but same Storage, i.e.:
// t[start:end:step]
Tensor* tensor_slice(Tensor* t, int start, int end, int step) {
    // 1) handle negative indices by wrapping around
    if (start < 0) { start = t->size + start; }
    if (end < 0) { end = t->size + end; }
    // 2) handle out-of-bounds indices: clip to [0, t->size] range
    start = min(max(start, 0), t->size);
    end = min(max(end, 0), t->size);
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
    Tensor* s = mallocCheck(sizeof(Tensor));
    s->storage = t->storage; // inherit the underlying storage!
    s->size = ceil_div(end - start, step);
    s->offset = t->offset + start * t->stride;
    s->stride = t->stride * step;
    s->repr = NULL;
    storage_incref(s->storage); // increment the reference count
    return s;
}

Tensor* tensor_addf(Tensor* t, float val) {
    // adds a float to each element of the tensor, returns a new tensor
    Tensor* result = tensor_empty(t->size);
    for (int i = 0; i < t->size; i++) {
        float old_val = tensor_getitem(t, i);
        float new_val = old_val + val;
        tensor_setitem(result, i, new_val);
    }
    return result;
}

bool broadcastable(Tensor* t1, Tensor* t2) {
    // two tensors broadcast if, in each dimension (we only have 1 here)
    // tensors either have the same size, or one of their sizes is 1
    return t1->size == t2->size || t1->size == 1 || t2->size == 1;
}

Tensor* tensor_add(Tensor* t1, Tensor* t2) {
    if (!broadcastable(t1, t2)) { return NULL; }
    int result_size = max(t1->size, t2->size);
    Tensor* result = tensor_empty(result_size);
    int t1_index = 0;
    int t2_index = 0;
    int t1_stride = t1->size > 1 ? 1 : 0; // either we walk this tensor or not
    int t2_stride = t2->size > 1 ? 1 : 0; // either we walk this tensor or not
    // walk the output tensor and add the values
    for (int result_index = 0; result_index < result_size; result_index++) {
        float val1 = tensor_getitem(t1, t1_index);
        float val2 = tensor_getitem(t2, t2_index);
        float val = val1 + val2;
        tensor_setitem(result, result_index, val);
        t1_index += t1_stride;
        t2_index += t2_stride;
    }
    return result;
}

char* tensor_to_string(Tensor* t) {
    // if we already have a string representation, return it
    if (t->repr != NULL) { return t->repr; }
    // otherwise create a new string representation
    int max_size = t->size * 20 + 3; // 20 chars/number, brackets and commas
    t->repr = mallocCheck(max_size);
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
    // ensure we didn't write past the end of the buffer
    assert(current - t->repr < max_size);
    return t->repr;
}

void tensor_print(Tensor* t) {
    char* str = tensor_to_string(t);
    printf("%s\n", str);
}

void tensor_free(Tensor* t) {
    storage_decref(t->storage);
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
    // slice that tensor as s[2:7:2]
    Tensor* ss = tensor_slice(s, 2, 7, 2);
    tensor_print(ss);
    // print element -1
    float val = tensor_getitem(ss, -1);
    printf("ss[-1] = %.1f\n", val);

    tensor_free(ss);
    tensor_free(s);
    tensor_free(t);

    return 0;
}
