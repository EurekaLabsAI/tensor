#include "tensor2d.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// ----------------------------------------------------------------------------
// memory allocation

void *malloc_check(size_t size, const char *file, int line) {
    void *ptr = malloc(size);
    if (ptr == NULL) {
        fprintf(stderr, "Error: Memory allocation failed at %s:%d\n", file,
                line);
        exit(EXIT_FAILURE);
    }
    return ptr;
}
#define mallocCheck(size) malloc_check(size, __FILE__, __LINE__)

// ----------------------------------------------------------------------------
// utils

int ceil_div(int a, int b) {
    // integer division that rounds up, i.e. ceil(a / b)
    return (a / b) + ((a % b) != 0);
}

int min(int a, int b) {
    return (a < b) ? a : b;
}

int max(int a, int b) {
    return (a > b) ? a : b;
}

Storage *storage_new(int size) {
    assert(size >= 0);
    Storage *storage = mallocCheck(sizeof(Storage));
    storage->data = mallocCheck(size * sizeof(float));
    storage->data_size = size;
    storage->ref_count = 1;
    return storage;
}

float storage_getitem(Storage *s, int idx) {
    assert(idx >= 0 && idx < s->data_size);
    return s->data[idx];
}

void storage_setitem(Storage *s, int idx, float val) {
    assert(idx >= 0 && idx < s->data_size);
    s->data[idx] = val;
}

void storage_incref(Storage *s) {
    s->ref_count++;
}

void storage_decref(Storage *s) {
    s->ref_count--;
    if (s->ref_count == 0) {
        free(s->data);
        free(s);
    }
}

// ----------------------------------------------------------------------------
// Tensor class functions
Tensor *tensor_empty(int nrows, int ncols) {
    Tensor *t = mallocCheck(sizeof(Tensor));
    t->storage = storage_new(nrows * ncols);
    t->offset[0] = 0;
    t->offset[1] = 0;
    t->nrows = nrows;
    t->ncols = ncols;
    t->size = nrows * ncols;
    // row major ordering
    t->stride[0] = ncols;
    t->stride[1] = 1;
    t->repr = NULL;
    return t;
}

int logical_to_physical(Tensor *t, int row, int col) {
    int index =
        t->offset[0] + row * t->stride[0] + t->offset[1] + col * t->stride[1];
    return index;
}

float tensor_getitem(Tensor *t, int row, int col) {
    // handle negative indices by wrapping around
    if (row < 0) {
        row = t->nrows + row;
    }
    if (col < 0) {
        col = t->ncols + col;
    }
    if (row * col >= t->size) {
        fprintf(stderr,
                "IndexError: index [%d, %d] is out of bounds for Tensor of "
                "size: %d",
                row, col, t->size);
        return NAN;
    }
    int index = logical_to_physical(t, row, col);
    float val = storage_getitem(t->storage, index);
    return val;
}

void tensor_setitem(Tensor *t, int row, int col, float val) {
    // handle negative indices by wrapping around
    if (row < 0) {
        row = t->nrows + row;
    }
    if (col < 0) {
        col = t->ncols + col;
    }
    if (row * col >= t->size) {
        fprintf(stderr,
                "IndexError: index [%d, %d] is out of bounds for Tensor of "
                "size: %d",
                row, col, t->size);
        return;
    }
    int index = logical_to_physical(t, row, col);
    storage_setitem(t->storage, index, val);
}

char *tensor_to_string(Tensor *t) {
    if (t->repr != NULL) {
        return t->repr;
    }
    int max_size = t->size * 20 + 5;
    t->repr = mallocCheck(max_size);
    char *current = t->repr;
    current += sprintf(current, "[");
    for (int i = 0; i < t->nrows; i++) {
        if (i > 0) {
            current += sprintf(current, " ");
        }
        current += sprintf(current, "[");
        for (int j = 0; j < t->ncols; j++) {
            current += sprintf(current, "%.1f", tensor_getitem(t, i, j));
            if (j < t->ncols - 1) {
                current += sprintf(current, ", ");
            }
        }
        current += sprintf(current, "]");
        if (i < t->nrows - 1) {
            current += sprintf(current, "\n");
        }
    }
    current += sprintf(current, "]");
    assert(current - t->repr < max_size);
    return t->repr;
}

Tensor *reshape(Tensor *t, int nrows, int ncols) {
    // ensure tensor is reshapable
    if (nrows * ncols != t->size) {
        fprintf(stderr,
                "ValueError: cannot reshape tensor of size: %d to tensor of "
                "size: %d",
                t->size, nrows * ncols);
        return NULL;
    }
    Tensor *view = mallocCheck(sizeof(Tensor));
    view->storage = t->storage;
    view->nrows = nrows;
    view->ncols = ncols;
    view->size = nrows * ncols;
    // row major ordering
    view->stride[0] = ncols;
    view->stride[1] = 1;
    view->offset[0] = 0;
    view->offset[1] = 0;
    view->repr = NULL;
    storage_incref(view->storage);
    return view;
}

// generate a (1, size) tensor
Tensor *tensor_arange(int size) {
    Tensor *t = tensor_empty(1, size);
    for (int i = 0; i < t->size; i++) {
        tensor_setitem(t, 0, i, (float)i);
    }
    return t;
}

// t[rstart:rstop:rend, cstart:cend:cstep]
Tensor *tensor_slice(Tensor *t, int rstart, int rend, int rstep, int cstart,
                     int cend, int cstep) {
    // 1) handle negative indices by wrapping around
    if (rstart < 0) rstart = t->nrows + rstart;
    if (rend < 0) rend = t->nrows + rend;
    if (cstart < 0) cstart = t->ncols + cstart;
    if (cend < 0) cend = t->ncols + cend;
    // 2) handle inappropriate indices
    if ((rstart > rend && rstep > 0) || (cstart > cend && cstep > 0)) {
        fprintf(stderr, "ValueError: slice step cannot be positive\n");
        return tensor_empty(0, 0);
    }
    // 3) handle out-of-bounds indices: clip to [0, t->nrows] and [0, t->ncols]
    rstart = min(max(rstart, 0), t->nrows);
    rend = min(max(rend, 0), t->nrows);
    cstart = min(max(cstart, 0), t->ncols);
    cend = min(max(cend, 0), t->ncols);
    // 4) handle step
    if (rstep == 0 || cstep == 0) {
        fprintf(stderr, "ValueError: slice step cannot be zero\n");
        return tensor_empty(0, 0);
    }
    // create a new view of the Tensor t.
    Tensor *view = mallocCheck(sizeof(Tensor));
    view->storage = t->storage;
    view->nrows = ceil_div(rend - rstart, rstep);
    view->ncols = ceil_div(cend - cstart, cstep);
    view->size = view->nrows * view->ncols;
    view->offset[0] = t->offset[0] + rstart * t->stride[0];
    view->offset[1] = t->offset[1] + cstart * t->stride[1];
    view->stride[0] = t->stride[0] * rstep;
    view->stride[1] = t->stride[1] * cstep;
    view->repr = NULL;
    storage_incref(view->storage);
    return view;
}

Tensor *tensor_addf(Tensor *t, float val) {
    Tensor *result = tensor_empty(t->nrows, t->ncols);
    for (int i = 0; i < t->nrows; i++) {
        for (int j = 0; j < t->ncols; j++) {
            float old_val = tensor_getitem(t, i, j);
            tensor_setitem(result, i, j, old_val + val);
        }
    }
    return result;
}

// add two tensors of similar shape; doesn't support broadcasting
Tensor *tensor_add(Tensor *t1, Tensor *t2) {
    if (t1->nrows != t2->nrows && t1->ncols != t2->ncols) {
        fprintf(stderr,
                "ValueError: incompatible shapes. cannot add tensor of shape "
                "(%d, %d) to tensor of shape (%d, %d)\n",
                t1->nrows, t1->ncols, t2->nrows, t2->ncols);
        return NULL;
    }
    Tensor *result = tensor_empty(t1->nrows, t1->ncols);
    for (int i = 0; i < t1->nrows; i++) {
        for (int j = 0; j < t1->ncols; j++) {
            float val1 = tensor_getitem(t1, i, j);
            float val2 = tensor_getitem(t2, i, j);
            tensor_setitem(result, i, j, val1 + val2);
        }
    }
    return result;
}

// scalar multiplication
Tensor *tensor_mulf(Tensor *t, float val) {
    Tensor *result = tensor_empty(t->nrows, t->ncols);
    for (int i = 0; i < t->nrows; i++) {
        for (int j = 0; j < t->ncols; j++) {
            float old_val = tensor_getitem(t, i, j);
            tensor_setitem(result, i, j, old_val * val);
        }
    }
    return result;
}

// element-wise multiplication
Tensor *tensor_mul(Tensor *t1, Tensor *t2) {
    if ((t1->nrows != t2->nrows) || (t1->ncols != t2->ncols)) {
        fprintf(stderr, "ValueError: shapes not equal\n");
        return NULL;
    }
    Tensor *result = tensor_empty(t1->nrows, t1->ncols);
    for (int i = 0; i < t1->nrows; i++) {
        for (int j = 0; j < t1->ncols; j++) {
            float val1 = tensor_getitem(t1, i, j);
            float val2 = tensor_getitem(t2, i, j);
            tensor_setitem(result, i, j, val1 * val2);
        }
    }
    return result;
}

// similar to np.dot(t1, t2)
Tensor *tensor_dot(Tensor *t1, Tensor *t2) {
    if (t1->ncols != t2->nrows) {
        fprintf(stderr,
                "ValueError: incompatible shapes. cannot perform dot product "
                "shape: (%d, %d) with shape: (%d, %d) %d not equal to %d\n",
                t1->nrows, t1->ncols, t2->nrows, t2->ncols, t1->ncols,
                t2->nrows);
        return NULL;
    }
    Tensor *result = tensor_empty(t1->nrows, t2->ncols);
    for (int i = 0; i < t1->nrows; i++) {
        for (int j = 0; j < t2->ncols; j++) {
            float value = 0.0f;
            for (int k = 0; k < t1->ncols; k++) {
                float val1 = tensor_getitem(t1, i, k);
                float val2 = tensor_getitem(t2, k, j);
                value += val1 * val2;
            }
            tensor_setitem(result, i, j, value);
        }
    }
    return result;
}

void tensor_print(Tensor *t) {
    char *str = tensor_to_string(t);
    printf("%s\n", str);
}

void tensor_free(Tensor *t) {
    storage_decref(t->storage);
    free(t->repr);
    free(t);
}

int main(int argc, char *argv[]) {
    Tensor *t = tensor_arange(10);
    Tensor *t2 = reshape(t, 5, 2);
    printf("shape: (%d, %d)\n", t2->nrows, t2->ncols);
    tensor_print(t2);

    printf("---------------------------------\n");

    Tensor *t3 = tensor_slice(t2, 3, 1, -1, 0, 2, 1);
    printf("shape: (%d, %d)\n", t3->nrows, t3->ncols);
    tensor_print(t3);

    tensor_free(t);
    tensor_free(t2);
    tensor_free(t3);
    return 0;
}
