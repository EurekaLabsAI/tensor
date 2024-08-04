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
    return (a + b - 1) / b;
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
    t->offset = 0;
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
    // TODO: support sliced indexing
    int index = row * t->stride[0] + col * t->stride[1];
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
    }
    Tensor *view = mallocCheck(sizeof(Tensor));
    view->storage = t->storage;
    view->nrows = nrows;
    view->ncols = ncols;
    view->size = nrows * ncols;
    // row major ordering
    view->stride[0] = ncols;
    view->stride[1] = 1;
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

void tensor_print(Tensor *t) {
    char *str = tensor_to_string(t);
    printf("%s\n", str);
}

int main(int argc, char *argv[]) {
    Tensor *t = tensor_arange(10);
    printf("Tensor shape: (%d, %d)\n", t->nrows, t->ncols);
    tensor_print(t);

    Tensor *t2 = reshape(t, 5, 2);
    printf("Tensor shape: (%d, %d)\n", t2->nrows, t2->ncols);
    tensor_print(t2);
    return 0;
}
