#ifndef TENSOR2D_H
#define TENSOR2D_H

#include <stdbool.h>
#include <stddef.h>

typedef struct {
    float *data;
    int data_size;
    int ref_count;
} Storage;

// The equivalent of tensor in PyTorch
typedef struct {
    Storage *storage;
    int offset;
    int size;
    int nrows;
    int ncols;
    int stride[2];
    char *repr; // holds the text representation of the tensor
} Tensor;

Tensor *tensor_empty(int nrows, int ncols);
Tensor *reshape(Tensor *t, int nrows, int ncols);
float tensor_getitem(Tensor *t, int row, int col);
void tensor_setitem(Tensor *t, int row, int col, float val);
char *tensor_to_string(Tensor *t);
Tensor *tensor_addf(Tensor *t, float val);
Tensor *tensor_add(Tensor *t1, Tensor *t2);
Tensor *tensor_dot(Tensor *t1, Tensor *t2);
Tensor *reshape(Tensor *t, int nrows, int ncols);
#endif

