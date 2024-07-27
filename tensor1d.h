/*
tensor1d.h
*/

#ifndef TENSOR1D_H
#define TENSOR1D_H

#include <stddef.h>
#include <stdbool.h>

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
int logical_to_physical(Tensor *t, int ix);
float tensor_getitem(Tensor* t, int ix);
Tensor* tensor_getitem_astensor(Tensor* t, int ix);
float tensor_item(Tensor* t);
void tensor_setitem(Tensor* t, int ix, float val);
Tensor* tensor_arange(int size);
char* tensor_to_string(Tensor* t);
void tensor_print(Tensor* t);
Tensor* tensor_slice(Tensor* t, int start, int end, int step);
void tensor_incref(Tensor* t);
void tensor_decref(Tensor* t);
void tensor_free(Tensor* t);
Tensor* tensor_add(Tensor* t1, Tensor* t2);

#endif // TENSOR1D_H
