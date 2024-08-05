import pytest
import torch
import numpy as np
import tensor2d


def assert_tensor_equal(torch_tensor, tensor2d_tensor):
    assert torch_tensor.tolist() == tensor2d_tensor.tolist()


@pytest.mark.parametrize("range_size,size", [(10, (5, 2)), (4, (2, 2))])
def test_arange(range_size, size):
    torch_tensor = torch.arange(range_size, dtype=torch.float32).reshape(
        (size[0], size[1])
    )
    tensor2d_tensor = tensor2d.arange(range_size).reshape((size[0], size[1]))
    assert_tensor_equal(torch_tensor, tensor2d_tensor)


@pytest.mark.parametrize("case", [[[1, 2], [2, 3]], [[]], [list(range(10))]])
def test_tensor_creation(case):
    torch_tensor = torch.tensor(case)
    tensor2d_tensor = tensor2d.tensor(case)
    assert_tensor_equal(torch_tensor, tensor2d_tensor)


@pytest.mark.parametrize("size", [(2, 2), (10, 1), (1, 100)])
def test_empty(size):
    torch_tensor = torch.empty(size)
    tensor2d_tensor = tensor2d.empty(size)
    assert_tensor_equal(torch_tensor, tensor2d_tensor)


@pytest.mark.parametrize("index", [(0, 5), (0, 9), (0, 0)])
def test_indexing(index):
    torch_tensor = torch.arange(10, dtype=torch.float32).reshape((1, 10))
    tensor2d_tensor = tensor2d.arange(10)
    assert torch_tensor[index].item() == tensor2d_tensor[index]


@pytest.mark.parametrize(
    "slice_params",
    [
        [(None, None, None), (None, None, None)],  # [:, :]
        [(1, None, None), (None, None, None)],  # [5:, :]
        [(None, None, None), (0, 2, 1)],  # [:, 0:2:1]
        [(1, 10, 2), (0, 1, None)],  # [1:10:2, 0:1:1]
        [(1, -2, 2), (0, 1, None)],  # [1:-2:2, 0:1:1]
    ],
)
def test_slicing(slice_params):
    torch_tensor = torch.arange(20, dtype=torch.float32).reshape((10, 2))
    tensor2d_tensor = tensor2d.arange(20).reshape((10, 2))
    tmp_slice_params = []
    for s in slice_params:
        tmp_slice_params.append(slice(*s))
    assert_tensor_equal(
        torch_tensor[tuple(tmp_slice_params)], tensor2d_tensor[tuple(tmp_slice_params)]
    )


def test_invalid_input():
    with pytest.raises(TypeError):
        tensor2d.tensor("not a valid input")


def test_invalid_index():
    t = tensor2d.arange(5)
    with pytest.raises(TypeError):
        t["invalid index"]


@pytest.mark.parametrize(
    "initial_slice, second_slice",
    [
        ([(0, 2, 1), (None, None, 1)], [(None, None, 1), (1, 3, 1)]),  # Basic case
        (
            [(0, 4, 1), (None, None, 1)],
            [(None, None, 2), (None, None, 1)],
        ),  # Every other element
        (
            [(-1, 0, 1), (None, None, 1)],
            [(-1, -5, 2), (None, None, 1)],
        ),  # Negative index
        (
            [(0, 10, 1), (None, None, 1)],
            [(20, None, 2), (None, None, 1)],
        ),  # Out of range index
        ([(0, 5, 1), (None, None, 1)], [(0, 0, 1), (None, None, 1)]),  # Empty slice
        (
            [(0, 0, 1), (None, None, 1)],
            [(None, None, 1), (None, None, 1)],
        ),  # Slice of empty slice
    ],
)
def test_slice_of_slice(initial_slice, second_slice):
    torch_tensor = torch.arange(20, dtype=torch.float32).reshape((5, 4))
    tensor2d_tensor = tensor2d.arange(20).reshape((5, 4))

    tmp_slice_store = []
    for s in initial_slice:
        tmp_slice_store.append(slice(*s))

    torch_slice = torch_tensor[tuple(tmp_slice_store)]
    tensor2d_slice = tensor2d_tensor[tuple(tmp_slice_store)]

    tmp_slice_store = []
    for s in second_slice:
        tmp_slice_store.append(slice(*s))
    torch_result = torch_slice[tuple(tmp_slice_store)]
    tensor2d_result = tensor2d_slice[tuple(tmp_slice_store)]

    assert_tensor_equal(torch_result, tensor2d_result)


def test_multiple_slices():
    torch_tensor = torch.arange(100, dtype=torch.float32).reshape((10, 10))
    tensor2d_tensor = tensor2d.arange(100).reshape((10, 10))

    torch_result = torch_tensor[0:8:2, 1:5:1][1:6:2, 0:3:1]
    tensor2d_result = tensor2d_tensor[0:8:2, 1:5:1][1:6:2, 0:3:1]

    assert_tensor_equal(torch_result, tensor2d_result)


def test_getitem():
    torch_tensor = torch.arange(20, dtype=torch.float32).reshape((5, 4))
    tensor2d_tensor = tensor2d.arange(20).reshape((5, 4))
    assert torch_tensor[0, 0].item() == tensor2d_tensor[0, 0]
    assert torch_tensor[1, 2].item() == tensor2d_tensor[1, 2]
    assert torch_tensor[4, 3].item() == tensor2d_tensor[4, 3]


def test_setitem():
    torch_tensor = torch.arange(20, dtype=torch.float32).reshape((5, 4))
    tensor2d_tensor = tensor2d.arange(20).reshape((5, 4))

    torch_tensor[0, 0] = 10
    tensor2d_tensor[0, 0] = 10
    assert_tensor_equal(torch_tensor, tensor2d_tensor)

    torch_tensor[-1, -1] = 40
    tensor2d_tensor[-1, -1] = 40
    assert_tensor_equal(torch_tensor, tensor2d_tensor)


def test_addition():
    # scalar addition
    torch_tensor = torch.arange(20, dtype=torch.float32).reshape((5, 4))
    tensor2d_tensor = tensor2d.arange(20).reshape((5, 4))
    torch_result = torch_tensor + 5.0
    tensor2d_result = tensor2d_tensor + 5.0
    assert_tensor_equal(torch_result, tensor2d_result)

    # tensor addition
    torch_result = torch_tensor + torch_tensor 
    tensor2d_result = tensor2d_tensor + tensor2d_tensor 
    assert_tensor_equal(torch_result, tensor2d_result)

    # invalid cases
    with pytest.raises(TypeError):
        tensor2d_tensor + "not a valid input"

    with pytest.raises(TypeError):
        tensor2d_tensor + [1, 2, 3]

    with pytest.raises(ValueError):
        tensor2d_tensor + tensor2d.arange(5).reshape((1, 5))


def test_multiplication():
    # scalar multiplication
    torch_tensor = torch.arange(20, dtype=torch.float32).reshape((5, 4))
    tensor2d_tensor = tensor2d.arange(20).reshape((5, 4))
    torch_result = torch_tensor * 5.0
    tensor2d_result = tensor2d_tensor * 5.0
    assert_tensor_equal(torch_result, tensor2d_result)

    # tensor element-wise product
    torch_result = torch_tensor * torch_tensor 
    tensor2d_result = tensor2d_tensor * tensor2d_tensor 
    assert_tensor_equal(torch_result, tensor2d_result)

    # invalid cases
    with pytest.raises(TypeError):
        tensor2d_tensor * "not a valid input"

    with pytest.raises(TypeError):
        tensor2d_tensor * [1, 2, 3]

    with pytest.raises(ValueError):
        tensor2d_tensor * tensor2d.arange(5).reshape((1, 5))

def test_dot_product():
    torch_tensor1 = torch.arange(20, dtype=torch.float32).reshape((5,4))
    torch_tensor2 = torch.arange(20, dtype=torch.float32).reshape((4, 5))
    tensor2d_tensor1 = tensor2d.arange(20).reshape((5, 4))
    tensor2d_tensor2 = tensor2d.arange(20).reshape((4, 5))

    torch_result = torch_tensor1.matmul(torch_tensor2) 
    tensor2d_result = tensor2d_tensor1.dot(tensor2d_tensor2)
    assert_tensor_equal(torch_result, tensor2d_result)
    # invalid cases
    with pytest.raises(ValueError):
        tensor2d_tensor1.dot(tensor2d_tensor2.reshape((5, 4)))
