import pytest
import torch
import tensor1d

def assert_tensor_equal(torch_tensor, tensor1d_tensor):
    assert torch_tensor.tolist() == tensor1d_tensor.tolist()

@pytest.mark.parametrize("size", [0, 1, 10, 100])
def test_arange(size):
    torch_tensor = torch.arange(size, dtype=torch.float32)
    tensor1d_tensor = tensor1d.arange(size)
    assert_tensor_equal(torch_tensor, tensor1d_tensor)

@pytest.mark.parametrize("case", [[], [1], [1, 2, 3], list(range(100))])
def test_tensor_creation(case):
    torch_tensor = torch.tensor(case)
    tensor1d_tensor = tensor1d.tensor(case)
    assert_tensor_equal(torch_tensor, tensor1d_tensor)

@pytest.mark.parametrize("size", [0, 1, 10, 100])
def test_empty(size):
    torch_tensor = torch.empty(size)
    tensor1d_tensor = tensor1d.empty(size)
    assert len(torch_tensor) == len(tensor1d_tensor)

@pytest.mark.parametrize("index", range(1, 10))
def test_indexing(index):
    torch_tensor = torch.arange(10, dtype=torch.float32)
    tensor1d_tensor = tensor1d.arange(10)
    assert torch_tensor[index].item() == tensor1d_tensor[index].item()

@pytest.mark.parametrize("slice_params", [
    (None, None, None),  # [:]
    (5, None, None),     # [5:]
    (None, 15, None),    # [:15]
    (5, 15, None),       # [5:15]
    (None, None, 2),     # [::2]
    (5, 15, 2),          # [5:15:2]
    (5, 15, 15),         # [5:15:15]
])
def test_slicing(slice_params):
    torch_tensor = torch.arange(20, dtype=torch.float32)
    tensor1d_tensor = tensor1d.arange(20)
    s = slice(*slice_params)
    assert_tensor_equal(torch_tensor[s], tensor1d_tensor[s])

def test_invalid_input():
    with pytest.raises(TypeError):
        tensor1d.tensor("not a valid input")

def test_invalid_index():
    t = tensor1d.arange(5)
    with pytest.raises(TypeError):
        t["invalid index"]

@pytest.mark.parametrize("initial_slice, second_slice", [
    ((5, 15, 1), (2, 7, 1)),     # Basic case
    ((5, 15, 1), (None, None, 1)),  # Full slice
    ((5, 15, 1), (None, None, 2)),  # Every other element
    ((5, 15, 2), (None, None, 2)),  # Every other of every other
    ((0, 20, 1), (-5, None, 1)),  # Negative start index
    ((0, 20, 1), (None, -5, 1)),  # Negative end index
    ((0, 20, 1), (-15, -5, 1)),  # Negative start and end indices
    ((5, 15, 1), (100, None, 1)),  # Start index out of range
    ((5, 15, 1), (None, 100, 1)),  # End index out of range
    ((5, 15, 1), (-100, None, 1)),  # Negative start index out of range
    ((5, 15, 1), (None, -100, 1)),  # Negative end index out of range
    ((0, 20, 1), (0, 0, 1)),  # Empty slice
    ((0, 0, 1), (None, None, 1)),  # Slice of empty slice
])
def test_slice_of_slice(initial_slice, second_slice):
    torch_tensor = torch.arange(20, dtype=torch.float32)
    tensor1d_tensor = tensor1d.arange(20)

    torch_slice = torch_tensor[slice(*initial_slice)]
    tensor1d_slice = tensor1d_tensor[slice(*initial_slice)]

    torch_result = torch_slice[slice(*second_slice)]
    tensor1d_result = tensor1d_slice[slice(*second_slice)]

    assert_tensor_equal(torch_result, tensor1d_result)

def test_multiple_slices():
    torch_tensor = torch.arange(100, dtype=torch.float32)
    tensor1d_tensor = tensor1d.arange(100)

    torch_result = torch_tensor[10:90:2][5:35:3][::2]
    tensor1d_result = tensor1d_tensor[10:90:2][5:35:3][::2]

    assert_tensor_equal(torch_result, tensor1d_result)

# Test for behavior with step sizes > 1
@pytest.mark.parametrize("step", [2, 3, 5])
def test_slices_with_steps(step):
    torch_tensor = torch.arange(50, dtype=torch.float32)
    tensor1d_tensor = tensor1d.arange(50)

    torch_result = torch_tensor[::step][5:20]
    tensor1d_result = tensor1d_tensor[::step][5:20]

    assert_tensor_equal(torch_result, tensor1d_result)

# Test for behavior with different slice sizes
@pytest.mark.parametrize("size", [10, 20, 50, 100])
def test_slices_with_different_sizes(size):
    torch_tensor = torch.arange(size, dtype=torch.float32)
    tensor1d_tensor = tensor1d.arange(size)

    torch_result = torch_tensor[size//4:3*size//4][::2]
    tensor1d_result = tensor1d_tensor[size//4:3*size//4][::2]

    assert_tensor_equal(torch_result, tensor1d_result)

# Test for behavior with overlapping slices
def test_overlapping_slices():
    torch_tensor = torch.arange(30, dtype=torch.float32)
    tensor1d_tensor = tensor1d.arange(30)

    torch_result = torch_tensor[5:25][3:15]
    tensor1d_result = tensor1d_tensor[5:25][3:15]

    assert_tensor_equal(torch_result, tensor1d_result)

# Test for behavior with adjacent slices
def test_adjacent_slices():
    torch_tensor = torch.arange(20, dtype=torch.float32)
    tensor1d_tensor = tensor1d.arange(20)

    torch_result = torch_tensor[5:15][0:10]
    tensor1d_result = tensor1d_tensor[5:15][0:10]

    assert_tensor_equal(torch_result, tensor1d_result)

# Test accessing elements, including negative indices
def test_getitem():
    torch_tensor = torch.arange(20, dtype=torch.float32)
    tensor1d_tensor = tensor1d.arange(20)
    assert torch_tensor[0].item() == tensor1d_tensor[0].item()
    assert torch_tensor[5].item() == tensor1d_tensor[5].item()
    assert torch_tensor[-1].item() == tensor1d_tensor[-1].item()
    assert torch_tensor[-5].item() == tensor1d_tensor[-5].item()

# Test setting elements, including negative indices
def test_setitem():
    torch_tensor = torch.arange(20, dtype=torch.float32)
    tensor1d_tensor = tensor1d.arange(20)

    torch_tensor[0] = 100
    tensor1d_tensor[0] = 100
    assert_tensor_equal(torch_tensor, tensor1d_tensor)

    torch_tensor[5] = 200
    tensor1d_tensor[5] = 200
    assert_tensor_equal(torch_tensor, tensor1d_tensor)

    torch_tensor[-1] = 300
    tensor1d_tensor[-1] = 300
    assert_tensor_equal(torch_tensor, tensor1d_tensor)

    torch_tensor[-5] = 400
    tensor1d_tensor[-5] = 400
    assert_tensor_equal(torch_tensor, tensor1d_tensor)

# Test setting elements indirectly (via a slice)
def test_setitem_indirect():
    torch_tensor = torch.arange(20, dtype=torch.float32)
    tensor1d_tensor = tensor1d.arange(20)
    torch_view = torch_tensor[5:15]
    tensor1d_view = tensor1d_tensor[5:15]

    torch_view[0] = 100
    tensor1d_view[0] = 100
    assert_tensor_equal(torch_tensor, tensor1d_tensor)

    torch_view[-1] = 200
    tensor1d_view[-1] = 200
    assert_tensor_equal(torch_tensor, tensor1d_tensor)

# test addition
def test_addition():

    # simple element-wise addition
    torch_tensor = torch.arange(20, dtype=torch.float32)
    tensor1d_tensor = tensor1d.arange(20)
    torch_result = torch_tensor + 5.0
    tensor1d_result = tensor1d_tensor + 5.0
    assert_tensor_equal(torch_result, tensor1d_result)

    # now test adding a float
    torch_result = torch_tensor + 6.0
    tensor1d_result = tensor1d_tensor + 6.0
    assert_tensor_equal(torch_result, tensor1d_result)

    # test broadcasting add with a 1-element tensor on the right
    torch_result = torch_tensor + torch.tensor([123.0])
    tensor1d_result = tensor1d_tensor + tensor1d.tensor([123.0])
    assert_tensor_equal(torch_result, tensor1d_result)

    # and on the left
    torch_result = torch.tensor([42.0]) + torch_tensor
    tensor1d_result = tensor1d.tensor([42.0]) + tensor1d_tensor
    assert_tensor_equal(torch_result, tensor1d_result)

    # and now test invalid cases
    with pytest.raises(TypeError):
        tensor1d_tensor + "not a valid input"

    with pytest.raises(TypeError):
        tensor1d_tensor + [1, 2, 3]

    with pytest.raises(ValueError):
        tensor1d_tensor + tensor1d.arange(5)
