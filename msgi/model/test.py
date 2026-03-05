import numpy 

def test_func():
    assert numpy.array_equal(numpy.array([1, 2, 3]), numpy.array([1, 2, 3]))
    return "Test passed!"