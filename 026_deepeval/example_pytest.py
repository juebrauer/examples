def f(x,y):
    result = x
    result += y
    return result

def test_f():
    assert f(0,0) == 0
    assert f(3,7) == 10
    assert f(-5,-9) == -14
    assert f(-5,10) == 5