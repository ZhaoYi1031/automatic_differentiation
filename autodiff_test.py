import autodiff as ad
import numpy as np
import copy

def f(x):
    x1 = x[0]
    x2 = x[1]
    x3 = x[2]
    y  = (np.sin(x1+1) + np.cos(2*x2)) * np.tan(np.log(x3)) + (np.sin(x2+1)) + np.cos(2*x1) * np.exp(1+np.sin(x3))
    return y

def change_x(x, i, val):
    y = copy.deepcopy(x)
    y[i] += val
    return y

def numerical_diff(f, x0, epilson):
    x0 = np.array(x0).flatten()
    n = len(x0)
    ep = epilson * np.ones(n)
    df = np.zeros_like(x0, dtype=np.float)
    for i in range(n):
        x_plus = change_x(x0, i, epilson)  # [i] += 100#epilson
        x_minus = change_x(x0, i, -epilson)  # [i] -= 100#epilson
        f_plus = f(x_plus)
        f_minus = f(x_minus)
        df[i] = (f_plus - f_minus) / (2*epilson)
    return f(x0), df

def test():
    x1 = ad.Variable(name = "x1")
    x2 = ad.Variable(name = "x2")
    x3 = ad.Variable(name = "x3")

    y = (ad.sin_op(x1 + 1) + ad.cos_op(2 * x2)) * ad.tan_op(ad.log_op(x3)) + (ad.sin_op(x2 + 1)) + ad.cos_op(2*x1) * ad.exp_op(1+ad.sin_op(x3))
    grad_x1, grad_x2, grad_x3 = ad.gradients(y, [x1, x2, x3])

    executor = ad.Executor([y, grad_x1, grad_x2, grad_x3])
    x1_val = 1 * np.ones(1)
    x2_val = 2 * np.ones(1)
    x3_val = 3 * np.ones(1)
    y_val, grad_x1_val, grad_x2_val, grad_x3_val = executor.run(feed_dict = {x1: x1_val, x2 : x2_val, x3: x3_val})

    print('x1=', x1_val[0])
    print('x2=', x2_val[0])
    print('x3=', x3_val[0])
    print('---------------------------------------------------------------')

    print('y0_val=', y_val[0])
    print('grad_x1_val= ', grad_x1_val[0])
    print('grad_x2_val= ', grad_x2_val[0])
    print('grad_x3_val= ', grad_x3_val[0])
    print('---------------------------------------------------------------')
    y_numerical, grad_numerical = numerical_diff(f, [x1_val, x2_val, x3_val], 1e-10)
    print('y0_numerical= ', y_numerical)
    grad_numerical_x1, grad_numerical_x2, grad_numerical_x3 = grad_numerical[0],grad_numerical[1], grad_numerical[2]
    print('grad_numerical_x1 =', grad_numerical_x1)
    print('grad_numerical_x2 =', grad_numerical_x2)
    print('grad_numerical_x3 =', grad_numerical_x3)
    print('---------------------------------------------------------------')
    print('gradients Offset:')
    print('x1:', abs(grad_x1_val - grad_numerical_x1))
    assert abs(grad_x1_val - grad_numerical_x1) < 1e-5
    print('x2:', abs(grad_x2_val - grad_numerical_x2))
    assert abs(grad_x2_val - grad_numerical_x2) < 1e-5
    print('x3:', abs(grad_x3_val - grad_numerical_x3))
    assert abs(grad_x3_val - grad_numerical_x3) < 1e-5


if __name__ == "__main__":
    test()
