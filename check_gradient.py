import numpy as np
import logging
import sys
from functools import wraps

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)
_logger.addHandler(logging.NullHandler())


def _obj_plus_minus_tuple(f, x, eps, *args, **kwargs):
    x = np.asarray(x)
    obj_plus = np.empty(x.shape)
    obj_minus = np.empty(x.shape)
    for idx in np.ndindex(x.shape):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[idx] += eps
        x_minus[idx] -= eps

        op = f(x_plus, *args, **kwargs)
        if isinstance(op, tuple):
            assert len(op) == 2
            op = op[0]
        obj_plus[idx] = op

        om = f(x_minus, *args, **kwargs)
        if isinstance(om, tuple):
            assert len(om) == 2
            om = om[0]
        obj_minus[idx] = om
    return obj_plus, obj_minus


# the default function for showing both analytical gradient and empirical gradient
def _comp_grads_func_default(a_grad, e_grad, logger=_logger, verbose=False):
    if np.isscalar(a_grad):
        assert np.isscalar(e_grad), \
            "The analytical gradient is scalar but the empirical gradient is not."
    else:
        a_grad = np.asarray(a_grad)
        e_grad = np.asarray(e_grad)
        assert a_grad.shape == e_grad.shape, \
            "The shapes of the analytical gradient and the empirical gradient don't match."
    if np.allclose(a_grad, e_grad):
        sys.stdout.flush()
        logger.info("Gradient checking passed.")
        sys.stderr.flush()
        return True
    else:
        sys.stdout.flush()
        logger.warning("Gradient checking didn't pass.")
        if verbose:
            logger.warning("Analytical gradient = %s" % repr(a_grad))
            logger.warning("Empirical gradient = %s" % repr(e_grad))
        sys.stderr.flush()
        return False


def check_gradient(eps=1e-7, comp_grads_func=_comp_grads_func_default, **comp_grads_func_kwargs):
    def specialized_decorator(f):
        # f is expected to return a tuple of (objective value, gradient)
        @wraps(f)
        def wrapper(x, *args, **kwargs):
            obj, a_grad = f(x, *args, **kwargs)
            assert np.isscalar(obj), \
                "The objective value of %s should be scalar." % f.__name__
            if np.isscalar(x):
                assert np.isscalar(a_grad), \
                    "x is scalar but the analytical gradient is not."
                obj_plus, _ = f(x + eps, *args, **kwargs)
                obj_minus, _ = f(x - eps, *args, **kwargs)
            else:
                x = np.asarray(x, dtype=float)
                a_grad = np.asarray(a_grad, dtype=float)
                assert x.shape == a_grad.shape, \
                    "The shapes of x and the analytical gradient don't match."
                obj_plus, obj_minus = _obj_plus_minus_tuple(f, x, eps, *args, **kwargs)
            e_grad = (obj_plus - obj_minus) / (2 * eps)
            comp_grads_func(a_grad, e_grad, **comp_grads_func_kwargs)
            return obj, a_grad

        return wrapper

    return specialized_decorator


def is_gradient_correct(obj_func, grad_func, x, args=(), kwargs=None,
                        eps=1e-7, comp_grads_func=_comp_grads_func_default, **comp_grads_func_kwargs):
    if kwargs is None:
        kwargs = {}
    a_grad = grad_func(x, *args, **kwargs)
    if np.isscalar(x):
        assert np.isscalar(a_grad), \
            "x is scalar but the analytical gradient is not."
        obj_plus = obj_func(x + eps, *args, **kwargs)
        obj_minus = obj_func(x - eps, *args, **kwargs)
    else:
        x = np.asarray(x, dtype=float)
        a_grad = np.asarray(a_grad, dtype=float)
        assert x.shape == a_grad.shape, \
            "The shapes of x and the analytical gradient don't match."
        obj_plus, obj_minus = _obj_plus_minus_tuple(obj_func, x, eps, *args, **kwargs)
    e_grad = (obj_plus - obj_minus) / (2 * eps)
    return comp_grads_func(a_grad, e_grad, **comp_grads_func_kwargs)