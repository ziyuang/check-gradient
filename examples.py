#!/usr/bin/env python3

from check_gradient import *
import logging

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)
_logger.addHandler(logging.StreamHandler())


def _run_decorator_examples():
    @check_gradient(logger=_logger)
    def correct_func(x, alpha):
        return x ** alpha, alpha * x ** (alpha - 1)

    @check_gradient(logger=_logger, verbose=True)
    def incorrect_func(x, alpha):
        return x ** alpha, x ** (alpha + 1) / (alpha + 1)  # Oops!

    @check_gradient(logger=_logger)
    def correct_func_vec_in(x, w):
        x = np.asarray(x)
        w = np.asarray(w)
        obj = np.sum(w * (x ** 2)) ** 0.5
        return obj, w * x / obj

    @check_gradient(logger=_logger, verbose=True)
    def incorrect_func_vec_in(x, w):
        x = np.asarray(x)
        w = np.asarray(w)
        obj = np.linalg.norm(w * x, 2)
        return obj, w * x / obj  # the gradient should be (w ** 2) * x / obj

    _logger.info("Checking correct_func")
    correct_func(x=2, alpha=2)

    _logger.info("\nChecking incorrect_func")
    incorrect_func(x=2, alpha=2)

    _logger.info("\nChecking correct_func_vec_in")
    correct_func_vec_in(x=[1, 1, 1], w=[1, 2, 3])

    _logger.info("\nChecking incorrect_func_vec_in")
    incorrect_func_vec_in(x=[1, 1, 1], w=[1, 2, 3])


def _run_function_examples():
    def correct_func_obj(x, alpha):
        return x ** alpha

    def correct_func_grad(x, alpha):
        return alpha * x ** (alpha - 1)

    def incorrect_func_obj(x, alpha):
        return x ** alpha

    def incorrect_func_grad(x, alpha):
        return x ** (alpha + 1) / (alpha + 1)  # Oops!

    def correct_func_vec_in_obj(x, w):
        x = np.asarray(x)
        w = np.asarray(w)
        return np.sum(w * (x ** 2)) ** 0.5

    def correct_func_vec_in_grad(x, w):
        x = np.asarray(x)
        w = np.asarray(w)
        obj = np.sum(w * (x ** 2)) ** 0.5
        return w * x / obj

    def incorrect_func_vec_in_obj(x, w):
        x = np.asarray(x)
        w = np.asarray(w)
        return np.linalg.norm(w * x, 2)

    def incorrect_func_vec_in_grad(x, w):
        x = np.asarray(x)
        w = np.asarray(w)
        obj = np.linalg.norm(w * x, 2)
        return w * x / obj  # the gradient should be (w ** 2) * x / obj

    objs = [correct_func_obj, incorrect_func_obj, correct_func_vec_in_obj, incorrect_func_vec_in_obj]
    grads = [correct_func_grad, incorrect_func_grad, correct_func_vec_in_grad, incorrect_func_vec_in_grad]
    xs = [2, 2, [1, 1, 1], [1, 1, 1]]
    args_vec = [(2,), (2,), ([1, 2, 3],), ([1, 2, 3],)]

    for obj, grad, x, args in zip(objs, grads, xs, args_vec):
        obj_name = obj.__name__
        grad_name = grad.__name__
        _logger.info("\nChecking %s and %s" % (obj_name, grad_name))
        verbose = obj_name.startswith("incorrect")
        correct = is_gradient_correct(obj, grad, x, args, logger=_logger, verbose=verbose)
        if correct:
            _logger.info("%s is correct" % grad_name)
        else:
            _logger.info("%s is incorrect" % grad_name)


def run_examples():
    _run_decorator_examples()
    _run_function_examples()


if __name__ == "__main__":
    run_examples()
