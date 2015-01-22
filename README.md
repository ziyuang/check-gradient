# Check gradient
A helper for checking the correctness of the analytical gradient of a function by comparing it with the empirical (numerical) gradient.

## Usage

### Via the decorator
Suppose the target function `f` accepts `x` as the first argument, and return a tuple consisting of the objective value and the gradient at `x`. In the simplest case,

  ```python
  @check_gradient()
  def f(x, ...):
        ...
  ```
    
will check the gradient of `f` at `x` and print the comparison result when `f` is called.

#### Optional arguments

* `eps`: the step for calculating the empirical gradient (default: 1e-7)
* `comp_grads_func`: the function for comparing the two gradients and show the information, a default implementation is provided
    * signature: `comp_grads_func(analytical_gradient, empirical_gradient, logger=default_logger, verbose=False)`
    * optional arguments:
        * `logger`: the logger for the comparison message (default: a logger with `NullHandler()` as the handler)
        * `verbose`: when set to `True`, print the contents of the two gradients.
* `**comp_grads_func_kwargs`: `comp_grads_func`'s optional arguments


### Via the function
If you have two separated functions `obj` and `grad` with the same signature and `x` as their first argument, respectively for the value and the gradient, `is_gradient_correct` will be the helper to check the correctness of the gradient. In the simplest case, 

    is_gradient_correct(obj, grad, x, ...)
    
will do the same job. Optional arguments include `args` and `kwargs`, serving as the `*args` and `**kwargs` for `obj` and `grad`. Other optional arguments are the same as the decorator's.

## Examples

See `examples.py`.
