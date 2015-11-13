# SEP-QN: A Scalable and Extensible Proximal Quasi-Newton Method

SEP-QN is a MATLAB package that based on [PQN](https://github.com/yuekai/pnopt), which uses proximal Newton-type methods to minimize composite functions. SEP-QN extends the simple 
regularizer to a hybrid regularizer by employing Quasi-LBFGS and SCD.

## Usage

SEP-QN has the calling sequence:

    [ w, f, output ] = pnopt( smoothF, nonsmoothF_scalar_matrix, dual_proxF, primal_init, dual_init, nonsmoothF, strongly_convex_parameter, stop_critical)

The required input arguments are:
* `smoothF`: a smooth function,
* `nonsmoothF_scalar_matrix`: that is W and b in \Phi (W x + b),
* `dual_proxF`: dual proximal function of each nonsmooth function,
* `primal_init`:  a starting point for the solver,
* `dual_init`: a starting point for the sub solver.
* `nonsmoothF`: a array of nonsmooth functions,
* `strongly_convex_parameter`: Strongly convex parameter for Quasi-LBFGS, will be adaptively updated.

The user can also supply an `options` structure created using `pnopt_optimset` to customize the behavior of PNOPT. `pnopt_optimset` shares a similar interface with MATLAB's `optimset` function:

  options = pnopt_optimset( 'param1', val1, 'param2', val2, ... );

Calling `pnopt_optimset` with no inputs and outputs prints available options.

pnopt returns:
* `x`: an optimal solution,
* `f`: the optimal value,
* `output`: a structure containing information collected during the execution of PNOPT.

### Creating smooth and nonsmooth functions

Smooth and nonsmooth functions must satisfy these conventions:

* `smoothF(x)` should return the function value and gradient at `x`, i.e. `[ fx, gradx ] = smoothF(x)`,
* `nonsmoothF(x)` should return function value at `x`, i.e. `f_x = nonsmoothF(x)`,
* `dual_proxF(x,t)` should return the proximal point `y` and the function value at `y`, i.e. `[ f_y, y ] = dual_proxF(x,t )`.

SEP-QN is compatible with the function generators included with TFOCS that accept vector arguments so users can use these generators to create commonly used smooth and nonsmooth functions.
Please refer to section 3 of the [TFOCS user guide](https://github.com/cvxr/TFOCS/raw/master/userguide.pdf) for details.

## Demo: sparse fused logistic regression

In File: [fused_sparse_demo.m](https://github.com/NewtonOptimization/SEP-QN/blob/master/fused_sparse_demo.m)

