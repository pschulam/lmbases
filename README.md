# lmbases

`lmbases` is a library of functional bases that can be used to make linear models non-linear in their arguments while maintaining linear dependence on model parameters. Currently, the library only contains B-spline bases, which are relatively flexible.

## Usage

Basis classes define a property `dimension` and a method `design`. To get the number of basis functions in the basis, simply retrieve the `dimension` property from a class. To compute a design matrix for a sequence of observations `x`, pass them to `design`, which will return a numpy array with `len(x)` rows and `basis_class.dimension` columns. You can use this matrix as the design matrix (feature matrix in machine learning) for a linear model.



