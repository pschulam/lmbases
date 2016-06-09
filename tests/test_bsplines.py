import numpy as np

import lmbases


def test_against_r_splines_uniform():
    '''Compare BSplines class against R's bsplines with uniform knots.

    Generate the ground truth with the following R commands:

    > library(splines)
    > x <- c(1.5, 3.3, 5.1, 7.2, 9.9)
    > k <- c(2.5, 5.0, 7.5)
    > b <- bs(x, knots=k, degree=2, intercept=TRUE, Boundary.knots=c(0, 10))
    > print(b)
            1      2      3      4      5      6
    [1,] 0.16 0.6600 0.1800 0.0000 0.0000 0.0000
    [2,] 0.00 0.2312 0.7176 0.0512 0.0000 0.0000
    [3,] 0.00 0.0000 0.4608 0.5384 0.0008 0.0000
    [4,] 0.00 0.0000 0.0072 0.6056 0.3872 0.0000
    [5,] 0.00 0.0000 0.0000 0.0008 0.0776 0.9216
    attr(,"degree")
    [1] 2
    attr(,"knots")
    [1] 2.5 5.0 7.5
    attr(,"Boundary.knots")
    [1]  0 10
    attr(,"intercept")
    [1] TRUE
    attr(,"class")
    [1] "bs"     "basis"  "matrix"

    '''
    b_ground_truth = np.array(
        [[0.16, 0.6600, 0.1800, 0.0000, 0.0000, 0.0000],
         [0.00, 0.2312, 0.7176, 0.0512, 0.0000, 0.0000],
         [0.00, 0.0000, 0.4608, 0.5384, 0.0008, 0.0000],
         [0.00, 0.0000, 0.0072, 0.6056, 0.3872, 0.0000],
         [0.00, 0.0000, 0.0000, 0.0008, 0.0776, 0.9216]])

    x = np.array([1.5, 3.3, 5.1, 7.2, 9.9])
    bs = lmbases.BSplines(low=0.0, high=10.0, num_bases=6, degree=2)
    assert np.allclose(bs.design(x), b_ground_truth)


def test_against_r_splines_quantiles():
    '''Compare BSplines class against R's bsplines with quantile knots.

    Generate the ground truth with the following R commands:

    > library(splines)
    > x <- c(1.5, 3.3, 5.1, 7.2, 9.9)
    > b <- bs(x, degree=2, df=6, intercept=TRUE, Boundary.knots=c(0, 10))
    > print(b)
                 1         2         3           4         5         6
    [1,] 0.2975207 0.5687895 0.1336898 0.000000000 0.0000000 0.0000000
    [2,] 0.0000000 0.3529412 0.6470588 0.000000000 0.0000000 0.0000000
    [3,] 0.0000000 0.0000000 0.5384615 0.461538462 0.0000000 0.0000000
    [4,] 0.0000000 0.0000000 0.0000000 0.571428571 0.4285714 0.0000000
    [5,] 0.0000000 0.0000000 0.0000000 0.000728863 0.0694242 0.9298469
    attr(,"degree")
    [1] 2
    attr(,"knots")
    25% 50% 75% 
    3.3 5.1 7.2 
    attr(,"Boundary.knots")
    [1]  0 10
    attr(,"intercept")
    [1] TRUE
    attr(,"class")
    [1] "bs"     "basis"  "matrix"
        
    '''
    b_ground_truth = np.array(
        [[0.2975207, 0.5687895, 0.1336898, 0.000000000, 0.0000000, 0.0000000],
         [0.0000000, 0.3529412, 0.6470588, 0.000000000, 0.0000000, 0.0000000],
         [0.0000000, 0.0000000, 0.5384615, 0.461538462, 0.0000000, 0.0000000],
         [0.0000000, 0.0000000, 0.0000000, 0.571428571, 0.4285714, 0.0000000],
         [0.0000000, 0.0000000, 0.0000000, 0.000728863, 0.0694242, 0.9298469]])

    x = np.array([1.5, 3.3, 5.1, 7.2, 9.9])    
    bs = lmbases.BSplines(low=0.0, high=10.0, num_bases=6, degree=2, x=x)
    assert np.allclose(bs.design(x), b_ground_truth)
