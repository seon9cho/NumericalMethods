# NumericalMethods

My solutions to the numerical methods labs from BYU's Applied and Computational Mathematics Emphasis (ACME), drawn from *Volume 1* and *Volume 2* of the lab manuals. Each lab is a notebook that pairs the spec's problem statements with my solutions and their output, plus condensed notes on the concepts the lab covers.

| Lab | Volume | Notebook |
|-----|--------|----------|
| Differentiation | Vol. 1, Lab 8 | [Differentiation/differentiation.ipynb](Differentiation/differentiation.ipynb) |
| Conditioning and Stability | Vol. 1, Lab 10 | [Conditioning_Stability/conditioning_stability.ipynb](Conditioning_Stability/conditioning_stability.ipynb) |
| Monte Carlo Integration | Vol. 1, Lab 11 | [MonteCarloIntegration/montecarlo_integration.ipynb](MonteCarloIntegration/montecarlo_integration.ipynb) |
| The Drazin Inverse | Vol. 1, Lab 14 | [DrazinInverse/drazin.ipynb](DrazinInverse/drazin.ipynb) |
| Iterative Solvers | Vol. 1, Lab 15 | [IterativeSolvers/iterative_solvers.ipynb](IterativeSolvers/iterative_solvers.ipynb) |
| Polynomial Interpolation | Vol. 2, Lab 9 | [PolynomialInterpolation/polynomial_interpolation.ipynb](PolynomialInterpolation/polynomial_interpolation.ipynb) |

## Running the notebooks

Each notebook runs from its own folder. The data each lab uses sits next to its notebook. The notebooks need NumPy, SciPy, SymPy and Matplotlib, and the Differentiation lab also needs [autograd](https://github.com/HIPS/autograd) (`pip install autograd`).

## Credits

The lab specifications come from the [Foundations of Applied Mathematics](https://foundations-of-applied-mathematics.github.io/) lab manuals, © Dr. J. Humpherys, licensed under [CC BY 3.0 US](http://creativecommons.org/licenses/by/3.0/us/). The data files come from the project's official [data repository](https://github.com/Foundations-of-Applied-Mathematics/Data). Problem statements are quoted from the specs; the solutions are my own.
