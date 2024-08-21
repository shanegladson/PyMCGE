# PyMCGE - Monte Carlo Gradient Estimation in Python

PyMCGE is a package that leverages Monte Carlo methods to estimate gradients in objective functions of the form

$$ \mathcal{F}(\theta) = \int p(x|\theta)f(x|\phi) dx $$

where $p$ is a probability distribution with unknown parameters $\theta$ to be determined, and $f$ is a cost function with *fixed structurual* parameters $\phi$. 

## Installation
To install this package, simply clone the repository and run
```pip install -r requirements.txt```
to install all necessary dependencies. After this, you are all set to go!

If you are unsure that the package has been installed correctly, run
```./run_tests.sh```
in the terminal. This will run all necessary unit tests and type checking. If this passes, then the package has been correctly installed.

## Usage
The basic theory implemented in this library is described in the file Report.pdf, which is a paper I wrote for a graduate class I recently took. This paper was based off of *Monte Carlo Gradient Estimation in Machine Learning* by Mnih et. al. (2020). There is currently support for estimation via the score gradient method, and development of measure-valued gradients is in progress. Requests for new features or methods should be directed to the repository author.

## Contributing
If you are interested in contributing to the repository, feel free to create a branch from any of the existing issues and check out a branch to begin work. All pull requests must be validated by the repository owner Shane Gladson. At the bare minimum, pull requests should have extensive type hints and unit testing, and the bash script **run_tests.sh** should pass as well. For any further questions, feel free to reach out to shanegladson@gmail.com.

