

## 1. Project Overview

The purpose of this project is to apply mathematical and computational methods to solve different types of problems in data science and machine learning.
All implementations are written in pure Python using standard scientific libraries.

| Section | Topic                               | Description                                                                                            |
| ------- | ----------------------------------- | ------------------------------------------------------------------------------------------------------ |
| **1**   | PageRank Algorithm                  | Calculates web-page importance using eigenvector and power iteration methods.                          |
| **2**   | Principal Component Analysis (PCA)  | Performs dimensionality reduction on standardized height–weight data and visualizes the 1D projection. |
| **3**   | Linear Regression via Least Squares | Predicts house prices from square footage, bedrooms, and age using least squares regression.           |
| **4**   | Gradient Descent Optimization       | Minimizes a quadratic loss function f(X) = ½ Σ(X − A)² using gradient descent and plots convergence.   |

---



## 2. Dependencies and Installation

### Required Libraries

Make sure you have the following Python packages installed:

* `numpy`
* `scipy`
* `matplotlib`
* `pandas`

### Installation Steps

1. Clone the repository to your local machine:

```bash
git clone <your_repository_link>
cd <repository_name>
```

2. Install dependencies (using pip):

```bash
pip install numpy scipy matplotlib pandas
```

3. Ensure that `data.csv` is present in the same directory as `project3.py`.
   It should contain two columns named **Height** and **Weight** with standardized values.

---

## 4. How to Run the Code

You can run the entire project from the command line:

```bash
python project3.py
```

The script will:

* Print outputs for each question directly in the terminal.
* Generate plots for PCA and gradient descent sections.
* Stop automatically when each step finishes.

If you want to run individual sections, you can comment out other parts of the code.

---

## 5. Detailed Description of Each Part

### 1. PageRank Algorithm

Implements Google’s original PageRank concept to find the relative importance of web pages.
Two methods are implemented:

* **Eigenvector Method** – Finds the dominant eigenvector for eigenvalue 1 using `scipy.linalg.eig`.
* **Power Iteration** – Iteratively updates the rank vector until convergence.

Both methods yield the same steady-state rank distribution, normalized to sum to 1.

---

### 2. Principal Component Analysis (PCA)

This part reduces two-dimensional (height and weight) data into a single dimension while preserving maximum variance.

* Computes the **covariance matrix** using `numpy.cov`.
* Performs **eigen-decomposition** with `scipy.linalg.eigh`.
* Projects the data onto the first principal component.
* Visualizes both the original data and its 1D projection.

Output plots:

1. Original standardized data (Height vs Weight)
2. Projected data onto the first principal component

---

### 3. Linear Regression via Least Squares

Predicts house prices using a small dataset of features: square footage, bedrooms, and age.

* Solves the least-squares problem ( X β = y ) with `scipy.linalg.lstsq`.
* Predicts the price of a new house with features [2400, 3, 20].
* Compares results to the closed-form normal equation solution.

This section illustrates how linear regression can be computed using basic matrix operations.

---

### 4. Gradient Descent Optimization

Demonstrates numerical optimization on a quadratic loss function ( f(X) = ½ ∑ (X − A)² ).

* Initializes two random matrices `A` and `X`.
* Defines the loss and gradient functions.
* Uses `scipy.optimize.minimize` with the L-BFGS-B algorithm to minimize the loss.
* Tracks loss values over iterations and plots the convergence curve.

Output:

* A plot showing the loss value decreasing with each iteration until convergence.

---

## 6. Results and Discussion

* **PageRank** successfully finds a stable ranking vector that represents each page’s relative importance.
* **PCA** shows that most of the variance in the height–weight data can be captured by a single principal component.
* **Linear Regression** produces reasonable coefficients and accurate predictions for the new sample.
* **Gradient Descent** converges smoothly as the loss decreases, demonstrating effective optimization.
