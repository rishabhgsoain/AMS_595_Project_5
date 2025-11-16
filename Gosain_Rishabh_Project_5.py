import numpy as np
from scipy.linalg import eig, eigh, lstsq# type: ignore
from scipy.optimize import minimize # type: ignore
import matplotlib.pyplot as plt
import pandas as pd# type: ignore


#  The PageRank Algorithm     
#  The objective is to calculate page significance ratings in two ways:
#  Eigenvalue 1's eigenvector
#  iteration of power till convergence
# Note: P = transitionMatrix[i, j] (Go from page j to page i)

transitionMatrix = np.array([
    [0.0,       0.0,     1/2,      0.0],
    [1/3,       0.0,     0.0,      1/2],
    [1/3,       1/2,     0.0,      1/2],
    [1/3,       1/2,     1/2,      0.0]
], dtype=float)

print("Transition matrix shape:", transitionMatrix.shape)

#  1.1 Eigenvector-based PageRank with eigenvalue 
# Concept: Since a stable distribution v satisfies M v = v, 1 is an eigenvalue, 
# and the PageRank scores are obtained from the associated eigenvector (normalized to sum 1).
eigenValues, eigenVectors = eig(transitionMatrix)
indexEigenOne = np.argmin(np.abs(eigenValues - 1))
pageRankEigen = eigenVectors[:, indexEigenOne].real
pageRankEigen = pageRankEigen / pageRankEigen.sum()

print("\n[Eigenvector method] PageRank scores:")
print(pageRankEigen)
print("Check: sum of scores =", pageRankEigen.sum())

#1.2 PageRank with power iteration 
# The concept is to begin with a uniform vector and continually multiply it by M until the vector ceases to change.
numPages = transitionMatrix.shape[0]
rankVector = np.ones(numPages) / numPages

tolerance = 1e-8
maxIterations = 1000

for iteration in range(maxIterations):
    nextRankVector = transitionMatrix @ rankVector
    diff = np.linalg.norm(nextRankVector - rankVector, ord=1)  # total change
    rankVector = nextRankVector
    if diff < tolerance:
        print(f"\n[Power iteration] Converged after {iteration+1} iterations.")
        break

pageRankPower = rankVector / rankVector.sum()

print("\n[Power iteration] PageRank scores:")
print(pageRankPower)
print("Check: sum of scores =", pageRankPower.sum())

# Highest-ranked page according to each method (convert 0-based to 1-based page id)
topPageEigen = np.argmax(pageRankEigen) + 1
topPagePower = np.argmax(pageRankPower) + 1

print(f"\nHighest-ranked page (eigenvector method): Page {topPageEigen}")
print(f"Highest-ranked page (power iteration):    Page {topPagePower}")

print("\nFinished Question 1 (PageRank).")


 # Dimensionality Reduction using PCA
 # The objective is to calculate covariance, eigen-decompose it, then project it onto PC1.
 # Data: data.csv contains the normalized height and weight of 100 individuals.

print("\n--- Question 2: PCA ---")

# Load the standardized height-weight data (columns "Height" and "Weight" are expected).
dataFrame = pd.read_csv("data.csv")
heightWeightData = dataFrame[["Height", "Weight"]].values   # shape (100, 2)

# 2.1 Covariance matrix of the two standardized features
covarianceMatrix = np.cov(heightWeightData, rowvar=False)
print("\nCovariance matrix (Height, Weight):")
print(covarianceMatrix)

# 2.2 Eigen-decomposition for a symmetric covariance matrix
# eigh returns sorted ascending by default; we will sort descending
eigenValuesPCA, eigenVectorsPCA = eigh(covarianceMatrix)
sortedIndices = np.argsort(eigenValuesPCA)[::-1]
eigenValuesPCA = eigenValuesPCA[sortedIndices]
eigenVectorsPCA = eigenVectorsPCA[:, sortedIndices]

print("\nEigenvalues (variance captured by each PC, descending):")
print(eigenValuesPCA)
print("\nPrincipal component directions (columns are PCs):")
print(eigenVectorsPCA)

# 2.3 Project data to the first principal component (PC1)
firstPrincipalComponent = eigenVectorsPCA[:, 0]
projectedData1D = heightWeightData @ firstPrincipalComponent

# Reconstruct points along the PC1 line (for a 2D visualization of the 1D projection)
reconstructedDataPC1 = np.outer(projectedData1D, firstPrincipalComponent)

# 2.4 Plots: original points and their PC1 reconstruction
plt.figure(figsize=(10, 4))

# Left: original standardized scatter
plt.subplot(1, 2, 1)
plt.scatter(heightWeightData[:, 0], heightWeightData[:, 1], alpha=0.7)
plt.title("Height vs Weight (Standardized)")
plt.xlabel("Height (z-score)")
plt.ylabel("Weight (z-score)")
plt.axhline(0, linewidth=0.5)
plt.axvline(0, linewidth=0.5)

# Right: points projected onto the line spanned by PC1
plt.subplot(1, 2, 2)
plt.scatter(reconstructedDataPC1[:, 0], reconstructedDataPC1[:, 1], alpha=0.7)
plt.title("Data Projected onto PC1")
plt.xlabel("PC1 axis (x projection)")
plt.ylabel("PC1 axis (y projection)")
plt.axhline(0, linewidth=0.5)
plt.axvline(0, linewidth=0.5)

plt.tight_layout()
plt.show()

print("\nFinished Question 2 (PCA).")


#Using Least Squares for Linear Regression
 # Objective: forecast a new house price after fitting β in Xβ ≈ y.
 # Note: y is expressed in thousands of dollars.

print("\n--- Question 3: Linear Regression ---")

# Features: [square footage, bedrooms, age]
featureMatrix = np.array([
    [2100, 3, 20],
    [2500, 4, 15],
    [1800, 2, 30],
    [2200, 3, 25]
], dtype=float)

# Targets in $1000s
targetVector = np.array([460, 540, 330, 400], dtype=float)

#3.1 Find the least squares solution, which yields β along with some diagnostics
regressionCoefficients, residualSumSquares, matrixRank, singularValues = lstsq(featureMatrix, targetVector)

print("\nLeast-squares fit:")
print("Estimated coefficients β =", regressionCoefficients)
print("Residual sum of squares   =", residualSumSquares)
print("Rank of X                 =", matrixRank)

# 3.2 Predict on a new sample
newHouseFeatures = np.array([2400, 3, 20], dtype=float)
predictedPrice = newHouseFeatures @ regressionCoefficients
print(f"\nPredicted price for [2400 sq ft, 3 beds, 20 yrs]: ${predictedPrice:.2f}K")
# Contrast with the closed form of the normal equation solution on X^T X.
normalEqCoefficients = np.linalg.solve(featureMatrix.T @ featureMatrix, featureMatrix.T @ targetVector)
print("\nCoefficients via normal equation:")
print(normalEqCoefficients)

print("\nFinished Question 3 (Linear Regression).")


#Gradient Descent to Reduce the Loss Function
 # Loss: f(X) = 0.5 * sum_{i,j} (X_ij - A_ij)^2
 # Gradient: f(X) = X - A
 # Method: use scipy. optimize.minimize (L-BFGS-B) using our gradient and loss.     
 # Every iteration, we additionally monitor the loss value and chart the decline.
print("\n--- Question 4: Gradient Descent ---")

# Problem size (as required)
numRows, numCols = 100, 50

# Fix the seed so the random matrices are reproducible
np.random.seed(1)
targetMatrix = np.random.randn(numRows, numCols)       # A in the description
initialGuessMatrix = np.random.randn(numRows, numCols) # X_0 to start from

# Loss function: takes a flattened vector, reshapes to (numRows, numCols), and computes MSE * 0.5
def computeLoss(flattenedX):
    reshapedX = flattenedX.reshape(numRows, numCols)
    diffMatrix = reshapedX - targetMatrix
    return 0.5 * np.sum(diffMatrix * diffMatrix)

# Analytic gradient: f(X) = X - A, flattened to conform to the optimizer API
def computeGradient(flattenedX):
    reshapedX = flattenedX.reshape(numRows, numCols)
    gradientMatrix = reshapedX - targetMatrix
    return gradientMatrix.ravel()

# For visualization, maintain a straightforward list of loss values over iterations.
lossHistory = []

def trackLoss(currentX):
    # Callback gets the current parameter vector; we record the current loss
    currentLoss = computeLoss(currentX)
    lossHistory.append(currentLoss)

# Run L-BFGS-B with our gradient and loss; halt when the maxiter is reached or the gradient tolerance is low.
optimizationResult = minimize(
    fun=computeLoss,
    x0=initialGuessMatrix.ravel(),
    jac=computeGradient,
    method="L-BFGS-B",
    callback=trackLoss,
    options={"maxiter": 1000, "gtol": 1e-6}
)

print("\nOptimization summary:")
print("Converged successfully:", optimizationResult.success)
print("Final loss value:", optimizationResult.fun)
print("Number of iterations:", optimizationResult.nit)

# Plot how the loss dropped over iterations
plt.figure()
plt.plot(lossHistory, marker="o", linewidth=1)
plt.title("Loss vs Iteration (Gradient-Based Optimization)")
plt.xlabel("Iteration")
plt.ylabel("Loss f(X)")
plt.grid(True)
plt.show()

print("\nFinished Question 4 (Gradient Descent).")
