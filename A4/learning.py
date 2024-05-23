import numpy as np
import matplotlib.pyplot as plt
import itertools

# Read the data from the file
with open('galaxy_data.txt', 'r') as file:
    data = file.readlines()

# Process the data
features = []
labels = []
for line in data:
    # Split the line and convert to float
    values = [float(val) for val in line.split()]
    # Take the first 4 columns as features
    features.append(values[:4])
    # Take the 5th column as label
    labels.append(values[4])

# Convert lists to numpy arrays
features = np.array(features)
labels = np.array(labels)

# Calculate the mean and standard deviation of the features 
mean = [sum(feature) / len(feature) for feature in zip(*features)]
std_dev = [((sum((xi - mean[i]) ** 2 for xi in feature)) / len(feature)) ** 0.5 for i, feature in enumerate(zip(*features))]

# Apply feature scaling (standardization)
scaled_features = [[(xi - mean[i]) / std_dev[i] for i, xi in enumerate(x)] for x in features]
scaled_features = np.array(scaled_features)

# Add a column of ones to the features for the bias term
scaled_features = np.hstack([np.ones((scaled_features.shape[0], 1)), scaled_features])

# Plot the distributions of the rescaled features
fig, ax = plt.subplots(2, 2, figsize=(10, 8))
ax[0, 0].hist(features[:, 0], bins=20)
ax[0, 0].set(ylabel='N', xlabel=r'$\kappa_{CO}$')
ax[0, 1].hist(features[:, 1], bins=20)
ax[0, 1].set(xlabel='Color')
ax[1, 0].hist(features[:, 2], bins=20)
ax[1, 0].set(ylabel='N', xlabel='Extended')
ax[1, 1].hist(features[:, 3], bins=20)
ax[1, 1].set(xlabel='Emission line flux')
plt.savefig("./plots/fig3a.png")
plt.close()

with open('3a.txt', 'w') as f:
    f.write("First 10 scaled features:\n")
    for i in range(10):
        f.write(str(scaled_features[i]) + "\n")

# Bracketing function
def bracketing(f, a, b, n):
    '''
    Input:
    - f: function to be minimized
    - a: initial guess for one point
    - b: initial guess for another point
    - n: number of iterations
    Output:
    - a, b, c: points such that f(b) < f(a) and f(b) < f(c)
    '''
    golden_ratio = (np.sqrt(5) + 1) / 2
    w = 1 / (1 + golden_ratio)
    if f(b) > f(a):
        a, b = b, a
    for _ in range(n):
        c = b + (b - a) * w
        if f(c) > f(b):
            return a, b, c
        else:
            fa = f(a)
            fb = f(b)
            fc = f(c)
            d = b - 0.5 * ((b - a) ** 2 * (fb - fc) - (b - c) ** 2 * (fb - fa)) / ((b - a) * (fb - fc) - (b - c) * (fb - fa))
            if b < d < c:
                if f(d) < f(c):
                    a = b
                    b = d
                elif f(d) > f(b):
                    c = d
                else:
                    d = c + (c - b) * w
            elif abs(d - b) > 100 * abs(c - b):
                d = c + (c - b) * w
            a = b
            b = c
            c = d
    return a, b, c

# Golden search function
def golden_search(f, a, b, tol=1e-7):
    '''
    Input:
    - f: function to be minimized
    - a: initial guess for one point
    - b: initial guess for another point
    - tol: tolerance for stopping criterion
    Output:
    - b: minimum point
    - f(b): minimum value
    '''
    golden_ratio = (1 + np.sqrt(5)) / 2
    w = 1 / (1 + golden_ratio)
    a, b, c = bracketing(f, a, b, n=3)
    while np.abs(c - a) > tol:
        ab_larger = np.abs(b - a) > np.abs(c - b)
        if ab_larger:
            x = a
        else:
            x = c
        d = b + (x - b) * w
        if f(d) < f(b):
            if ab_larger:
                c, b = b, d
            else:
                a, b = b, d
        elif ab_larger:
            a = d
        else:
            c = d
        if np.abs(c - a) <= tol:
            if f(d) < f(b):
                return d, f(d)
            else:
                return b, f(b)
    return b, f(b)

# Logistic function
def sigmoid(x):
    '''
    Input:
    - x: input value or array
    Output:
    - sigmoid of x
    '''
    return 1 / (1 + np.exp(-x))

# Cost function
def cost_function(theta, X, y):
    '''
    Input:
    - theta: weights
    - X: features
    - y: labels
    Output:
    - cost: logistic regression cost
    '''
    m = len(y)
    h_theta = sigmoid(np.dot(X, theta))
    error = (y * np.log(h_theta)) + ((1 - y) * np.log(1 - h_theta))
    cost = -1 / m * np.sum(error)
    return cost

# Gradient of the cost function
def gradient(theta, X, y):
    '''
    Input:
    - theta: weights
    - X: features
    - y: labels
    Output:
    - grad: gradient of the cost function
    '''
    m = len(y)
    h_theta = sigmoid(np.dot(X, theta))
    grad = np.dot(X.T, (h_theta - y)) / m
    return grad

# Perform logistic regression using the conjugate gradient method
def logistic_regression_cg(X, y, theta, num_iters=150, tol=1e-7):
    '''
    Input:
    - X: features
    - y: labels
    - theta: initial weights
    - num_iters: number of iterations
    - tol: tolerance for stopping criterion
    Output:
    - theta: optimized weights
    - cost_history: history of cost function values
    '''
    cost_history = []
    d = -gradient(theta, X, y)
    for _ in range(num_iters):
        old_cost = cost_function(theta, X, y)
        alpha = golden_search(lambda a: cost_function(theta + a * d, X, y), 0, 1)[0]
        theta += alpha * d
        new_grad = gradient(theta, X, y)
        beta = np.dot(new_grad.T, new_grad) / np.dot(d.T, d)
        d = -new_grad + beta * d
        new_cost = cost_function(theta, X, y)
        cost_history.append(new_cost)
        if abs(new_cost - old_cost) < tol:
            break
    return theta, cost_history

# Features 1 and 2
X_12 = scaled_features[:, [0, 1, 2]]
initial_theta_12 = np.ones(X_12.shape[1])
theta_12, cost_history_12 = logistic_regression_cg(X_12, labels, initial_theta_12)

# Features 1 and 3
X_13 = scaled_features[:, [0, 1, 3]]
initial_theta_13 = np.ones(X_13.shape[1])
theta_13, cost_history_13 = logistic_regression_cg(X_13, labels, initial_theta_13)

# Plotting
fig, ax = plt.subplots(1, 1, figsize=(10, 5), constrained_layout=True)
ax.plot(np.arange(0, len(cost_history_12)), cost_history_12, label='Features 1+2')
ax.plot(np.arange(0, len(cost_history_13)), cost_history_13, label='Features 1+3')
ax.set(xlabel='Number of iterations', ylabel='Cost function')
plt.legend(loc=(1.05, 0))
plt.savefig("./plots/fig3b.png")
plt.close()

# Define a function to calculate the decision boundary
def decision_boundary(x, theta):
    '''
    Input:
    - x: input feature value
    - theta: weights
    Output:
    - y: decision boundary value
    '''
    return -(theta[0] + theta[1] * x) / theta[2]

fig, ax = plt.subplots(3, 2, figsize=(10, 15))
names = [r'$\kappa_{CO}$', 'Color', 'Extended', 'Emission line flux']
plot_idx = [[0, 0], [0, 1], [1, 0], [1, 1], [2, 0], [2, 1]]

combinations = list(itertools.combinations(range(4), 2))

with open('3c.txt', 'w') as f:
    f.write(f"{'Feature 1':<15}{'TP':<10}{'FP':<10}{'TN':<10}{'FN':<10}{'F1':<10}\n")
    for i, comb in enumerate(combinations):
        X = scaled_features[:, [0, comb[0] + 1, comb[1] + 1]]
        initial_theta = np.ones(X.shape[1])
        theta_min, _ = logistic_regression_cg(X, labels, initial_theta)

        y_pred = sigmoid(np.dot(X, theta_min)) >= 0.5
        tp = np.sum((y_pred == 1) & (labels == 1))
        fp = np.sum((y_pred == 1) & (labels == 0))
        tn = np.sum((y_pred == 0) & (labels == 0))
        fn = np.sum((y_pred == 0) & (labels == 1))
        f1_score = tp / (tp + 0.5 * (fp + fn))
        f.write(f"{names[comb[0]]:<15}{tp:<10}{fp:<10}{tn:<10}{fn:<10}{f1_score:<10.2f}\n")
        ax[plot_idx[i][0], plot_idx[i][1]].scatter(scaled_features[:, comb[0] + 1], scaled_features[:, comb[1] + 1], c=labels)
        x_values = [np.min(scaled_features[:, comb[0] + 1]), np.max(scaled_features[:, comb[0] + 1])]
        y_values = [decision_boundary(x, theta_min) for x in x_values]
        ax[plot_idx[i][0], plot_idx[i][1]].plot(x_values, y_values, color='red')
        ax[plot_idx[i][0], plot_idx[i][1]].set(xlabel=names[comb[0]], ylabel=names[comb[1]])


        ax[0, 1].set_ylim(-1, 30)
        ax[1, 0].set_ylim(-0.5, 1.25)
        ax[1, 1].set_ylim(-0.5, 15)
        ax[2, 0].set_ylim(-0.5, 1.5)
        ax[2, 1].set_ylim(-0.25, 1.25)
        ax[2, 1].set_xlim(-1, 4)

plt.savefig("./plots/fig3c.png")
plt.close()

