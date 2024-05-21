import numpy as np
import matplotlib.pyplot as plt
import itertools

## a)

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

# Concatenate the features into a matrix of dimensions m x n
# In this case, the features are already in the required format (m x n) where m is the number of data points and n is the number of features (4 in this case)

# Calculate the mean and standard deviation of the features 
mean = [sum(feature) / len(feature) for feature in zip(*features)]
std_dev = [((sum((xi - mean[i]) ** 2 for xi in feature)) / len(feature)) ** 0.5 for i, feature in enumerate(zip(*features))]

# Apply feature scaling (standardization)
scaled_features = [[(xi - mean[i]) / std_dev[i] for i, xi in enumerate(x)] for x in features]

# Now, the scaled_features array has features with mean 0 and standard deviation 1

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

## b)
        
def bracketing(f, a, b, n):
    '''
    Inputs:
    f: Objective function
    args: Arguments of the objective function
    a, b: Starting and ending point of the bracket
    n: Number of iterations
    '''
    golden_ratio = (np.sqrt(5) + 1) / 2
    w = 1 / (1 + golden_ratio)

    # Ensure that f(b) < f(a)
    # If necessary, swap 'a' and 'b'
    if f(b) > f(a):
        a, b = b, a

    for _ in range(n):
        # Propose a point c using the golden ratio
        c = b + (b - a) * w

        # If f(c) > f(b), we have found a bracket
        if f(c) > f(b):
            return a, b, c
        # Otherwise, use a, b and c to find a new point by fitting a parabola
        else:
            # Compute the function values at a, b, and c
            fa = f(a)
            fb = f(b)
            fc = f(c)

            # The new point is given by the minimum of the parabola
            d = b - 0.5 * ((b - a) ** 2 * (fb - fc) - (b - c) ** 2 * (fb - fa)) / ((b - a) * (fb - fc) - (b - c) * (fb - fa))
     
            if b < d < c:
                # We might be done, so check if f(d) < f(c)
                if f(d) < f(c):
                    # New bracket = [b, d, c]
                    a = b
                    b = d
                # Otherwise, check if f(d) > f(b)
                elif f(d) > f(b):
                    # New bracket = [a, b, d]
                    c = d
                # If neither condition is met, the parabola is a bad fit
                # Set d using the golden ratio
                else:
                    d = c + (c - b) * w
            # If d is not in the interval [b, c], check if it is too far away
            # If it is, take another step
            elif abs(d - b) > 100 * abs(c - b):
                d = c + (c - b) * w

            # If not done, move all points over
            a = b
            b = c
            c = d

    # Final bracket
    return a, b, c


def golden_search(f, a, b, tol=1e-7):
    
    golden_ratio = (1+np.sqrt(5)) / 2
    w = 1/ (1+golden_ratio)
    
    # Set the intial bracket
    a, b, c = bracketing(f, a, b, n=3)
    
    while np.abs(c-a) > tol:
        # Set a boolean indicating whether [a,b] is the larger bracket compared to [b,c]
        # This is done for efficiency, since at each iteration the larger interval will be the one we didn't tighten in the previous step
        ab_larger = np.abs(b-a) > np.abs(c-b)
        
        if ab_larger:
            # x is the other edge of the larger interval
            x = a
        else:
            # If the condition is not true, we have [b,c] as the larger bracket
            x = c

        # Choose a new point inside the largest bracket 
        d = b + (x-b)* w

        if f(d) < f(b):
            # Tighten the bracket towards d
            # Depending on the largest bracket, update the values of a, b, and c by swapping
            if ab_larger:
                c, b = b, d
            else:
                a, b = b, d
        
        # If d is not a better point than b, adjust the brackets
        elif ab_larger:
            a=d
        else:
            c=d

        # If c-a is less than the tolerance
        if np.abs(c-a) <= tol:
            # If the function value at d is less than at b, return d
            if f(d) < f(b):
                return d, f(d)
            # Otherwise, return b
            else:
                return b, f(b)
            
    # If the loop ends without returning, return b (= central value of the bracket) and the function value at b
    return b, f(b)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def cost_function(theta, X, y):
    m = len(y)
    h_theta = sigmoid(np.dot(X, theta))
    error = (y * np.log(h_theta)) + ((1-y)*np.log(1-h_theta))
    cost = -1/m * np.sum(error)
    return cost

def gradient(theta, X, y):
    m = len(y)
    h_theta = sigmoid(np.dot(X, theta))
    grad = np.dot(X.T, (h_theta - y)) / m
    return grad

def logistic_regression_cg(X, y, theta, num_iters=150, tol=1e-7):
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

scaled_features = np.array(scaled_features)

# Features 1 and 2
X_12 = scaled_features[:, :2]
theta = np.zeros(X_12.shape[1])
theta, cost_history_12 = logistic_regression_cg(X_12, labels, theta)

# Features 1 and 3
X_13 = scaled_features[:, [0, 2]]
theta = np.zeros(X_13.shape[1])
theta, cost_history_13 = logistic_regression_cg(X_13, labels, theta)

# Plotting
fig, ax  = plt.subplots(1,1, figsize=(10,5), constrained_layout=True)
ax.plot(np.arange(0, len(cost_history_12)), cost_history_12, label='Features 1+2')
ax.plot(np.arange(0, len(cost_history_13)), cost_history_13, label='Features 1+3')

ax.set(xlabel='Number of iterations', ylabel='Cost function')
plt.legend(loc=(1.05,0))
plt.savefig("./plots/fig3b.png")
plt.close()

## c)

# Define a function to calculate the decision boundary
def decision_boundary(x, theta):
    return -theta[0]/theta[1] * x

fig, ax = plt.subplots(3,2,figsize=(10,15))
names = [r'$\kappa_{CO}$', 'Color', 'Extended', 'Emission line flux']
plot_idx = [[0,0], [0,1], [1,0], [1,1], [2,0], [2,1]]

with open('3c.txt', 'w') as f:
    # Iterate over all combinations of features
    for i, comb in enumerate(itertools.combinations(np.arange(0,4), 2)):
        # Select the features
        X = scaled_features[:, comb]
        theta = np.zeros(X.shape[1])
        
        # Run logistic regression
        theta_min, _ = logistic_regression_cg(X, labels, theta)
        
        # Predict the labels
        y_pred = sigmoid(np.dot(X, theta)) >= 0.5
        
        # Compute the confusion matrix
        tp = np.sum((labels == 1) & (y_pred == 1))
        fp = np.sum((labels == 0) & (y_pred == 1))
        tn = np.sum((labels == 0) & (y_pred == 0))
        fn = np.sum((labels == 1) & (y_pred == 0))
        
        # Compute precision, recall, and F1 score
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * precision * recall / (precision + recall)
        
        # Write the results to the file
        f.write(f"Features: {comb}\n")
        f.write(f"True Positives: {tp}\n")
        f.write(f"False Positives: {fp}\n")
        f.write(f"True Negatives: {tn}\n")
        f.write(f"False Negatives: {fn}\n")
        f.write(f"F1 Score: {f1}\n")
        
        # Plot the features and the decision boundary
        ax[plot_idx[i][0],plot_idx[i][1]].scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
        x_values = np.linspace(np.min(X[:, 0]), np.max(X[:, 0]), 100)
        ax[plot_idx[i][0],plot_idx[i][1]].plot(x_values, decision_boundary(x_values, theta_min), color='r')
        ax[plot_idx[i][0],plot_idx[i][1]].set(xlabel=names[comb[0]], ylabel=names[comb[1]])
        
        ax[0, 1].set_ylim(-1, 30)
        ax[1, 0].set_ylim(-0.5, 1.25)
        ax[1, 1].set_ylim(-0.5, 15)
        ax[2, 0].set_ylim(-0.5, 1.5)
        ax[2, 1].set_ylim(-0.25, 1.25)
        ax[2, 1].set_xlim(-1, 4)


plt.savefig("./plots/fig3c.png")
plt.close()