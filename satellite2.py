import numpy as np
import matplotlib.pyplot as plt

## a)

# Define the number of satellites in the infinitesimal range
def N(x,A,Nsat,a,b,c):
    # This includes the factor x^2
    return 4*np.pi* A * Nsat * x**(a-1) * b**(3-a) * np.exp(-(x/b)**c)

# Bracketing algorithm to find the initial bracket for the step size (the first argument of update)
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
    # If necessary, swap a and b
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
            #Compute the function values at a, b, and c
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
                #Set d using the golden ratio
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

              
# Parameters
a = 2.4
b = 0.25
c = 1.6
A = 256 / (5 * np.pi ** (3 / 2))
Nsat = 100
xmax = 5

# Define the objective function to maximize
objective_function = lambda x: -N(x, A, Nsat, a, b, c)

# Find the maximum of N(x) for x in [0,5) using golden section search
# Minimize the negative of the function you want to maximize
max_x, max_N = golden_search(objective_function, 0, xmax)

with open("output_a.txt", "w") as f:
    f.write("Maximum at x = {}\n".format(max_x))
    # When retrieving the result, we reverse the sign back to get the correct maximum value
    f.write("Maximum N(x) = {}\n".format(-max_N))

##b)
    
# Define the trapezoidal rule integration
def trapezoidal_rule(f, a, b, n):
    h = (b - a) / n
    result = 0.5 * (f(a) + f(b))
    for i in range(1, n):
        result += f(a + i * h)
    result *= h
    return result

# Romberg integration
def romberg(f, a, b, m=6, n=4):
    '''
    Inputs:
    f: Function to integrate
    a, b: Integration limits
    n: Order of the Romberg integration
    n: Number of intervals for the first estimate
    '''
    # Initialize the Romberg table
    R = np.zeros(m)

    for i in range(0,m):
        # At each iteration double the number of intervals
        R[i] = trapezoidal_rule(f, a, b, n * 2 ** i)
   
    # Loop over columns
    for j in range(1, m):
        # Loop over rows
        for k in range(0, m-j):
            R[k] = (4**j * R[k+1] - R[k]) / (4**j - 1)
    return R[0] # Return the best estimate


# Define the function n(x) without the constants and already including x^2
# Including this helps to avoid numerical issues when integrating
def n(x, a, b, c):
    return x**(a-1) * b**(3-a) * np.exp(-(x/b)**c)

# Compute the normalization factor A
def compute_A(a, b, c):
    def integrand(x):
        return 4 * np.pi * n(x, a, b, c) 
    return 1 / romberg(integrand, 0, 5) 

# Compute the average number of satellites per halo
def calculate_Nsat(filename):
    radius, nhalo = readfile(filename)
    Nsat_avg = len(radius) / nhalo
    return Nsat_avg

# Define the model N(x) for the number of satellites
def N_model(edges, Nsat, a, b, c):
    N = []
    A = compute_A(a, b, c)
    for i in range(len(edges) - 1):
        integral = 4 * np.pi * A * Nsat * romberg(lambda x: n(x, a, b, c), edges[i], edges[i+1]) 
        N.append(integral)
    return np.array(N)

# Define the chi-squared function
def chi_squared(data, edges, Nsat, params):
    a, b, c = params
    model = N_model(edges, Nsat, a, b, c)
    chi = np.sum((data - model)**2 / model) 
    return chi

# Analytical derivatives wrt a, b, c
def derivative_a(edges,a,b,c, Nsat,A):
    d_a = []
    for i in range(len(edges)-1):
        da = 4 * np.pi * A * Nsat * b**(3-a) * romberg(lambda x: np.log(x/b) * x**(a-1) * np.exp(-(x / b) ** c), edges[i], edges[i+1])
        d_a.append(da)
    return np.array(d_a)
    
def derivative_b(edges, a, b, c, Nsat, A):
    d_b = []
    for i in range(len(edges)-1):
        db = 4 * np.pi * A * Nsat * b**(3-a)* romberg(lambda x: x**(a-1) * np.exp(-(x/b)**c) * (c * x**c / b**(c+1) + (3-a)/b), edges[i], edges[i+1])
        d_b.append(db)
    return  np.array(d_b)
   
def derivative_c(edges,a,b,c, Nsat,A):
    d_c = []
    for i in range(len(edges)-1):
        dc = 4 * np.pi * Nsat * A * b**(3-a) * romberg(lambda x: -np.log(x/b) * x**(a-1) * np.exp(-(x/b)**c) * (x/b)**c , edges[i], edges[i+1])
        d_c.append(dc)
    return np.array(d_c)
    

# Define the gradient of your chi-square function with respect to parameters 
def chi_squared_gradient(a, b, c , edges, N_data, Nsat):
    A = compute_A(a, b, c) 
    # Multiplying factor 
    fac = N_data**2 / N_model(edges, Nsat, a, b, c)**2 -1
    # Gradient with respect to a - Sum over all bins
    grad_a = np.sum(derivative_a(edges,a,b,c,Nsat,A) * fac)
    # Gradient with respect to b
    grad_b = np.sum(derivative_b(edges,a,b,c,Nsat,A) * fac)
    # Gradient with respect to c
    grad_c = np.sum(derivative_c(edges,a,b,c,Nsat,A) * fac)
    
    return grad_a, grad_b, grad_c

# Now we adapt the algorithms implemented in part a) to do line minimization in the conjugate gradient method
# We have to minimize the funcion f(xi + lambda * ni) where xi is the current point, ni is the direction, and lambda is the step size
#Our goal is to find the best step size lambda that minimizes the function

# Bracketing algorithm to find the initial bracket for the step size (the first argument of update)
def bracketing_for_step(f, args, a, b, n):
    '''
    Inputs:
    f: Objective function
    args: Arguments of the objective function
    a, b: Starting and ending point of the bracket
    n: Number of iterations

    Returns:
    a, b, c: The bracketing points for the step size

    The argument in which we are interested is the step size, so we will treat the args explicitly as a tuple
    to be able to swap when necessary the step variable
    '''
    golden_ratio = (np.sqrt(5) + 1) / 2
    w = 1 / (1 + golden_ratio)
    N_data, edges, Nsat, params, direction = args

    # Ensure that f(b) < f(a)
    # If necessary, swap 'a' and 'b'
    if f(N_data, edges, Nsat, update(b, params, direction)) > f(N_data, edges, Nsat, update(a, params, direction)):
        a, b = b, a

    for _ in range(n):
        # Propose a point c using the golden ratio
        c = b + (b - a) * w

        # If f(c) > f(b), we have found a bracket
        if f(N_data, edges, Nsat, update(c, params, direction)) > f(N_data, edges, Nsat, update(b, params, direction)):
            return a, b, c
        # Otherwise, use a, b and c to find a new point by fitting a parabola
        else:
            #Compute the function values at a, b, and c
            fa = f(N_data, edges, Nsat, update(a, params, direction))
            fb = f(N_data, edges, Nsat, update(b, params, direction))
            fc = f(N_data, edges, Nsat, update(c, params, direction))

            # The new point is given by the minimum of the parabola
            d = b - 0.5 * ((b - a) ** 2 * (fb - fc) - (b - c) ** 2 * (fb - fa)) / ((b - a) * (fb - fc) - (b - c) * (fb - fa))

            
            if b < d < c:
                # We might be done, so check if f(d) < f(c)
                if f(N_data, edges, Nsat, update(d, params, direction)) < f(N_data, edges, Nsat, update(c, params, direction)):
                    # New bracket = [b, d, c]
                    a = b
                    b = d
                # Otherwise, check if f(d) > f(b)
                elif f(N_data, edges, Nsat, update(d, params, direction)) > f(N_data, edges, Nsat, update(b, params, direction)):
                    # New bracket = [a, b, d]
                    c = d
                # If neither condition is met, the parabola is a bad fit
                #Set d using the golden ratio
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


# Define the golden search algorithm to find the best step size
def golden_search_for_step(f, args, a, b, tol=1e-7):
    '''
    Inputs:
    f: Objective function
    args: Arguments of the objective function
    a, b: Starting and ending point of the bracket
    tol: Tolerance for the convergence

    Returns:
    The minimum of the function f within the specified interval [a, b]
    '''
    # Unpack the arguments
    N_data, edges, Nsat, params, direction = args
    
    golden_ratio = (1+np.sqrt(5)) / 2
    w = 1/ (1+golden_ratio)
    
    # Set the intial bracket
    a, b, c = bracketing_for_step(f, args, a, b, n=3)
    
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

        if f(N_data, edges, Nsat, update(d, params, direction)) < f(N_data, edges, Nsat, update(b, params, direction)):
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
            if f(N_data, edges, Nsat, update(d, params, direction)) < f(N_data, edges, Nsat, update(b, params, direction)):
                return d
            # Otherwise, return b
            else:
                return b
            
    # If the loop ends without returning, return b (the central value of the bracket) 
    #In this case we don't need the function value at b
    return b

    
# Define the update of the parameters using the conjugate gradient method
def update(step, params, direction):
    return params + step * direction

# Define the conjugate gradient descent method
def conjugate_gradient(a,b,c, edges, N_data, Nsat, max_iter, f, g, tol=1e-6):
    '''
    Inputs:
    a, b, c: Initial parameters
    edges: Bin edges
    N_data: Data
    Nsat: Average number of satellites
    max_iter: Maximum number of iterations
    f: Objective function
    g: Gradient function
    tol: Tolerance for convergence
    '''
    params = np.array([a, b, c])
    
    iter_count = 0

    for _ in range(max_iter):
        # Calculate the value and gradient at the current parameters
        f_value = f(N_data, edges, Nsat, params) 
        gradient_new = - np.array(g(*params, edges, N_data, Nsat))
        norm = [(np.sum(gradient_new**2))**0.5]
        # Check for convergence on the norm of the gradient by comparing to the target accuracy
        if norm[0] < tol:
            return params, f_value

        # If there is no previous gradient, set weight to 0
        if iter_count == 0:
            weight = 0
            direction_new = -gradient_new
            # Set lambda ends of the interval based on the norm of the gradient
            l_start = -1/np.abs(norm)
            l_end = l_start/3

        # Determine the weight using the Polak-Ribiere formula
        else:
            weight = np.dot((gradient_new - gradient), gradient_new)/np.dot(gradient, gradient)
            direction_new = gradient_new + weight * direction

        # Apply line minimization over f to get minimum lambda
        best_step = golden_search_for_step(f, [N_data, edges, Nsat, params, direction_new], l_start, l_end) 
        # Update parameters accordingly
        params = update(best_step, params, direction_new)
        # Evaluate the new function value
        f_value_new = f(N_data, edges, Nsat, params)
        
        # Check for convergence by comparison with the target accuracy
        convergence = np.abs(f_value_new - f_value) / np.abs(0.5*(f_value_new+f_value))
        if convergence < tol:
            # Check for convergence by comparison with the target accuracy
            return params, f_value_new
        else:
            # Go back to first step
            gradient = gradient_new
            direction = direction_new

    return params, f_value_new


xmin, xmax = 1e-4, 5. 
n_bins = 20
edges = np.exp(np.linspace(np.log(xmin), np.log(xmax), n_bins+1))

# Parameters
a_initial = 2.4
b_initial = 0.25
c_initial = 1.6

def readfile(filename):
    f = open(filename, 'r')
    data = f.readlines()[3:] #Skip first 3 lines 
    nhalo = int(data[0]) #number of halos
    radius = []
    
    for line in data[1:]:
        if line[:-1]!='#':
            radius.append(float(line.split()[0]))
    radius = np.array(radius, dtype=float)   
    f.close()
    return radius, nhalo #Return the virial radius for all the satellites in the file, and the number of halos, and the mean number of satellites


files = ['satgals_m11.txt', 'satgals_m12.txt', 'satgals_m13.txt','satgals_m14.txt','satgals_m15.txt'] 


results_dict = {}

fig1b, ax = plt.subplots(3,2,figsize=(6.4,8.0))
for idx, filename in enumerate(files):
    x_radii, nhalo = readfile(filename)
    Nsat = calculate_Nsat(filename)
    binned_data=np.histogram(x_radii,bins=edges)[0]/nhalo
    # Perform conjugate gradient descent
    grad_a, grad_b, grad_c = chi_squared_gradient(a_initial, b_initial, c_initial,edges, binned_data, Nsat)
    best_fit_params_chi, min_chi_square = conjugate_gradient(a_initial, b_initial, c_initial, edges, binned_data, Nsat, max_iter=20, f=chi_squared, g=chi_squared_gradient, tol=1e-6)

    Ntilda = N_model(edges, Nsat, *best_fit_params_chi)  

    # Store results in dictionary
    results_dict[filename] = {
        "Nsat_avg": Nsat,
        "Best_fit_params": best_fit_params_chi,
        "Minimum_loglikelihood": min_chi_square
    }

    row = idx // 2
    col = idx % 2
    ax[row,col].step(edges[:-1], binned_data, where='post', label='binned data')
    ax[row,col].step(edges[:-1], Ntilda, where='post', label='best-fit profile')
    ax[row,col].set(yscale='log', xscale='log', xlabel='x', ylabel='N', title=f"$M_h \\approx 10^{{{11+idx}}} M_{{\\odot}}/h$")

# Write results to file
with open("output_b.txt", "w") as f:
    for filename, results in results_dict.items():
        f.write("For {}: \n".format(filename))
        f.write("Nsat avg: {}\n".format(results["Nsat_avg"]))
        f.write("Best-fit parameters (a, b, c): {}\n".format(results["Best_fit_params"]))
        f.write("Minimum of the loglikelihood: {}\n".format(results["Minimum_loglikelihood"]))
        f.write("\n")

ax[2,1].set_visible(False)
plt.tight_layout()
handles,labels=ax[2,0].get_legend_handles_labels()
plt.figlegend(handles, labels, loc=(0.65,0.15))
plt.savefig('./plots/my_solution_1b.png', dpi=600)

## c)

# Poisson log-likelihood
def poisson_log_likelihood(mu, y):
    return y * np.log(mu) - mu
   
# Define the negative Poisson log-likelihood
def neg_log_likelihood(N_data, edges, Nsat, params):
    a, b, c = params
    log_likelihood = 0
    for i in range(len(edges) - 1):
        N_model_values = N_model(edges, Nsat, a, b, c)
        log_likelihood += poisson_log_likelihood(N_model_values[i], N_data[i])
    return -log_likelihood

# Define the gradient of the negative Poisson log-likelihood
def likelihood_gradient(a, b, c, edges, N_data, Nsat):
    A = compute_A(a, b, c)
    # Multiplying factor
    fac = (N_data/N_model(edges, Nsat, a, b, c) - 1)
    # Gradient with respect to a - sum all over the bins
    grad_a = np.sum(derivative_a(edges,a,b,c,Nsat,A) * fac)
    # Gradient with respect to b
    grad_b = np.sum(derivative_b(edges,a,b,c,Nsat,A) * fac)
    # Gradient with respect to c
    grad_c = np.sum(derivative_c(edges,a,b,c,Nsat,A) * fac)
   
    return grad_a, grad_b, grad_c


fig1c, ax = plt.subplots(3, 2, figsize=(6.4, 8.0))
for idx, filename in enumerate(files):
    Nsat = calculate_Nsat(filename)
    x_radii, nhalo = readfile(filename)
    binned_data = np.histogram(x_radii, bins=edges)[0] / nhalo

    # Perform conjugate gradient descent
    initial_guess = np.array([a_initial, b_initial, c_initial])
    best_fit_params_like, min_log = conjugate_gradient(*initial_guess, edges, binned_data, Nsat, max_iter=5, f=neg_log_likelihood, g=likelihood_gradient, tol=1e-6)
    
    Ntilda = N_model(edges, Nsat, *best_fit_params_like) 
    
    # Write results to file
    with open("output_c.txt", "w") as f:
        f.write("For {}: \n".format(filename))
        f.write("Best-fit parameters (a, b, c): {}\n".format(best_fit_params_like))
        f.write("Minimum of the loglikelihood: {}\n".format(min_log))
        f.write("\n")

    row = idx // 2
    col = idx % 2
    ax[row, col].step(edges[:-1], binned_data, where='post', label='binned data')
    ax[row, col].step(edges[:-1], Ntilda, where='post', label='best-fit profile')
    ax[row, col].set(yscale='log', xscale='log', xlabel='x', ylabel='N', title=f"$M_h \\approx 10^{{{11+idx}}} M_{{\\odot}}/h$")

ax[2, 1].set_visible(False)
plt.tight_layout()
handles, labels = ax[2, 0].get_legend_handles_labels()
plt.figlegend(handles, labels, loc=(0.65, 0.15))
plt.savefig('./plots/my_solution_1c.png', dpi=600)

## d)
from scipy.special import gammainc, gamma

# Define the G-test
# O_i are the observations in each bin (integer values)
# E_i are the expected values in each bin (coming from the model)
def g_test(O_i, E_i):
    # Add a pseudocount to handle zero counts
    O_i += 1
    E_i += 1
    G = 2 * np.sum(O_i * np.log(O_i / E_i))
    return G

# Define the CDF of the chi-square distribution
def chi_square_cdf(x, k):
    return gammainc(k / 2, x / 2)/gamma(k/2)

# This function scales the model's predictions to match the total number of observed events
def N_scaled(edges, Nsat, a, b, c, observed_counts):
    N = []
    A = compute_A(a, b, c)
    for i in range(len(edges) - 1):
        integral = 4 * np.pi * A * Nsat * romberg(lambda x: n(x, a, b, c), edges[i], edges[i+1]) 
        N.append(integral)
    N = np.array(N)
    
    # Total number of events predicted by the model
    total_predicted = np.sum(N)
    
    # Total number of observed events
    total_observed = np.sum(observed_counts)
    
    # Scale the model's predictions to match the total number of observed events
    N_scaled = N * (total_observed / total_predicted)
    
    return N_scaled

for idx, filename in enumerate(files):
    Nsat = calculate_Nsat(filename)
    x_radii, nhalo = readfile(filename)

    # Get the observed data as the counts falling in each bin
    N_data = np.histogram(x_radii, bins=edges)[0] 

    best_fit_params_chi = results_dict[filename]["Best_fit_params"]
    best_fit_params_like = results_dict[filename]["Best_fit_params"]

    Ntilda_chi = N_scaled(edges, Nsat, *best_fit_params_chi, N_data)
    Ntilda_like = N_scaled(edges, Nsat, *best_fit_params_like, N_data)

    # Calculate the G-test for the chi-squared method
    G_chi = g_test(binned_data, Ntilda_chi)
    # Calculate the G-test for the Poisson log-likelihood method
    G_like = g_test(binned_data, Ntilda_like)

    # Calculate the degrees of freedom 
    # It is the same for both methods since we have the same number of parameters
    k = len(N_data) - len(best_fit_params_chi)

    # Calculate the p-value for the chi-squared method
    p_chi = 1 - chi_square_cdf(G_chi, k)
    # Calculate the p-value for the Poisson log-likelihood method
    p_like = 1 - chi_square_cdf(G_like, k)

    G_chi_reduced = G_chi / k
    G_likelihood_reduced = G_like / k

    with open("output_d.txt", "w") as f:
        f.write("For {}: \n".format(filename))
        f.write("G-test for chi-squared method: {}\n".format(G_chi))
        f.write("Reduced G-test for chi-squared method: {}\n".format(G_chi_reduced))
        f.write("p-value for chi-squared method: {}\n".format(p_chi))
        f.write("G-test for Poisson log-likelihood method: {}\n".format(G_like))
        f.write("p-value for Poisson log-likelihood method: {}\n".format(p_like))
        f.write("\n")







