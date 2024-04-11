import numpy as np
import matplotlib.pyplot as plt

## 1a)
def n(x,A,Nsat,a,b,c):
    #if x == 0:
        #return 0
    return A * Nsat *(x/b)**(a-1)*np.exp(-(x/b)**c)

# Define the number fo satellites in the infinitesimal range
def N_A(x,A,Nsat,a,b,c):
    return 4*np.pi*x**2*n(x,A,Nsat,a,b,c)

# Find the max of N for x in [0,5) using a maximization algorithm
# Golden section search algorithm
def golden_section_search(f, a, b, tol=1e-5):
    golden_ratio = (np.sqrt(5) + 1) / 2
    c = b - (b - a) / golden_ratio
    d = a + (b - a) / golden_ratio

    while abs(c - d) > tol:
        if f(c) < f(d):
            b = d
        else:
            a = c

        # Update c and d
        c = b - (b - a) / golden_ratio
        d = a + (b - a) / golden_ratio

    return (b + a) / 2, f((b + a) / 2)

# Parameters
a = 2.4
b = 0.25
c = 1.6
A = 256 / (5 * np.pi ** (3 / 2))
Nsat = 100
xmax = 5

# Define the objective function to maximize
objective_function = lambda x: -N_A(x, A, Nsat, a, b, c)

# Find the maximum of N(x) using golden section search
# Minimize the negative of the function you want to maximize
max_x, max_N = golden_section_search(objective_function, 0, xmax)

# Open a text file in write mode
with open("output.txt", "w") as f:
    # Write the maximum x value to the file
    f.write("Maximum at x = {}\n".format(max_x))
    # Write the maximum N(x) value to the file
    # When retrieving the result, we reverse the sign back to get the correct maximum value
    f.write("Maximum N(x) = {}\n".format(-max_N))


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
    return radius, nhalo #Return the virial radius for all the satellites in the file, and the number of halos


files = ['satgals_m11.txt', 'satgals_m12.txt', 'satgals_m13.txt','satgals_m14.txt','satgals_m15.txt']  

# Iterate over each file in the list
for file in files:
    # Call the readfile function for each file
    radius, nhalo = readfile(file)

## 1b)
# TO DO: change the minimization algo into derivative-based one
# Change the plotting routine to match the one provided

# Define the function n(x)
def n(x, A, Nsat, a, b, c):
    if x == 0:
        return 0
    return A * Nsat * (x / b) ** (a - 1) * np.exp(-(x / b) ** c)

# Define the function N(x) using numerical integration (Trapezoidal rule)
def N(x_min, x_max, A, Nsat, a, b, c):
    def integrand(x):
        return 4 * np.pi * x**2 * n(x, A, Nsat, a, b, c)
    
    return trapezoidal_rule(integrand, x_min, x_max, 1000) # Use trapezoidal rule for integration

# Define the trapezoidal rule integration
def trapezoidal_rule(f, a, b, n):
    h = (b - a) / n
    result = 0.5 * (f(a) + f(b))
    for i in range(1, n):
        result += f(a + i * h)
    result *= h
    return result

# Golden section search algorithm - to minimize
def golden_section_search(f, a_initial, b_initial, c_initial, x_bins, N_data, tol=1e-5):
    golden_ratio = (np.sqrt(5) + 1) / 2
    a = a_initial
    b = b_initial
    c = c_initial
    d = b - (b - a) / golden_ratio
    e = a + (b - a) / golden_ratio

    while abs(c - d) > tol:
        if f([a, b, c], x_bins, N_data) < f([a, b, d], x_bins, N_data):
            b = d
        else:
            a = c

        if f([a, b, b], x_bins, N_data) < f([a, b, e], x_bins, N_data):
            c = e
            d = b - (b - a) / golden_ratio
            e = a + (b - a) / golden_ratio
        else:
            a = c
            c = d
            d = b - (b - c) / golden_ratio
            e = c + (b - c) / golden_ratio

    return [b, a, c], f([b, a, c], x_bins, N_data)

# Function to define the χ^2 
def chi_square(params, x_bins, N_data, filename):
    Nsat_avg = calculate_Nsat(filename)
    a, b, c = params
    A = calculate_A(a, b, c, Nsat_avg)  # Recalculate A using integration routine
    chi_square = 0
    for i in range(len(x_bins) - 1):
        x_min = x_bins[i]
        x_max = x_bins[i+1]
        N_model = N(x_min, x_max, A, Nsat_avg, a, b, c) 
        # Calculate variance (assuming Poisson statistics)
        variance = max(N_data[i], 1) # Avoid division by zero if there are no satellites in the bin
        chi_square += (N_data[i] - N_model)**2 / variance
    return chi_square

# Calculate A using integration routine
def calculate_A(a, b, c, Nsat_avg):
    # Perform numerical integration to calculate A
    A = Nsat_avg / trapezoidal_rule(lambda x: (x / b) ** (a - 3) * np.exp(-(x / b) ** c) if x != 0 else 0, 0, xmax, 1000)
    return A

# Read file and calculate ⟨Nsat⟩
def calculate_Nsat(filename):
    radius, nhalo = readfile(filename)
    Nsat_avg = len(radius) / nhalo
    #Nsat_avg = np.mean(nhalo)
    return Nsat_avg


# Function to compute the observed number of satellites in the specified range [x_min, x_max]
def N_observed(x_min, x_max, radius, nhalo):
    # Initialize array to store the number of satellites per halo in the specified range
    satellites_per_halo = []

    # Iterate over each halo
    for _ in range(nhalo):
        # Initialize a counter to keep track of satellites within the range for the current halo
        satellites_in_range_count = 0

        # Count the number of satellites within the range [x_min, x_max] for the current halo
        # Considering each coordinate 'radius' corresponds to a satellite
        for r in radius:
            if x_min <= r < x_max:
                satellites_in_range_count += 1

        # Append the count to the list for the current halo
        satellites_per_halo.append(satellites_in_range_count)

    # Compute the mean number of satellites per halo in the specified range
    return np.mean(satellites_per_halo)


# Parameters
a_initial = 2.4
b_initial = 0.25
c_initial = 1.6
xmax = 5

# Files
#files = ['file1.txt', 'file2.txt', 'file3.txt', 'file4.txt', 'file5.txt']
files = ['satgals_m15.txt']

# Initialize empty dictionary to store the best-fit parameters for each file (fix this)
#best_fit_params_chi = {}
#min_chi_square = {} 

for file in files:
    Nsat_avg = calculate_Nsat(file)
    #x_bins = np.linspace(1e-4, 5, 50) # Choose any number of radial bins
    x_bins = np.linspace(np.min(radius), np.max(radius), 50)
    N_data = [N_observed(x_min, x_max, radius, nhalo) for x_min, x_max in zip(x_bins[:-1], x_bins[1:])]
    # Perform minimization of χ^2 using golden section search
    result = golden_section_search(lambda params, x_bins, N_data: chi_square(params, x_bins, N_data, file), a_initial, b_initial, c_initial, x_bins, N_data)

    best_fit_params_chi = result[0]
    min_chi_square = result[1]
    
    # Calculate A using best-fit parameters
    A = calculate_A(*best_fit_params_chi, Nsat_avg)
 
    print("For", file, ":")
    print("⟨Nsat⟩:", Nsat_avg)
    print("Best-fit parameters (a, b, c):", best_fit_params_chi)
    print("Minimum χ^2:", min_chi_square)
    
    # Compute the N observed for each bin
    x_radii = []
    for i in range(len(x_bins)-1):
        x_min = x_bins[i]
        x_max = x_bins[i+1]
        x_radii.append(N_observed(x_min, x_max, radius, nhalo)) 
    print('x_radii:', x_radii)
    
    xmin = 1e-4
    xmax = 5
    edges = np.exp(np.linspace(np.log(xmin), np.log(xmax), len(x_bins)+1))
    fig1b, ax = plt.subplots(3,2,figsize=(6.4,8.0))
    for i in range(len(x_bins) - 1):
        x_min = x_bins[i]
        x_max = x_bins[i+1]
        N_model_chi = [N(x_min, x_max, A, Nsat_avg, *best_fit_params_chi)/Nsat_avg for x_min, x_max in zip(edges[:-1], edges[1:])]
        binned_data = np.histogram(x_radii, bins=edges)[0]/Nsat_avg

        file_index = files.index(file)
        row = file_index // 2
        col = file_index % 2

        # Plot 
  
        ax[row,col].step(edges[:-1], binned_data, where='mid', label='Binned data', color='r')  
        ax[row,col].step(edges[:-1], N_model_chi, where='mid', label='Model', color='b') 
        ax[row,col].set(yscale='log', xscale='log', xlabel='x', ylabel='N', title=f"$M_h \\approx 10^{{{11+i}}} M_{{\\odot}}/h$")
           
ax[2,1].set_visible(False)
plt.tight_layout()
handles,labels=ax[2,0].get_legend_handles_labels()
plt.figlegend(handles, labels, loc=(0.65,0.15))
plt.savefig('my_solution_1b.png', dpi=600)

