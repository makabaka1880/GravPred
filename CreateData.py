import pandas as pd
import math
import csv
import random

# Define the ranges for the random values
v_x_0_range = [0.5, 5]
v_y_0_range = [0.5, 5]
y_0_range = [1, 20]
g_range = [5, 15]

def calculate_p_x_at_t_star(y_0, v_x_0, v_y_0, g):
    discriminant = v_y_0**2 + 2 * g * y_0
    t_star = (-v_y_0 + math.sqrt(discriminant)) / g

    p_x_t_star = v_x_0 * t_star
    
    return p_x_t_star

# Number of samples to generate
num_samples = 10000

# Create and open the CSV file in write mode
with open('training_data.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['v_x_0', 'v_y_0', 'y_0', 'g', 'p_x_t_star'])  # Write header

    # Generate random samples and write them to the CSV file
    for _ in range(num_samples):
        v_x_0 = random.uniform(v_x_0_range[0], v_x_0_range[1])
        v_y_0 = random.uniform(v_y_0_range[0], v_y_0_range[1])
        y_0 = random.uniform(y_0_range[0], y_0_range[1])
        g = random.uniform(g_range[0], g_range[1])

        # Calculate p_x(t*) using the provided function
        p_x_t_star = calculate_p_x_at_t_star(y_0, v_x_0, v_y_0, g)

        # Write the values to the CSV file
        writer.writerow([v_x_0, v_y_0, y_0, g, p_x_t_star])

print("Done writing")