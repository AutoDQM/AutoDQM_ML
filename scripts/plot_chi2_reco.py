import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Read the CSV file
df = pd.read_csv('your_file.csv')

# Step 2: Extract Data
names = df['']
array1 = df['Array1']
array2 = df['Array2']

# Step 3: Plot the Arrays
plt.figure(figsize=(10, 6))  # Adjust figure size if needed

for i in range(len(names)):
    plt.plot(array1[i], label=names[i] + ' - Array 1')
    plt.plot(array2[i], label=names[i] + ' - Array 2')

plt.xlabel('Index')
plt.ylabel('Value')
plt.title('Arrays Overlaid for Each Name')
plt.legend()
plt.grid(True)
plt.show()
