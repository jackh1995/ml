import numpy as np
from matplotlib import pyplot as plt

# Random seeds
np.random.seed(0)

# Theoratical weights
w = 7
b = 4
n_data = 100
noise_level = 10

# Generate data
x_list = np.random.uniform(low=-10.0, high=10.0, size=n_data)
noise_list = np.random.normal(0, noise_level, n_data)
y_list = b + w * x_list + noise_list

with open('data.txt', 'w+') as f:
    for i in range(100):
        f.write(f'{x_list[i]},{y_list[i]}\n')

# Visualization of data
plt.title('All data')
plt.xlabel('x')
plt.ylabel('y')
plt.scatter(x_list, y_list)
plt.savefig('./img/data')
# plt.show()
plt.close()