import numpy as np
arr = np.array([1, 2, 3, 4, 5])

print(arr)
print(type(arr))
print(np.__version__)

arr = np.array([[1, 2, 3, 4], [4, 5, 6, 7]])
print(arr)

# 3D array
arr = np.array([[[1, 2, 3, 4], [4, 5, 6, 7]], [[1, 2, 3, 4], [4, 5, 6, 7]]])
print(arr)

# Create a higher dimension array
arr = np.array([1, 2, 3, 4], ndmin=5)
print('number of dimensions:', arr.ndim)
print(arr)


arr = np.array([1, 2, 3, 4, 5])
print(arr[0])  # get the first element
print(arr[1])  # get the second element


arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
print('print element first row second column', arr[0, 1])
print('print element second row second column', arr[1, 1])


arr = np.array([[[1, 2, 3, 4], [5, 6, 7, 8]], [[1, 2, 3, 4], [4, 5, 6, 7]]])
# access the third element of the second array of the first array
print(arr[0, 1, 2])


arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
print('print from the end of the array', arr[0, -1])
print('print from the end of the array', arr[1, -2])

# Slicing arrays
arr = np.array([1, 2, 3, 4, 5, 6, 7])
print(arr[1:5])
# notice that it EXCLUDES the end index
print(arr[4:])
print(arr[:4])
print(arr[-3:])
print(arr[-3:-1])
