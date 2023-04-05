#!/usr/bin/env python
# coding: utf-8

# # Array-Oriented Programming with NumPy

# ### Python lists are essentially an array of pointers, 
# * Each pointing to a location that contains the information related to the element. 
# * This adds a lot of overhead in terms of memory and computation. 
# * And most of this information is rendered redundant when all the objects stored in the list are of the same type!

# ### NumPy arrays that contain only homogeneous elements, i.e. elements having the same data type.
# * NumPy is a Python library used for working with arrays
# * NumPy stands for Numerical Python
# * The array object in NumPy is called ndarray
# * Many popular data science libraries such as Pandas, SciPy (Scientific Python) and Keras (for deep learning) are built on or depend on NumPy

# In[ ]:


# installation of NumPy. If your already installed NumPy then ignore the step
#!pip install numpy
#Install pandas and matplotlib to work on a sample example


# In[ ]:


#!pip install pandas


# In[ ]:


#!pip install matplotlib


# ### Import Numpy

# In[ ]:


import numpy as np


# In[ ]:





# ### Why NumPy? What is the difference from Python List?
# Because of Python's dynamic typing, we can even create heterogeneous lists:
# ![image-2.png](attachment:image-2.png)
# - But this flexibility comes at a cost: to allow these flexible types, each item in the list must contain its own type info, reference count, and other information–that is, each item is a complete Python object. 
# - In the special case that all variables are of the same type, much of this information is redundant: it can be much more efficient to store data in a fixed-type array. 
# ![image-3.png](attachment:image-3.png)
# ref- https://colab.research.google.com/github/jakevdp/PythonDataScienceHandbook/blob/master/notebooks/02.01-Understanding-Data-Types.ipynb#scrollTo=uGMjhC3JAgzf

# ### Difference between Array and List
# ![image.png](attachment:image.png)

# ### Creating a numpy array from list

# In[ ]:





# In[ ]:





# In[ ]:





# ### Initial Placeholders
# ![image.png](attachment:image.png)
# - random.randint(low, high=None, size=None, dtype=int) low included high excluded

# In[ ]:


# Create a length-10 integer array filled with zeros


# In[ ]:


# Create a 3x5 floating-point array filled with ones


# In[ ]:


# Create a 3x5 array filled with 3.14 (constant)


# In[ ]:


# Create an array filled with a linear sequence
# Starting at 0, ending at 30, stepping by 2


# In[ ]:





# In[ ]:


# Create an array of ten values evenly spaced between 0 and 1


# In[ ]:


# Create a 3x3 array of uniformly distributed of random values between 0 and 1


# In[ ]:


# Create a 3x3 array of random integers in the interval (0, 10)


# In[ ]:


# Create a 3x3 identity matrix


# In[ ]:


# Create an uninitialized array of three integers
# The values will be whatever happens to already exist at that memory location
# it requires the user to manually set all the values in the array, and should be used with caution.


# In[ ]:





# In[ ]:





# ### NumPy Standard Data Types
# ![image.png](attachment:image.png)

# ###  Inspection of Numpy array 
# ![image.png](attachment:image.png)

# In[ ]:


# seed for reproducibility


# In[ ]:


# Create a One-dimensional array of numbers between 0 to 10 with 6 elements


# In[ ]:





# In[ ]:


# Create a two-dimensional array of numbers between 0 to 10 of size 3x4


# In[ ]:


# Create a three-dimensional array of size 3x4x5 


# In[ ]:





# ### Array Indexing: Accessing Single Elements (Similar to lists)

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ### Array Slicing: Accessing Subarrays `x[start:stop:step]`

# In[ ]:


# first five elements


# In[ ]:


# elements after index 5


# In[ ]:


# middle sub-array


# In[ ]:


# every other element, starting at index 1


# In[ ]:


# all elements, reversed,In this case, the defaults for start and stop are swapped. 


# In[ ]:


# reversed every other from index 5


# In[ ]:


# first two rows, three columns


# In[ ]:


# first column of x2


# In[ ]:


# second column of x2


# In[ ]:


# first row of x2


# In[ ]:


# Second row of x2


# ### Reshaping of Arrays

# In[ ]:


x4=np.arange(10)
print(x4)


# In[ ]:


#Reshape did not change the underlaying array
x4


# ### resize
# resize() will change the underlaying  array.

# In[ ]:


x4.resize(2,5)


# In[ ]:


x4


# #### Concatenation of arrays
# - np.concatenate
# - np.vstack
# - np.hstack

# In[ ]:


np.random.seed(0)
x=np.random.randint(10, size= (2,3))
y=np.random.randint(10, size= (2,3))
print(x)
print(y)


# In[ ]:


# concatenate can work on more than two arrays and 2 d arrays
array1=np.concatenate([x,y])
print(array1)


# In[ ]:


# vertically stack the arrays
array2=np.vstack([x,y])
print([x])
print([y])


# In[ ]:


# horizontally stack the arrays
array3=np.hstack([x,y])
print([x])
print([y])


# ### Splitting of arrays
# - np.split 
# - np.hsplit 
# - np.vsplit

# In[ ]:


a,b,c,d,e=np.vsplit(array2,5)
e


# In[ ]:


print(array3)
a,b,c,d,e,f,g=np.hsplit(array3,7)
e


# In[ ]:





# In[ ]:





# In[ ]:





# ## NumPy Arithmatic Operation
# ![image.png](attachment:image.png)

# In[ ]:


import numpy as np
x = np.arange(4)
print("x     =", x)
print("x + 5 =", x + 5)
print("x - 5 =", x - 5)
print("x * 2 =", x * 2)
print("x / 2 =", x / 2)
print("x // 2 =", x // 2)  # floor division
print("x ** 2 = ", x ** 2)
print("x % 2  = ", x % 2)


# In[ ]:


theta = np.linspace(0, np.pi, 3)
print("theta", theta)
print("sin(theta) = ", np.sin(theta))
print("cos(theta) = ", np.cos(theta))
print("tan(theta) = ", np.tan(theta))


# In[ ]:


x = [-1, 0, 1]
print("x         = ", x)
print("arcsin(x) = ", np.arcsin(x))
print("arccos(x) = ", np.arccos(x))
print("arctan(x) = ", np.arctan(x))


# In[ ]:


x = [1, 2, 4, 10]
print("x        =", x)
print("ln(x)    =", np.log(x))
print("log2(x)  =", np.log2(x))
print("log10(x) =", np.log10(x))


# In[ ]:


np.add(x,4)


# ### Aggregation functions
# ![image.png](attachment:image.png)

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ### Example: What is the Average Height of US Presidents?

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[ ]:


data=pd.read_csv('president_heights.csv')


# In[ ]:


data.head()


# In[ ]:


heights=np.array(data['height(cm)'])


# In[ ]:


data.sample(10)


# In[ ]:


print("Mean height",round(heights.mean()))


# ### Comparing Arrays
# And we can compare arrays. This will result in an array that is contained with booleans.
# ![image.png](attachment:image.png)

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ### Working with Boolean Arrays
# - When the axis is set to 0. the function collapsed the columns
# - When the axis is set to 1. the function collapsed the rows

# In[ ]:


np.random.seed(10)
x = np.random.randint(10, size=(3, 4))


# In[ ]:


# how many values less than 6?


# In[ ]:


# how many values less than 6 in each row?


# In[ ]:


# how many values less than 6 in each column?


# In[ ]:


# are there any values greater than 8?


# In[ ]:


# are all values less than 10?


# In[ ]:


# are all values in each row less than 8?


# ### Array broadcasting
# https://numpy.org/doc/stable/user/basics.broadcasting.html
# - The term broadcasting describes how NumPy treats arrays with different shapes during arithmetic operations. 
# - Subject to certain constraints, the smaller array is “broadcast” across the larger array so that they have compatible shapes.
# - Broadcasting provides a means of vectorizing array operations so that looping occurs in C instead of Python. 

# In[ ]:


a = np.array([1.0, 2.0, 3.0])
b = 2.0
a * b


# ![image.png](attachment:image.png)

# #### General Broadcasting Rules
# When operating on two arrays, NumPy compares their shapes element-wise. It starts with the trailing (i.e. rightmost) dimension and works its way left. Two dimensions are compatible when
# 
# - they are equal, or
# - one of them is 1.

# In[ ]:


a = np.array([[ 0.0,  0.0,  0.0],
              [10.0, 10.0, 10.0],
              [20.0, 20.0, 20.0],
              [30.0, 30.0, 30.0]])

b = np.array([1.0, 2.0, 3.0])
a + b


# ![image.png](attachment:image.png)

# In[ ]:


b = np.array([1.0, 2.0, 3.0, 4.0])
#a + b


# ![image.png](attachment:image.png)

# ### Sorting Arrays

# In[ ]:


import numpy as np
np.random.seed(25)
X = np.random.randint(10, size=(4, 6))
print(X)


# In[ ]:


# sort each column of X


# In[ ]:


# sort each row


# In[ ]:


#Returns the indices that would sort an array.


# ## Saving/Loading Arrays
# - np.save()
# - np.load()

# In[ ]:





# In[ ]:





# In[ ]:




