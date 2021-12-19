import numpy as np

a = np.array(range(1,11))
size = 5

def split_x(dataset, size):
    aaa = []
    for i in range(len(dataset) - size + 1):     # 반복횟수
        subset = dataset[i : (i +size)]        # slicing
        aaa.append(subset)
    return np.array(aaa)

dataset = split_x(a, size)       
print(dataset)



 
x = dataset[:, :4]  # 행, 열 
y = dataset[:, 4]   # 행, 열
print(x,y) 
print(x.shape, y.shape) # (6, 4) (6,)