import numpy as np

# a = np.arange(0,17,1,size=(2,2,4))
a = np.array([[[1,2,3], [4,5,6]],[[7,6,0],[1,4,5]]])
# # a = np.random.random((2,2,4))
# b = np.sum(np.sum(a,axis=1),axis=0)
# # c = a/np.expand_dims(b,axis=2)
# print(a,'\n\n',b)
# c  = a/b
# size = a.shape
# b = np.empty((3,size[0],size[1],size[2]))
# b[0] = a
# print(b)

for i in range(8):
	a = np.where(a==i,'{:03b}'.format(a),a)

print(a)