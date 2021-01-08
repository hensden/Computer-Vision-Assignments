import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import pickle as pk

def plotter(ab, img, type):
	f = open(img + '_' + type + 'ab.pkl', 'wb')
	pk.dump(ab, f)
	f.close()
	'''
	X = []
	Y = [] 
	Z = []
	for i in range(0, len(ab), step):
		X.append(ab[i*step:i*step+step,0])
		Y.append(ab[i*step:i*step+step,1])
		Z.append(ab[i*step:i*step+step,2])

	fig = plt.figure()
	ax = plt.axes(projection='3d')
	ax.set_xlabel('alpha')
	ax.set_ylabel('beta')
	ax.set_zlabel('recall')
	color = ['red','green','blue','yellow','gray']
	j = 0
	for i in range(len(X)):
		ax.plot3D(X[i],Y[i], Z[i], color[j])
		j=(j+1)%5
	fig.show()
	plt.savefig('result_' + type + '.png')
	'''