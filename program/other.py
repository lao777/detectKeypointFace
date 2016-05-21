import matplotlib.pyplot as plt
import cPickle as pickle


def DrawImg(X, y, isDraw = True):
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(1, 1, 1)
    img = X.reshape(96, 96)
    ax.imshow(img, cmap='gray')
    ax.scatter(y[0::2] * 48 + 48, y[1::2] * 48 + 48, marker='x', s=10)
    if (isDraw == True):
    	plt.show()
    else:
    	plt.savefig('./img/' +str(i)+'.png')


def SaveNet(net):
	with open('net.pickle', 'wb') as f:
	    pickle.dump(net, f, -1)


def LoadNet():
	with open('net.pickle', 'rb') as f:
	    return pickle.load(f)