import matplotlib.pyplot as plt

def scatter( x, y, title,
	         xlabel="x",
	         ylabel="y"):
    plt.scatter(x, y, alpha=0.5)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(alpha=0.7)
    plt.show()
