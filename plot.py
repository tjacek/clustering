import matplotlib.pyplot as plt
import seaborn as sn

def scatter( x, y, title,
	         xlabel="x",
	         ylabel="y"):
    plt.scatter(x, y, alpha=0.5)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(alpha=0.7)
    plt.show()

def show_heatmap(matrix,title):
    sn.heatmap( matrix,
                cmap="YlGnBu",
                annot=False)#,
    plt.title(title)
    plt.tight_layout()
    plt.show()