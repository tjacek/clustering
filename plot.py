import matplotlib.pyplot as plt
import seaborn as sn

def scatter( x, 
             y, 
             title,
	         xlabel="x",
	         ylabel="y",
             out_path=None):
    if(x is None):
        x=range(len(y))
    plt.scatter(x, y, alpha=0.5)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(alpha=0.7)
    if(out_path):
        out_i=f"{out_path}/{title}.png"
        plt.tight_layout()
        plt.savefig(out_i,dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

def show_heatmap( matrix,
                  title,
                  x_axis='auto',
                  y_axis='auto'):
    sn.heatmap( matrix,
                cmap="YlGnBu",
                annot=False,
                xticklabels=x_axis,
                yticklabels=y_axis)
    plt.title(title)
    plt.tight_layout()
    plt.show()