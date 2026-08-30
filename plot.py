import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn

class ColorMap(object):
    def __init__(self,colors=None):
        if(colors is None):
            colors=['lime','red','blue','tomato',
                    'orange','skyblue','peachpuff',
                    'yellow','black' ]
        self.colors=colors

    def __len__(self):
        return len(self.colors)
    
    def __call__(self,i):
        return self.colors[i % len(self)]

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

def text_plot( x,
               y,
               label,
               title):
    for i,label_i in enumerate(label):
        plt.text(x[i],y[i],label_i)
    plt.xlim(compute_lim(x))
    plt.ylim(compute_lim(y)) 
    plt.title(title)
    plt.show()

def adno_plot( x,
               y, 
               label,
               title,               
               color_map=None):
    if(color_map is None):
        color_map=ColorMap()
    for i,label_i in enumerate(label):
        plt.annotate( label_i,
                      (x[i],y[i]),
                      color=color_map(label_i))
    plt.xlim(compute_lim(x))
    plt.ylim(compute_lim(y)) 
    plt.title(title)
    plt.show()
    
def compute_lim(x):
    delta=0.25*np.min(np.abs(x))
    return [min(x)-delta,max(x)+delta]

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