import tkinter as tk
from tkinter import messagebox
import argparse
import snapshot
import plot 
import reduct

class GuiApp:
    def __init__(self,
                 root,
                 by_labels):
        self.root = root
        self.root.title("Frame Clusters")
        self.root.geometry("600x350")
        self.root.resizable(False, False)
        self.by_labels = by_labels
        self.current_data = None
        self.selected_cluster = None
        self.selected_alg = None
 
        title_label = tk.Label(root, text="Clusters:", font=("Arial", 12))
        title_label.pack(pady=10)
        
        lists_container = tk.Frame(root)
        lists_container.pack(pady=5)

        self.init_cluster_list(root,lists_container)
        self.init_alg_list(root,lists_container)
        self.init_labels_list(root,lists_container)

        confirm_button = tk.Button(root, text="Show cluster", command=self.confirm_selection)
        confirm_button.pack(pady=10)

    def init_cluster_list(self,root,lists_container):
 
        cluster_frame = tk.Frame(lists_container)
        cluster_frame.pack(side=tk.LEFT, padx=10)
 
        scrollbar = tk.Scrollbar(cluster_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.cluster_list = tk.Listbox(
            cluster_frame,
            width=25,
            height=12,
            selectmode=tk.SINGLE,
            yscrollcommand=scrollbar.set,
            exportselection=False,
            font=("Arial", 11)
        )
 
        for item in self.by_labels.names():
            self.cluster_list.insert(tk.END, item)
 
        self.cluster_list.pack(side=tk.LEFT)
        scrollbar.config(command=self.cluster_list.yview)
 
        self.cluster_list.bind("<<ListboxSelect>>", self.set_selected_cluster)

    def init_alg_list(self,root,lists_container):
        algs_frame = tk.Frame(lists_container)
        algs_frame.pack(side=tk.LEFT, padx=10)
 
        algs_scrollbar = tk.Scrollbar(algs_frame)
        algs_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
 
        self.algs_list = tk.Listbox(
            algs_frame,
            width=25,
            height=12,
            selectmode=tk.SINGLE,
            yscrollcommand=algs_scrollbar.set,
            exportselection=False,
            font=("Arial", 11)
        )
 
        for item in reduct.ALGS.keys():
            self.algs_list.insert(tk.END, item)
 
        self.algs_list.pack(side=tk.LEFT)
        algs_scrollbar.config(command=self.algs_list.yview)
 
        self.algs_list.bind("<<ListboxSelect>>", self.set_selected_alg)
    
    def init_labels_list(self,root,lists_container):
        label_frame = tk.Frame(lists_container)
        label_frame.pack(side=tk.BOTTOM, padx=10)
        
        label_scrollbar = tk.Scrollbar(label_frame)
        label_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
       
        self.label_list = tk.Listbox(
            label_frame,
            width=25,
            height=12,
            selectmode=tk.SINGLE,
            yscrollcommand=label_scrollbar.set,
            exportselection=False,
            font=("Arial", 11)
        )
        
        for item in self.by_labels.info_types():
            self.label_list.insert(tk.END, item)
        self.label_list.pack(side=tk.BOTTOM)
        label_scrollbar.config(command=self.label_list.yview)

    def set_selected_cluster(self, event):
        selected_indices = self.cluster_list.curselection()
        if selected_indices:
            self.selected_cluster = self.cluster_list.get(selected_indices[0])
 
    def set_selected_alg(self, event):
        selected_indices = self.algs_list.curselection()
        if selected_indices:
            self.selected_alg = self.algs_list.get(selected_indices[0])
 
    def confirm_selection(self):
        if not self.selected_cluster:
            messagebox.showwarning("Uwaga", "Nie wybrano żadnego elementu z listy klastrów!")
            return
 
        if not self.selected_alg:
            messagebox.showwarning("Uwaga", "Nie wybrano żadnego algorytmu redukcji!")
            return
 
        data_i = self.by_labels[self.selected_cluster]
        reduct_func = reduct.ALGS[self.selected_alg]
        X_reduced = reduct_func(data_i)
 
        plot.adno_plot(x=X_reduced[:, 0],
                       y=X_reduced[:, 1],
                       label=data_i.y,
                       title=f"{self.selected_cluster} ({self.selected_alg})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cls_path", type=str,default="MSR/ae/layer_1/spectral_36")
    parser.add_argument("--seq_path", type=str,default="MSR/ae/layer_1/seqs")
    args=parser.parse_args()
    by_labels=snapshot.GroupedClust.make(args.cls_path,
    	                                 args.seq_path)
    root = tk.Tk()
    app = GuiApp( root,
    	          by_labels)
    root.mainloop()