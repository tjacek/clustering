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
        self.selected_label = None

        title_label = tk.Label(root, text="Clusters:", font=("Arial", 12))
        title_label.pack(pady=10)
        
        lists_container = tk.Frame(root)
        lists_container.pack(pady=5)
 
        self.cluster_list = self.create_listbox( lists_container, 
                                                 self.by_labels.names(),
                                                 side=tk.LEFT, 
                                                 on_select=self.set_selected_cluster
                                                )
        self.algs_list = self.create_listbox( lists_container, 
                                              reduct.ALGS.keys(),
                                              side=tk.LEFT, 
                                              on_select=self.set_selected_alg
                                            )
        self.label_list = self.create_listbox( lists_container, 
                                               self.by_labels.info_types(),
                                               side=tk.BOTTOM,
                                               on_select=self.set_selected_label
                                             )

        confirm_button = tk.Button(root, text="Show cluster", command=self.confirm_selection)
        confirm_button.pack(pady=10)

    def create_listbox(self, parent, items, side, on_select=None):
        frame = tk.Frame(parent)
        frame.pack(side=side, padx=10)
 
        scrollbar = tk.Scrollbar(frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
 
        listbox = tk.Listbox(
            frame,
            width=25,
            height=12,
            selectmode=tk.SINGLE,
            yscrollcommand=scrollbar.set,
            exportselection=False,
            font=("Arial", 11)
        )
        for item in items:
            listbox.insert(tk.END, item)
 
        listbox.pack(side=side)
        scrollbar.config(command=listbox.yview)
 
        if on_select:
            listbox.bind("<<ListboxSelect>>", on_select)
 
        return listbox

    def set_selected_cluster(self, event):
        selected_indices = self.cluster_list.curselection()
        if selected_indices:
            self.selected_cluster = self.cluster_list.get(selected_indices[0])
 
    def set_selected_alg(self, event):
        selected_indices = self.algs_list.curselection()
        if selected_indices:
            self.selected_alg = self.algs_list.get(selected_indices[0])
    
    def set_selected_label(self, event):
        selected_indices = self.label_list.curselection()
        if selected_indices:
            self.selected_label = self.label_list.get(selected_indices[0]) 
    
    def confirm_selection(self):
        if not self.selected_cluster:
            messagebox.showwarning("Uwaga", "Nie wybrano klastra!")
            return
        if not self.selected_alg:
            messagebox.showwarning("Uwaga", "Nie wybrano algorytmu redukcji!")
            return
        if not self.selected_label:
            messagebox.showwarning("Uwaga", "Nie wybrano etykiet!")
            return
        data_i = self.by_labels[self.selected_cluster]
        label_i = data_i.__dict__[self.selected_label]

        reduct_func = reduct.ALGS[self.selected_alg]
        X_reduced = reduct_func(data_i.frames)
 
        plot.adno_plot(x=X_reduced[:, 0],
                       y=X_reduced[:, 1],
                       label=label_i,
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