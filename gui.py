import tkinter as tk
from tkinter import messagebox
import argparse
import snapshot
import plot 
import reduct

class ClusterGui:
    def __init__( self,
                  root,
                  by_labels):
        self.root = root
        self.root.title("Frame Clusters")
        self.root.geometry("600x350")
        self.root.resizable(False, False)
        self.by_labels = by_labels
        self.selected_cluster = None
        self.selected_label = None
        
#       title_label = tk.Label(root, text="Clusters:", font=("Arial", 12))
#       title_label.pack(pady=10)
        self.lists_container = tk.Frame(root)
        self.lists_container.pack(pady=5)

        self.cluster_list = self.create_listbox( self.by_labels.names(),
                                                 side=tk.LEFT, 
                                                 on_select=self.set_selected_cluster)
        self.label_list = self.create_listbox( self.by_labels.info_types(),
                                               side=tk.LEFT,
                                               on_select=self.set_selected_label)
    
    def create_listbox( self, 
                        items, 
                        side, 
                        on_select=None):
        frame = tk.Frame(self.lists_container)
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
    
    def set_selected_label(self, event):
        selected_indices = self.label_list.curselection()
        if selected_indices:
            self.selected_label = self.label_list.get(selected_indices[0]) 

class MissingFieldGuard(dict):
    def __call__(self,obj):
        for attr_i,msg_i in self.items():
            value_i=getattr(obj,attr_i)
            if not value_i:
                messagebox.showwarning("Uwaga", msg_i)
                return True
        return False

class ReductionGui(ClusterGui):
    def __init__( self,
                  root,
                  by_labels):
        super(ReductionGui, self).__init__( root,
                                            by_labels)
        self.selected_alg = None
        self.algs_list = self.create_listbox( reduct.ALGS.keys(),
                                              side=tk.RIGHT, 
                                              on_select=self.set_selected_alg)
        self.field_guard=MissingFieldGuard({"selected_cluster": "Nie wybrano klastra!",
                                            "selected_label":"Nie wybrano etykiet!",
                                            "selected_alg":"Nie wybrano algorytmu redukcji!"})
        confirm_button = tk.Button(root, text="Show cluster", command=self.confirm_selection)
        confirm_button.pack(pady=10)

    def set_selected_alg(self, event):
        selected_indices = self.algs_list.curselection()
        if selected_indices:
            self.selected_alg = self.algs_list.get(selected_indices[0])
    
    def confirm_selection(self):
        if(self.field_guard(self)):
            return
        data_i = self.by_labels[self.selected_cluster]
        label_i = data_i.__dict__[self.selected_label]
        cat_i = data_i.__dict__["cat"]

        reduct_func = reduct.ALGS[self.selected_alg]
        X_reduced = reduct_func(data_i.frames)
        
        plot.adno_plot(x=X_reduced[:, 0],
                       y=X_reduced[:, 1],
                       label=cat_i,
                       color=label_i,
                       title=f"{self.selected_cluster} ({self.selected_alg})")

class HisogramGui(ClusterGui):
    def __init__( self,
                  root,
                  by_labels):
        super(HisogramGui, self).__init__( root,
                                            by_labels)
        self.field_guard=MissingFieldGuard({"selected_cluster": "Nie wybrano klastra!",
                                            "selected_label":"Nie wybrano etykiet!"})
        confirm_button = tk.Button( root, 
                                    text="Show hisogram", 
                                    command=self.confirm_selection)
        confirm_button.pack(pady=10)
    
    def confirm_selection(self):
        if(self.field_guard(self)):
            return
        data_i = self.by_labels[self.selected_cluster]
        desc_i=data_i[self.selected_label]

        plot.hist( desc_i,
                   value=self.selected_label,
                   title=self.selected_cluster)
#        raise Exception(label_i)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cls_path", type=str,default="MSR/ae/layer_1/spectral_36")
    parser.add_argument("--seq_path", type=str,default="MSR/ae/layer_1/seqs")
    parser.add_argument("--cmd", type=str,default="hist")
    args=parser.parse_args()
    by_labels=snapshot.GroupedClust.make(args.cls_path,
    	                                 args.seq_path)
    root = tk.Tk()
    if(args.cmd=="redu"):
        app = ReductionGui( root,
    	                    by_labels)
    if(args.cmd=="hist"):
        app = HisogramGui( root,
                           by_labels)
    root.mainloop()