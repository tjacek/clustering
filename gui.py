import tkinter as tk
from tkinter import messagebox
import argparse
import snapshot

class GuiApp:
    def __init__( self, 
    	          root,
    	          by_labels):
        self.root = root
        self.root.title("Frame Clusters")
        self.root.geometry("400x400")
        self.root.resizable(False, False)
        self.by_labels=by_labels
        title_label = tk.Label(root, text="Clusters:", font=("Arial", 12))
        title_label.pack(pady=10)
 
        list_frame = tk.Frame(root)
        list_frame.pack(pady=5)
 
        scrollbar = tk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
 
        self.listbox = tk.Listbox(
            list_frame,
            width=30,
            height=10,
            selectmode=tk.SINGLE,
            yscrollcommand=scrollbar.set,
            font=("Arial", 11)
        )
 
        for item in self.by_labels.names():
            self.listbox.insert(tk.END, item)
 
        self.listbox.pack(side=tk.LEFT)
        scrollbar.config(command=self.listbox.yview)
  
        confirm_button = tk.Button(root, text="Show cluster", command=self.confirm_selection)
        confirm_button.pack(pady=10)
 
    def confirm_selection(self):
        selected_indices = self.listbox.curselection()
        if selected_indices:
            selected_value = self.listbox.get(selected_indices[0])
            messagebox.showinfo("Potwierdzenie", f"Zatwierdzono wybór: {selected_value}")
        else:
            messagebox.showwarning("Uwaga", "Nie wybrano żadnego elementu z listy!")
 
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