import os
import torch

class Preprocess_Labels():
    """
    Preprocess the dataset to a form that can be usable by RNN
    """
    def __init__(self, data:str) -> None:
        self.data = data
        pass

    def get_labels(self):
        """
        Obtain labels from the dataset provided
        """
        labels = [file_name.split(".")[0] for file_name in os.listdir(self.data)]
        return labels
    
    def encode_labels(self, labels:list)->dict:
        """
        Convert the labels into torch tensors
        """
        lang2label = {}
        for i, label in enumerate(labels):
            lang2label[label] = torch.tensor([i], dtype=torch.long)
        return lang2label
    
    def main(self):
        lbl_list = self.get_labels()
        return self.encode_labels(lbl_list)
# path = r"/Users/farshanafathima/Documents/playground/NeuralNetsNotebooks/data/names"
# obj = Preprocess_Labels(path)
# lbl = obj.get_labels()
# lbl_dict = obj.encode_labels(lbl)
# print(lbl_dict)