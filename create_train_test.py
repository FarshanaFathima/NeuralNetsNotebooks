import os
from preprocess_examples import Preprocess_Examples
from preprocess_labels import Preprocess_Labels
from sklearn.model_selection import train_test_split

class Dataset():
    def __init__(self, path:str) -> None:
        self.path = path
        self.all_names = []
        self.all_labels = []
        self.ex_obj = Preprocess_Examples()
        print(self.ex_obj.vocabulary)
        self.lbl_obj = Preprocess_Labels(path)
        self.lang2label = self.lbl_obj.main()

        
    def create_dataset(self, ):
        for file in os.listdir(self.path):
            lang = file.split(".")[0]
            with open(os.path.join(self.path,file)) as f:
                name_list = [name.replace("\n", "") for name in f.readlines()]
                for name in name_list:
                    self.all_names.append(self.ex_obj.name2tensor(name))
                    self.all_labels.append(self.lang2label[lang])

    def split_data(self, examples:list, labels:list, test_size = 0.1):
        """
        Split dataset into train and test data
        """
        train_ind, test_ind = train_test_split(range(len(labels)), test_size=test_size, shuffle=True, stratify=labels)
        train_data = [(examples[i], labels[i])for i in train_ind]
        test_data = [(examples[i], labels[i]) for i in test_ind]
        return train_data, test_data

    
    
