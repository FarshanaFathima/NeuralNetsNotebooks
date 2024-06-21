import unicodedata
from string import ascii_letters
import torch

class Preprocess_Examples():

    def __init__(self):
        self.vocabulary = {}
        self.all_letters = ascii_letters+ " .,:;-'"
        self.generate_vocab()
        

        
    def unicodeToAscii(self, s):
        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn'
            and c in self.all_letters
        )
    
    def generate_vocab(self):
        """
        Assign a number to every ascii character
        """
        print(self.all_letters)
        for id_, char in enumerate(self.all_letters):
            self.vocabulary[char] = id_
    
    def name2tensor(self, name:str):
        """
        converts the example/name to 
        a tensor
        """
        p_name = torch.zeros(len(name), 1, len(self.vocabulary))
        name = self.unicodeToAscii(name)
        for i, char in enumerate(name):
            p_name[i][0][self.vocabulary[char]] = 1
        return p_name

obj = Preprocess_Examples()
# obj.generate_vocab()
# p_name = obj.decode_name("Farshana")
# p_name = obj.name2tensor(p_name)
print(obj.all_letters)