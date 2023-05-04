import numpy as np 
import pandas as pd


class model:
    def __init__(self,vocab_size,vector_size) -> None:
        self.w1 = np.ones((vocab_size,vector_size))
        self.w2 = np.ones((vector_size,vocab_size))