import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from illustris_python import illustris_python as il
import time
from joblib import Parallel, delayed
import numpy as np
import matplotlib.pyplot as plt

def addDF(snap):
    df = pd.DataFrame([1, 2, 3, 4])
    dfs[snap] = df
    print(dfs)
    
    
    
dfs = {}

Parallel(n_jobs = 12)(delayed(addDF)(snap) for snap in range(0,24))

print(dfs)