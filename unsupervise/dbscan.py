import numpy as np
import pandas as pd
from sklearn import datasets


X, _ = datasets.make_moons(500, noise=0.1, random_state=1)
df = pd.DataFrame(X, columns=["feature1", "feature2"])

df.plot.scatter("feature1", "feature2", s=100, alpha=0.6, title="dataset by make_moon")


