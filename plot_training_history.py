


from pylab import plt
import pandas as pd
import numpy as np

ifile='plots_wgancp_pixelnorm/hist.csv'

df = pd.read_csv(ifile, index_col=0)

n_epochs = 50

# compute mean over epochs
n_per_epoch = int(len(df) /n_epochs)

groups = pd.Series(np.repeat(range(int(len(df)/n_per_epoch)), n_per_epoch))

df_mean = df.groupby(groups).mean()


plt.figure()
df_mean.plot()

df_rollmean = df.rolling(10000).mean()
plt.figure()
df_rollmean.plot()


