import numpy as np
from pylab import plt
from raindisagg_gan_pretrained import generate_scenarios, plot_scenarios

ndomain = 16 #the domain used in the traing of the GAN. must be the same here
# create made-up input conditions (including empty last channel dimension) with 10mm/day at every gridpoint
# in a real application, here you would use your own data  (in mm/h).
cond1 = 10 * np.ones((ndomain, ndomain, 1))
# generate subdaily scenarios
n_scenarios = 10
scenarios1 = generate_scenarios(cond1, n_scenarios)
# plot the results
fig = plot_scenarios(scenarios1)
plt.savefig('generated_scenarios1.png')