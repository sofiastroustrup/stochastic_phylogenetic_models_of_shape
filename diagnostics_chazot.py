#!/usr/bin/env python
# coding: utf-8
# script for assessing convergence, can be used for plotting of rank test seed convergence 

# In[4]:
# import modules
import os
import numpy as np
import pandas as pd
import matplotlib.backends.backend_pdf as backend_pdf
import matplotlib.pyplot as plt
import arviz
import seaborn as sns
import matplotlib
from ete3 import Tree
import argparse
from scipy.stats import mode
import arviz as az
import gc


# In[5]:
parser = argparse.ArgumentParser(description='Arguments for script')
parser.add_argument('-folder_runs', help = 'folder for plots', nargs='?', type=str)
parser.add_argument('-MCMC_iter', help = '', nargs='?', type=int)
parser.add_argument('-burnin', help = '', nargs='?', type=int)
parser.add_argument('-nnodes', help = '', nargs='?', type=int, default=59)
parser.add_argument('-levelorder_tree', help = '', nargs='?', type=str, default = 'chazot_full_tree_levelorder.nw')
parser.add_argument('-nxd', help = '', nargs='?', type=int, default=40)

args = parser.parse_args()

#%%
MCMC_iter = args.MCMC_iter
burnin = args.burnin
nthin = 1 # see from script/running conditions, not used for plotting
folder_runs = args.folder_runs +'/' #"BM9/runs/3513656273068705/" #
folder_simdata = args.folder_simdata +'/' #'BM9/simdata/3513656273068705/' #args.folder_simdata +'/'
nnodes = args.nnodes
levelorder_tree = args.levelorder_tree
nxd = args.nxd
thresh_kdeplot_2D = 0.05 # probability mass below last contour line

#%%

MCMC_iter = 3000
burnin = 1500
nthin = 1 # see from script/running conditions, not used for plotting
folder_runs = 'chazot/runs_rb=10_ov=0.01/'
folder_simdata = 'chazot/data/' #'BM9/simdata/3513656273068705/' #args.folder_simdata +'/'
nnodes = 59
nxd=40
levelorder_tree = 'chazot/data/chazot_full_tree_levelorder.nw'

#%%
pars_name = ['kalpha', 'gtheta']
rep_path = len(pars_name)+1
chains = os.listdir(folder_runs) # use all chains in data seed folder 
chains = [c for c in chains if c[0] not in ['_', '.']] # remove files starting with underscore
print(chains)

# In[6]:
outputfolder = folder_runs
levelorder_tree = Tree(levelorder_tree)#Tree('chazot_subtree_levelorder.nw') #Tree('chazot_full_tree_levelorder.nw')
temp_name = ['' for i in range(len(chains))]#['runs_'+chains[i]+'_' for i in range(len(chains))] #[outputfolder+chains[i]+'/' for i in range(len(chains))] # #['runs_'+chains[i]+'_' for i in range(len(chains))] #['runs_'+chains[i]+'_' for i in range(len(chains))] #
path = outputfolder+'_*'+'-'.join(chains)
if not os.path.isdir(path): 
    os.mkdir(path)

# In[7]:
leafidx = []
for leaf in levelorder_tree:
    leafidx.append(leaf.name)
print(leafidx)

# get innernode idx 
# nnodes: small tree = 9, large tree = 59
leafidxint = [int(leafidx[i]) for i in range(len(leafidx))]
nodes = list(range(nnodes))
inneridx = set(nodes)-set(leafidxint)
inneridx

# In[9]:
raw_trees = [np.genfromtxt(folder_runs + chains[i]+'/'+temp_name[i]+"tree_nodes.csv", delimiter = ",") for i in range(len(chains))]
tree_counters = [np.genfromtxt(folder_runs + chains[i]+'/'+temp_name[i]+"tree_counter.csv", delimiter = ",").astype(int) for i in range(len(chains))]
flat_trees_raw = [raw_trees[i].reshape(len(tree_counters[i]),nnodes,nxd) for i in range(len(raw_trees))]
#flat_true_tree = np.genfromtxt(folder_simdata+"flat_true_tree.csv", delimiter = ",")
super_root = [np.genfromtxt(folder_runs + chains[i]+'/'+temp_name[i]+"inference_root_start.csv", delimiter = ",") for i in range(len(chains))]
_super_root = [np.concatenate((super_root[i], super_root[i][0:2])) for i in range(len(chains))]
_super_root = np.unique(np.array(_super_root), axis=0)

# In[12]:
flat_trees = np.array([np.repeat(flat_trees_raw[i], tree_counters[i], axis=0)[burnin*rep_path:(MCMC_iter//nthin)*rep_path] for i in range(len(flat_trees_raw))])
flat_trees.shape #nchains x MCMC_iter x nnodes x nxd


# In[14]:
# get rhat and ESS for all nodes and dimensions
rhats = []
esss = []
for idx in range(flat_trees.shape[2]):  # calculate for all nodes 
    innernodes = flat_trees[:,:,idx, :]
    keys = list(range(innernodes.shape[2]))
    MCMCres = arviz.convert_to_dataset({k:innernodes[:,:,i] for i,k in enumerate(keys)})
    rhats.append(arviz.rhat(MCMCres).to_array().to_numpy())
    esss.append(arviz.ess(MCMCres).to_array().to_numpy())

# save rhat for plotting
np.savetxt(path+'/'+"rhats_paths.csv",np.array(rhats), delimiter=",")


# In[15]:
# PLOT POSTERIOR BY SAMPLING FROM POSTERIOR 
sample_n = 50
colors = sns.color_palette('pastel', len(chains))
pdf = backend_pdf.PdfPages(path + f'/samples-posterior-sample_n={sample_n}_burnin={burnin}_MCMCiter={MCMC_iter}.pdf')
plt.figure(1);
for idx in inneridx: # loop over i innernodes
    print(idx)
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(5,5), sharex=False);
    innernodes = flat_trees[:,:,idx,:].reshape(-1, nxd)[::sample_n,:]
    inode = np.append(innernodes, innernodes[:,0:2],1)
    for i in range(inode.shape[0]):
        axes.plot(inode[i,::2], inode[i,1::2], '--.', color='steelblue', alpha=0.3, label='true shape')
    fig.suptitle(f'Node {idx}', size=40);
    fig.tight_layout();
    pdf.savefig();
    plt.clf();
pdf.close();

#%%
# PLOT TRACES FOR ALL DIMENSIONS OF ALL INNERNODES
colors = sns.color_palette('pastel', len(chains))
pdf = backend_pdf.PdfPages(path + f'/trace-innernodes_burnin={burnin}_MCMCiter={MCMC_iter}.pdf')
plt.figure(1)
#for idx in range(flat_trees.shape[2]): # loop over nodes
for idx in inneridx: # loop over innernodes
    fig, axes = plt.subplots(nrows=7, ncols=6, figsize=(25,15), sharex=True)
    for j in range(flat_trees.shape[0]): # loop over chains
        innernode = flat_trees[j,:,idx, :]
        #true_innernode = flat_true_tree[idx,:]
        curcol = colors[j]
        for i, ax in zip(range(innernode.shape[1]), axes.flat): # loop over dimensions
            ax.plot(innernode[:,i], color = curcol, alpha=0.5)
            #ax.hlines(y=true_innernode[i], xmin=0, xmax=innernode.shape[0], color='skyblue')
            cur_ess = esss[idx][i]#round(arviz.ess(innernode[:,i]),2)
            cur_rhat = rhats[idx][i]
            ax.set_title(f'{i}, Rhat={round(cur_rhat,2)}, ESS: {round(cur_ess,2)}')
    fig.suptitle(f'Node {idx}, nthin={nthin}', size=40)
    pdf.savefig()
    plt.clf()
pdf.close();


#%%

# plot posterior

n_nodes = len(inneridx) #tree.data['value'].shape[0]
# Determine the grid size
grid_size = int(np.ceil(np.sqrt(n_nodes)))

# Create subplots
fig, axes = plt.subplots(grid_size, grid_size, figsize=(20, 20), sharex=True, sharey=True)

# Flatten the axes array for easy iteration
axes = axes.flatten()

# Create a scatter plot for each k
k=0
sample_n=50

for idx in inneridx: # loop over i innernodes
    innernodes = flat_trees[:,:,idx,:].reshape(-1, nxd)[::sample_n,:]
    inode = np.append(innernodes, innernodes[:,0:2],1)
    for i in range(inode.shape[0]):
        axes[k].plot(inode[i,::2], inode[i,1::2], '--.', color='steelblue', alpha=0.3, label='true shape')
    #fig.suptitle(f'Node {idx}', size=40);
    k+=1
# Hide any unused subplots
for j in range(n_nodes, grid_size * grid_size):
    fig.delaxes(axes[j])

#fig.suptitle(f'kalpha1={round(params['k_alpha_1'].value, 2)}, kalpha2={round(params['k_alpha_2'].value,2)}, k_sigma={round(params['k_sigma'].value,2)} and min_weight={round(params['min_weight'].value,2)}')
plt.tight_layout()
#fig.subplots_adjust(top=0.95)
#plt.show()
gc.collect()
plt.savefig(path + f'/posterior_innernodes_burnin={burnin}_MCMCiter={MCMC_iter}.pdf')

# In[ ]:
# PLOT TRACE AND DENSITY FOR PARAMETERS
# wait with array in case of irregular dimensions 
raw_pars = [[np.genfromtxt(folder_runs + chains[i]+'/'+temp_name[i]+par+"s.csv", delimiter = ",") for i in range(len(chains))] for par in pars_name]
raw_acceptpars = [[np.genfromtxt(folder_runs + chains[i]+'/'+temp_name[i]+"acceptkalpha.csv", delimiter = ",") for i in range(len(chains))] for par in pars_name]

pars = [np.array([raw_pars[j][i][burnin:MCMC_iter] for i in range(len(raw_pars[0]))]) for j in range(len(raw_pars))]
[p.shape for p in pars]
acceptpars = [np.array([raw_acceptpars[j][i][burnin:MCMC_iter] for i in range(len(raw_acceptpars[0]))]) for j in range(len(raw_acceptpars))]
[ap.shape for ap in acceptpars]

# ## plot diagnostics for parameters

# In[ ]:
parsdict = dict(zip(pars_name, pars)) #{'kalpha': pars[0]}
MCMC_result = parsdict #parsdict|innernodedict
parsres = arviz.convert_to_dataset(MCMC_result)
rhat = arviz.rhat(parsres)
mcse = arviz.mcse(parsres)
ess = arviz.ess(parsres)
arviz.summary(parsres)

# save rhat for plotting
rhats_par = np.array([rhat['kalpha'], rhat['gtheta']])
np.savetxt(path+'/'+"rhats_pars.csv",np.array(rhats_par), delimiter=",")

# In[ ]:

#true_vals = true_pars #[true_pars]
keys = pars_name
print(keys)
#print(true_vals)
print([pars[i].shape for i in range(len(pars))])
print(keys)




# In[ ]:

colors = sns.color_palette('pastel', len(chains))
fig, axes = plt.subplots(nrows=len(keys), ncols=2, figsize=(20,10), sharex=False)
p = 0
for i, ax in zip(range(len(axes.flat)), axes.flat): 
        if i%2 == 0: 
            for j in range(pars[p].shape[0]): #loop over chains 
                ax.plot(pars[p][j,:], color=colors[j], alpha=0.5)
            #ax.hlines(y=true_vals[p], xmin=0, xmax=pars[p].shape[1], color='skyblue')
            ax.set_title(f'{keys[p]}, rhat: {round(float(np.array(rhat[keys[p]])),2)} \n (ess: {round(float(np.array(ess[keys[p]])),2)}) ')
        else:
            for j in range(pars[p].shape[0]):
                sns.kdeplot(pars[p][j,:], ax=ax)
                sns.rugplot(pars[p][j,:], ax=ax)
            #ax.axvline(x = true_vals[p], ymin = 0, ymax = 1, color='orange') 
            ax.set_title(f'{keys[p]}, rhat: {round(float(np.array(rhat[keys[p]])),2)} \n (ess: {round(float(np.array(ess[keys[p]])),2)}) ')#
            p+=1
fig.suptitle(f"Iter: {MCMC_iter}, Burnin: {burnin} \n", fontsize=15)
fig.tight_layout()
fig.savefig(path+f'/pars_burnin={burnin}_MCMCiter={MCMC_iter}.pdf')





# %%
