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
def get_mode(all_chains): 
    '''Function for marginal mode estimation '''
    mode_est = []
    for i in range(all_chains.shape[1]):
        kdes = az.kde(all_chains[:,i].flatten())
        mest = kdes[0][np.where(kdes[1] == max(kdes[1]))[0][0]]
        mode_est.append(mest)
    mode_est = np.array(mode_est)
    return(mode_est)


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
pars_name = ['kalpha', 'gtheta']
rep_path = len(pars_name)+1
chains = os.listdir(folder_runs) # use all chains in data seed folder 
chains = [c for c in chains if c[0] not in ['_', '.']] # remove files starting with underscore
print(chains)

#%%

MCMC_iter = 5000
burnin = 1500
nthin = 1 # see from script/running conditions, not used for plotting
folder_runs = 'BM14/12078537373791567/'
folder_simdata = 'BM14/simdata/12078537373791567/' #'BM9/simdata/3513656273068705/' #args.folder_simdata +'/'
nnodes = 9
nxd=40
levelorder_tree = 'levelorder_chazot_subtree.nw'
thresh_kdeplot_2D = 0.05 # probability mass below last contour line
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
flat_true_tree = np.genfromtxt(folder_simdata+"flat_true_tree.csv", delimiter = ",")
super_root = [np.genfromtxt(folder_runs + chains[i]+'/'+temp_name[i]+"inference_root_start.csv", delimiter = ",") for i in range(len(chains))]
_super_root = [np.concatenate((super_root[i], super_root[i][0:2])) for i in range(len(chains))]
_super_root = np.unique(np.array(_super_root), axis=0)
#flat_true_tree = np.genfromtxt(folder_simdata + chains[0]+'/'+temp_name[0]+"flat_true_tree.csv", delimiter = ",")

#flat_true_tree[0]
#[print(np.repeat(flat_trees_raw[i], tree_counters[i], axis=0).shape) for i in range(len(chains))]

# In[12]:
flat_trees = np.array([np.repeat(flat_trees_raw[i], tree_counters[i], axis=0)[burnin*rep_path:(MCMC_iter//nthin)*rep_path] for i in range(len(flat_trees_raw))])
flat_trees.shape #nchains x MCMC_iter x nnodes x nxd

# In[13]:
# Thin the chains for faster plotting
#flat_trees_orig = flat_trees.copy()
#flat_trees = flat_trees[:,::nthin, :, :] # keep only every fifth element in chain
#flat_trees.shape

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


#%%
# EVALUATE STATISTICS FOR PLOTTING
ft_=flat_trees.reshape(-1, nnodes, nxd)
mean_est = np.mean(ft_, axis=0)
median_est = np.median(ft_, axis=0)
#mode_est = mode(ft_, axis=0)[0]

# evaluate MSEs
squared_error_mean = (flat_true_tree-mean_est)**2
squared_error_median = (flat_true_tree-median_est)**2
#squared_error_mode = (flat_true_tree-mode_est)**2

# evaluate bias 
bias_mean = (flat_true_tree-mean_est)
bias_median = (flat_true_tree-median_est)
#bias_mode = (flat_true_tree-mode_est)

# save statistics for final plots
np.savetxt(path+'/'+f"squared_error_mean.csv",squared_error_mean, delimiter=",")
np.savetxt(path+'/'+"squared_error_median.csv",squared_error_median, delimiter=",")
#np.savetxt(path+'/'+"squared_error_mode.csv",squared_error_mode, delimiter=",")
np.savetxt(path+'/'+"bias_mean.csv",bias_mean, delimiter=",")
np.savetxt(path+'/'+"bias_median.csv",bias_median, delimiter=",")
#np.savetxt(path+'/'+"bias_mode.csv",bias_mode, delimiter=",")


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
    true_innernode = flat_true_tree[idx,:]
    tinode = np.concatenate((true_innernode, true_innernode[0:2]))  
    axes.plot(tinode[::2], tinode[1::2], '--.', color='black', label='true shape')
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
        true_innernode = flat_true_tree[idx,:]
        curcol = colors[j]
        for i, ax in zip(range(innernode.shape[1]), axes.flat): # loop over dimensions
            ax.plot(innernode[:,i], color = curcol, alpha=0.5)
            ax.hlines(y=true_innernode[i], xmin=0, xmax=innernode.shape[0], color='skyblue')
            cur_ess = esss[idx][i]#round(arviz.ess(innernode[:,i]),2)
            cur_rhat = rhats[idx][i]
            ax.set_title(f'{i}, Rhat={round(cur_rhat,2)}, ESS: {round(cur_ess,2)}')
    fig.suptitle(f'Node {idx}, nthin={nthin}', size=40)
    pdf.savefig()
    plt.clf()
pdf.close();

# In[16]:
# PLOT DENSITY FOR ALL DIMENSIONS OF ALL INNERNODES
'''
colors = sns.color_palette('pastel', len(chains))
pdf = backend_pdf.PdfPages(path + f'/density-innernodes_burnin={burnin}_MCMCiter={MCMC_iter}.pdf')
plt.figure(1);
#for idx in range(flat_trees.shape[2]): # loop over nodes
for idx in inneridx: # loop over innernodes
    fig, axes = plt.subplots(nrows=6, ncols=7, figsize=(25,15), sharex=False);
    for j in range(flat_trees.shape[0]): # loop over chains
        innernode = flat_trees[j,:,idx,:]
        true_innernode = flat_true_tree[idx,:]
        for i, ax in zip(range(innernode.shape[1]), axes.flat): # loop over dimensions
            sns.kdeplot(innernode[:,i], ax=ax, label=chains[j], color=colors[j], common_norm=True); #label=chains_proposal_par[j]
            sns.rugplot(innernode[:,i], ax=ax);
            ax.axvline(x = true_innernode[i], ymin = 0, ymax = 1, color='orange') 
            cur_ess = esss[idx][i]#round(arviz.ess(innernode[:,i]),2)
            cur_rhat = rhats[idx][i]
            ax.set_title(f'{i}, Rhat={round(cur_rhat,2)}, ESS: {round(cur_ess,2)}')
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower right')
    fig.suptitle(f'Node {idx}, nthin={nthin}', size=40);
    fig.tight_layout();
    pdf.savefig();
    plt.clf();
pdf.close();
'''

# In[ ]:
'''# PLOT ANCESTRAL STATE I.E. CONTOUR PLOT FOR ALL INNER NODES
colors = sns.color_palette('pastel', len(chains))
pdf = backend_pdf.PdfPages(path + f'/ancestral-state-inner-nodes_burnin={burnin}_MCMCiter={MCMC_iter}.pdf')
plt.figure(1);
for idx in inneridx: # loop over i innernodes
    print(idx)
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(5,5), sharex=False);
    innernodes = flat_trees[:,:,idx,:]
    true_innernode = flat_true_tree[idx,:]
    tinode = np.concatenate((true_innernode, true_innernode[0:2]))  
    innernode = innernodes.reshape(-1, nxd//2)
    inodet = innernode.reshape(innernode.shape[0]*innernode.shape[1]//2, 2)
    landmark_name = np.tile(list(range(20)), inodet.shape[0]//20).reshape(-1,1) #np.array([np.array(range(innernodes.shape[1]//2)) for i in range(innernodes.shape[0])]).reshape(-1,1) #np.array([np.array([i]*innernode.shape[0]) for i in range(innernode.shape[1]//2)]).reshape(-1, 1)
    data = pd.DataFrame(np.concatenate((inodet, landmark_name), axis=1), columns=['x', 'y', 'landmark'])
    #sns.kdeplot(data, x="x", y="y", hue="landmark", fill=True, legend=False, ax=axes, thresh=thresh_kdeplot_2D)
    sns.kdeplot(data, x="x", y="y", fill=True, legend=False, ax=axes, thresh=thresh_kdeplot_2D)
    axes.plot(tinode[::2], tinode[1::2], '--.', color='black', label='true shape')
    #handles, labels = ax.get_legend_handles_labels()
    #fig.legend(handles, labels, loc='lower right')
    fig.suptitle(f'Node {idx}', size=40);
    fig.tight_layout();
    pdf.savefig();
    plt.clf();
pdf.close();
'''

#%%
##### PLOT ROOT ESTIMATION WITH TRUE VALUE AND SUPER ROOT
idx=0 # root 
sample_n = 50
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(5,5), sharex=False);
innernodes = flat_trees[:,:,idx,:].reshape(-1, nxd)[::sample_n,:]
inode = np.append(innernodes, innernodes[:,0:2],1)
for i in range(inode.shape[0]):
    axes.plot(inode[i,::2], inode[i,1::2], '--.', color='steelblue', alpha=0.3)
true_innernode = flat_true_tree[idx,:] # root 
tinode = np.concatenate((true_innernode, true_innernode[0:2]))  
axes.plot(tinode[::2], tinode[1::2], '--.', color='green', label='Root')
for i in range(len(_super_root)):
    axes.plot(_super_root[i][::2], _super_root[i][1::2], '--.', color='orange', label='super root', alpha=.9)
axes.legend()
plt.savefig(path+f'/root_burnin={burnin}_MCMCiter={MCMC_iter}.pdf')


#%%
# PLOT LEAVES IN SIMULATED DATA ON TOP OF ROOT
# define node of interest, we plot this node in the background of all data points
idx=0
true_innernode = flat_true_tree[idx,:] # this should be the true root

# ancestral reconstruction evaluate
tinode = np.concatenate((true_innernode, true_innernode[0:2]))
leaves_ = [np.concatenate((flat_true_tree[int(idx)], flat_true_tree[int(idx)][0:2])) for idx in leafidx]
len(leaves_)

# read true parameters
true_pars = [np.genfromtxt(folder_simdata +p+"_sim.csv", delimiter = ",") for p in pars_name]
true_pars

if nnodes == 9: # 5 leaves 
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(25,15), sharex=True, sharey=True)
    for i, ax in zip(range(len(leaves_)), axes.flat):
        if i<5:
        #ax.plot(leaves_[i][::2], leaves_[i][1::2], '--.', color='orange', label='leaf', lw=5, alpha=0.5)
        #ax.plot(leaves_[i][::2], leaves_[i][1::2], 'o', color='black', label='leaf', lw=1)
            ax.plot(leaves_[i][::2], leaves_[i][1::2], '--o', color='orange', label='leaf', lw=5)
            ax.plot(tinode[::2], tinode[1::2], '--.', color='black', label='true root')
            ax.set_title(f'levelorder traversal idx = {int(leafidx[i])}')
            ax.legend(loc="upper right")
    
        else: 
            for j in range(5): # plot all leaves on top of each other 
                    ax.plot(leaves_[j][::2], leaves_[j][1::2], '--.', color='orange', label='leaf', lw=2)
                    ax.plot(tinode[::2], tinode[1::2], '--.', color='black', label='true root', lw=4)

                #ax=sns.kdeplot(data, x="x", y="y", hue="landmark", fill=True, legend=False, thresh=thresh_kdeplot_2D)
                #for j in range(len(xstart_)):
                 #       ax.plot(xstart_[j][::2], xstart_[j][1::2], '--.', color='lightblue', label='proposal shape')
                #ax.plot(tinode[::2], tinode[1::2], '--.', color='black', label='true root')
                #for k in range(len(leaves_)): 
                #        ax.plot(leaves_[k][::2], leaves_[k][1::2], '--.', color='orange', label='leaf', lw=1)
#for ax in axes[1:]:
#    ax.sharex(axes[0])
    fig.suptitle(f'Simulated data: alpha={np.round(true_pars[0],3)}, sigma={np.round(true_pars[1],3)}')
    plt.savefig(path+f'/_leaves_.pdf')

if nnodes == 59: # 30 leaves 
    fig, axes = plt.subplots(nrows=6, ncols=5, figsize=(45,35), sharex=True, sharey=True)
    for i, ax in zip(range(len(leaves_)), axes.flat):
    #    if i<5:
        #ax.plot(leaves_[i][::2], leaves_[i][1::2], '--.', color='orange', label='leaf', lw=5, alpha=0.5)
        #ax.plot(leaves_[i][::2], leaves_[i][1::2], 'o', color='black', label='leaf', lw=1)
        ax.plot(leaves_[i][::2], leaves_[i][1::2], '--o', color='orange', label='leaf', lw=5)
        ax.plot(tinode[::2], tinode[1::2], '--.', color='black', label='true root')
        ax.set_title(f'levelorder traversal idx = {int(leafidx[i])}')
        ax.legend(loc="upper right")
    fig.suptitle(f'Simulated data: alpha={np.round(true_pars[0],3)}, sigma={np.round(true_pars[1],3)}')
    plt.savefig(path+f'/_leaves_.pdf')
    plt.close()
    #plot variation in all leaves 
    for j in range(len(leafidx)): # plot all leaves on top of each other 
        plt.plot(leaves_[j][::2], leaves_[j][1::2], '--.', color='orange', label='leaf', lw=2)
    plt.plot(tinode[::2], tinode[1::2], '--.', color='black', label='true root', lw=4)
    fig.suptitle(f'Simulated data: alpha={np.round(true_pars[0],3)}, sigma={np.round(true_pars[1],3)}')
    plt.savefig(path+f'/_leaves-superimposed_.pdf')
# In[ ]:
# PLOT TRACE AND DENSITY FOR PARAMETERS
# wait with array in case of irregular dimensions 
raw_pars = [[np.genfromtxt(folder_runs + chains[i]+'/'+temp_name[i]+par+"s.csv", delimiter = ",") for i in range(len(chains))] for par in pars_name]
raw_acceptpars = [[np.genfromtxt(folder_runs + chains[i]+'/'+temp_name[i]+"acceptkalpha.csv", delimiter = ",") for i in range(len(chains))] for par in pars_name]

#raw_pars = [[np.genfromtxt(folder + chains[i]+'/'+temp_name[i]+par+"s.csv", delimiter = ",") for i in range(len(chains))] for par in pars_name]
#raw_acceptpars = [[np.genfromtxt(folder + chains[i]+'/'+temp_name[i]+"acceptkalpha.csv", delimiter = ",") for i in range(len(chains))] for par in pars_name]
#print([raw_pars[j][i].shape for j in range(len(raw_pars)) for i in range(len(raw_pars[0]))]) # check max MCMC iter
#print([raw_acceptpars[j][i].shape for j in range(len(raw_pars)) for i in range(len(raw_pars[0]))]) # check max MCMC iter
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

true_vals = true_pars #[true_pars]
keys = pars_name
print(keys)
print(true_vals)
print([pars[i].shape for i in range(len(pars))])
print(keys)

#%%
# EVALUATE STATISTICS FOR PLOTTING
#ft_=flat_trees.reshape(-1, nnodes, nxd)
pars_ = np.array([pars[0].flatten(), pars[1].flatten()])
mean_est = np.mean(pars_, axis=1)
median_est = np.median(pars_, axis=1)
mode_est = get_mode(pars_.T) #mode(pars_, axis=1)[0]

# evaluate MSEs
squared_error_mean = (true_pars-mean_est)**2
squared_error_median = (true_pars-median_est)**2
squared_error_mode = (true_pars-mode_est)**2

# evaluate bias 
bias_mean = (true_pars-mean_est)
bias_median = (true_pars-median_est)
bias_mode = (true_pars-mode_est)

# save statistics for final plots
np.savetxt(path+'/'+"squared_error_mean_pars.csv",squared_error_mean, delimiter=",")
np.savetxt(path+'/'+"squared_error_median_pars.csv",squared_error_median, delimiter=",")
np.savetxt(path+'/'+"squared_error_mode_pars.csv",squared_error_mode, delimiter=",")
np.savetxt(path+'/'+"bias_mean_pars.csv",bias_mean, delimiter=",")
np.savetxt(path+'/'+"bias_median_pars.csv",bias_median, delimiter=",")
np.savetxt(path+'/'+"bias_mode_pars.csv",bias_mode, delimiter=",")



# In[ ]:

colors = sns.color_palette('pastel', len(chains))
fig, axes = plt.subplots(nrows=len(keys), ncols=2, figsize=(20,10), sharex=False)
p = 0
for i, ax in zip(range(len(axes.flat)), axes.flat): 
        if i%2 == 0: 
            for j in range(pars[p].shape[0]): #loop over chains 
                ax.plot(pars[p][j,:], color=colors[j], alpha=0.5)
            ax.hlines(y=true_vals[p], xmin=0, xmax=pars[p].shape[1], color='skyblue')
            ax.set_title(f'{keys[p]}, rhat: {round(float(np.array(rhat[keys[p]])),2)} \n (ess: {round(float(np.array(ess[keys[p]])),2)}) ')
        else:
            for j in range(pars[p].shape[0]):
                sns.kdeplot(pars[p][j,:], ax=ax)
                sns.rugplot(pars[p][j,:], ax=ax)
            ax.axvline(x = true_vals[p], ymin = 0, ymax = 1, color='orange') 
            ax.set_title(f'{keys[p]}, rhat: {round(float(np.array(rhat[keys[p]])),2)} \n (ess: {round(float(np.array(ess[keys[p]])),2)}) ')#
            p+=1
fig.suptitle(f"Iter: {MCMC_iter}, Burnin: {burnin} \n", fontsize=15)
fig.tight_layout()
fig.savefig(path+f'/pars_burnin={burnin}_MCMCiter={MCMC_iter}.pdf')




