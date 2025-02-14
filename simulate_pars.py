import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Arguments for script')
parser.add_argument('-n', help = 'number of parameters to simulate', nargs='?', default=1, type=int)
parser.add_argument('-pkalpha', help='prior kalpha (uniform, loc scale)', nargs=2, default=[0.005, 0.025], metavar= ('loc', 'scale'), type=float)
parser.add_argument('-pgtheta', help='prior gtheta (uniform, loc scale)', nargs=2, default=[0.8, 0.4], metavar= ('loc', 'scale'), type=float)
parser.add_argument('-o', help = 'output folder', nargs='?', default='', type=str)
args = parser.parse_args()

# parse arguments 
kalpha_loc = args.pkalpha[0]
kalpha_scale = args.pkalpha[1]
gtheta_loc = args.pgtheta[0]
gtheta_scale = args.pgtheta[1]
outpath = args.o
n= args.n

# simulate parameters 
kalphas = np.random.uniform(kalpha_loc, kalpha_loc + kalpha_scale, n)
gthetas = np.random.uniform(gtheta_loc, gtheta_loc + gtheta_scale, n)
pars = np.array([kalphas, gthetas]).T
np.savetxt(f'{outpath}kalpha:{kalpha_loc}-{kalpha_scale}_gtheta:{gtheta_loc}-{gtheta_scale}.csv',pars, delimiter = ' ')