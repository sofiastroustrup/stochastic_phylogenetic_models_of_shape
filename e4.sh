#!/bin/bash
# ls _root-exp/e1/simdata | parallel --memsuspend 5G -j 50% "echo {} | bash e1.sh"
# screen -ls | grep Detached | cut -d. -f1 | awk '{print $1}' | xargs kill

read seed
              
screen -md -S e4 python mcmc.py -wandb root-asym-top -N 3000 -l 0.8 -datapath _root-exp/e4/simdata -pkalpha 0.0 0.05 -pgtheta 0.0 1.5 -ov 0.001 -super_root _root-exp/simulation-setup/circle_n=30_d=2_f1.csv -o _root-exp/e4/runs -sd_gt 0.2 -sd_ka 0.005 -rb 2 -ds $seed
screen -md -S e4 python mcmc.py -wandb root-asym-top -N 3000 -l 0.8 -datapath _root-exp/e4/simdata -pkalpha 0.0 0.05 -pgtheta 0.0 1.5 -ov 0.001 -super_root _root-exp/simulation-setup/circle_n=30_d=2_f1.csv -o _root-exp/e4/runs -sd_gt 0.2 -sd_ka 0.005 -rb 2 -ds $seed
screen -md -S e4 python mcmc.py -wandb root-asym-top -N 3000 -l 0.8 -datapath _root-exp/e4/simdata -pkalpha 0.0 0.05 -pgtheta 0.0 1.5 -ov 0.001 -super_root _root-exp/simulation-setup/circle_n=30_d=2_f1.csv -o _root-exp/e4/runs -sd_gt 0.2 -sd_ka 0.005 -rb 2 -ds $seed