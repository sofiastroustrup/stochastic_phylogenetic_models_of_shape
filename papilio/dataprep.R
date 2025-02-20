setwd("/Users/lkn315/Library/CloudStorage/OneDrive-UniversityofCopenhagen/stochastic_phylogenetic_models_of_shape")
library(geomorph)
library(tidyverse)
library(tibble)
library(readr)

# read data 
landmarks <- read_delim('papilio/data/papilio_data.csv')
