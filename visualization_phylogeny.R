library(ggtree)
library(ggplot2)

ftree <- read.tree("levelorder_chazot_full_tree.nw")
ggtree(ftree, layout='rectangular')+geom_tiplab(color='black', hjust=-0.2) + geom_nodelab(hjust=-0.1) + 
  geom_nodepoint(color="black", alpha=1, size=1) + #theme(plot.margin = unit(c(10,8,10,8), "mm")) +
  ggplot2::xlim(0, 50) #+ geom_label(aes(x=0,label=0) )
#+ geom_tippoint(color='darkgreen') 
  #geom_text(aes(label=node), hjust=-.3)+
  #geom_tiplab()


ggtree(ftree, layout='rectangular')+geom_nodepoint(color="black", alpha=1, size=1) + geom_text(aes(label=node), hjust=-.3)#+geom_tiplab()



# ggtree plot for section 5.4  --------------------------------------------
ftree <- read.tree("chazot/data/chazot_full_tree_5.3.nw")
ggtree(ftree, layout='rectangular')+ geom_nodelab(hjust=-0.15) + 
  geom_nodepoint(color="black", alpha=1, size=1) + geom_tiplab()+
  ggplot2::xlim(0, 50) + geom_nodelab(aes(node=1, label=0), hjust=-0.2)



# add images to tree ------------------------------------------------------
library(ggimage)
imgdir = system.file("_sim-30-leaves/runs_v2/78241558624040307/leaves", package="TDbook")
imgdir = "_sim-30-leaves/runs_v2/78241558624040307/leaves"
x = read.tree(text = "levelorder_chazot_full_tree.nw")
ggtree(x, layout='rectangular') +
  geom_tiplab(aes(image=paste0(imgdir, '/', '3.pdf')), geom="image", offset=2, align=2, size=.1) #+
  #geom_tiplab(geom='label', offset=1, hjust=.5) #+ 
  ##geom_image(x=.8, y=5.5, image=paste0(imgdir, "/frog.jpg"), size=.2)


url <- paste0("https://raw.githubusercontent.com/TreeViz/",
              "metastyle/master/design/viz_targets_exercise/")
x <- read.tree(paste0(url, "tree_boots.nwk"))
info <- read.csv(paste0(url, "tip_data.csv"))
p <- ggtree(x) %<+% info + xlim(NA, 6)
p + geom_tiplab(aes(image= imageURL), geom="image", offset=2, align=T, size=.16, hjust=0) +
  geom_tiplab(geom="label", offset=1, hjust=.5)


