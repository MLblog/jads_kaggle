# Jupyter Notebooks
In this folder we will put all the Jupyter Notebooks that we create. 

## Purpose of notebooks
- Experiments with the scripts (classes and methods) in the main folder
- Explorative Data Analysis 

## Important note: Checkout notebooks
In most cases, you will just use the structure of the already existing notebook to experiment. However, during experimentation your system will save the notebook frequently. For this reason
we need to take two things into account:
1. In most cases you just experimented with the scripts and you do not want to change anything to the notebook itself. In that case, you should not forget to checkout the notebook in Git:
`git checkout <filename>.ipynb`. In that case your HEAD and your local version will go back to the last version of the team repository.
2. In case you want to push an updated version of a Jupyter Notebook, make sure you delete the irrelevant output cells in the notebook. Since the output cells are also scripted (long nesty scripts), 
many changes are recorded in git and this may be confusing.