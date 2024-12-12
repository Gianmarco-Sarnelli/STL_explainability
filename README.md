## Repository for the study of global and local explainability using kernel methods
Local_Matrix.py: contains the class that handles the conversion from a global kernel to a local one

Test_independece_PHI.py: script to test whether the matrix PHI has independent columns (this is a requiremet for the local matrix to work)

Test_distance_local_matrix.py: script to test the differences from kernels obtained in the classical way (using local trajectories) and kernels obtained using importance sampling (with local_matrix).\
Both the euclidean distance and the cosine distance are used to measure the differences between kernels.

Results.ipynb: Notebook to visualize the results of Test_distance_local_matrix.py.\
At the moment I'm only using the cosine distance for the tests because the kernel obtained with importance sampling is consistently shorter than the classical kernel. This is either a statistical caracteristic of the importance sampling or a problem with the code itself.\
Either way, the results show a clear malfunction while the number of trajectories and formulae increases, meaning that there's a problem with the code implementation somewhere.
