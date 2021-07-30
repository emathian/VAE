#!/bin/bash
#SBATCH --job-name=LNEN_tiles
#SBATCH --ntasks-per-node=1         # nombre de taches MPI par noeud
#SBATCH --output=Copy_LNEN_Tiles_norm%j.out          # nom du fichier de sortie
#SBATCH --error=Copy_LNEN_Tiles_norm%j.out
#SBATCH --partition=high_p
#SBATCH --account=gcs
python filepathlist.py --root /home/mathiane/ln_LNEN_work_mathian/Tiles_512_512_1802 --outputfilename LNEN_tiles.txt
