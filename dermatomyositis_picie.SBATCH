#!/bin/bash
#
#SBATCH --job-name=training
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=5
#SBATCH --time=20:00:00
#SBATCH --mem=128GB
#SBATCH --mail-type=END
#SBATCH --output=train_dermatomyositis_resnet34_adam_lr_1e-4_scheduler_3098592.out

module purge
## Please change python file name in line 18

singularity exec --nv \
	    --overlay /scratch/ssc10020/overlay-50G-10M.ext3:ro \
	    /scratch/work/public/singularity/cuda11.2.2-cudnn8-devel-ubuntu20.04.sif \
	    /bin/bash -c "source /ext3/miniconda3/bin/activate; conda activate picie; 
		mkdir -p results/picie/train/dermatomyositis/3098592;
		python train_picie.py --data_root datasets/ --save_root results/picie/train/dermatomyositis/3098592 --device 'cuda' --pretrain --repeats 1 --lr 1e-4 --seed 3098592 --num_init_batches 26 --num_batches 1 --kmeans_n_iter 26 --K_train 2 --K_test 2 --stuff --thing --batch_size_cluster 128 --num_epoch 50 --res 480 --res1 480 --res2 480 --dataset 'dermatomyositis' --augment --jitter --blur --grey --equiv --random_crop --h_flip
		"