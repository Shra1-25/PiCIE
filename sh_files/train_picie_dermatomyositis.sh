K_train=2
K_test=2
bsize=64
num_epoch=50
KM_INIT=20
KM_NUM=1
KM_ITER=20
SEED=65967
LR=1e-4

mkdir -p results/picie/train/${SEED}

python train_picie.py \
--data_root datasets/ \
--save_root results/picie/train/${SEED} \
--device 'cuda' \
--pretrain \
--repeats 1 \
--lr ${LR} \
--seed ${SEED} \
--num_init_batches ${KM_INIT} \
--num_batches ${KM_NUM} \
--kmeans_n_iter ${KM_ITER} \
--K_train ${K_train} --K_test ${K_test} \
--stuff --thing  \
--batch_size_cluster ${bsize} \
--num_epoch ${num_epoch} \
--res 480 --res1 480 --res2 480 \
--augment --jitter --blur --grey --equiv --random_crop --h_flip 