#/bin/bash
dir='../save_results/fedis_ahc/cluster_fl/noniid-#label2/svhn'
if [ ! -e $dir ]; then
mkdir -p $dir
fi

python ../main_hc.py --trial=1 \
--rounds=500 \
--num_users=100 \
--frac=0.1 \
--local_ep=10 \
--local_bs=10 \
--lr=0.01 \
--momentum=0.5 \
--model=simple-cnn \
--dataset=svhn \
--datadir='/datasets/' \
--savedir='../save_results/fedis_ahc/' \
--partition='noniid-#label2' \
--alg='cluster_fl' \
--beta=0.1 \
--local_view \
--noise=0 \
--cluster_alpha=0.4 \
--nclasses=10 \
--nsamples_shared=2500 \
--gpu=0 \
--print_freq=50 \
2>&1 | tee $dir'/logs.txt'