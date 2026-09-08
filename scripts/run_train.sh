torchrun --standalone --nnodes=1 --nproc_per_node=2 train.py --lambda_delta 0.2 --lr_patience 3 --early_stop_patience 5 --save_prefix erna_100bp_same_locus_delta
