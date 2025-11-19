# python train_homo_recon.py -s /home/data/Single-Particle/4DGS/GaussianEM/dataset/IgG-1D/gaussian_preprocess \
#     --particle_name img_stack_part \
#     --pose poses_part.npy \
#     --ctf ctf_part.npy \
#     --point gaussians_1inter_049.npy \
#     --cfg cfg_scale_0.5_1.0.json \
#     --output total_homo_batch100 \
#     --batch 100


# python train_em_relion_vae_xyz_batch.py -s /home/data/Single-Particle/4DGS/GaussianEM/dataset/IgG-1D/gaussian_preprocess \
#     --particle_name img_stack \
#     --pose poses.npy \
#     --ctf ctf.npy \
#     --point point_cloud.ply \
#     --cfg cfg_scale_0.5_1.0.json \
#     --output heter_density_scaling_pos_zdim10_dp \
#     --batch 32


# python analyze_heter.py --source_dir /home/data/Single-Particle/4DGS/GaussianEM/dataset/10841 \
#     --gaussians point_cloud.ply \
#     --cfg cfg.json \
#     --model_weight weights.19.pkl \
#     --latent weights.19.output.pkl \
#     --nclass 10
