python train_homo_recon.py -s /home/EMPIAR/gaoxiang_group/gaussian_preprocess_256 -p sample_vol \
    --particle_name particles.256 \
    --pose poses.npy \
    --ctf ctf.npy \
    --point gaussians_2inter_total.npy \
    --cfg cfg_scale_0.5_1.0.json \
    --output total_homo_batch100 \
    --loss real_rfft \
    --batch 100


python train_em_relion_vae_xyz_batch.py -s /home/EMPIAR/10343/gaussian_preprocess_256 -p sample_vol \
    --particle_name particles.256 \
    --pose poses.npy \
    --ctf ctf.npy \
    --point point_cloud.ply \
    --cfg cfg_scale_0.5_1.0.json \
    --output heter_density_scaling_pos_zdim10_dp \
    --loss real_rfft \
    --batch 64 \
    --zdim 10 \
    --nclass 2


python train_em_relion_vae_xyz_batch.py -s /home/EMPIAR/10343/gaussian_preprocess_256 -p sample_vol \
    --particle_name particles.256 \
    --pose poses.npy \
    --ctf ctf.npy \
    --point point_cloud.ply \
    --cfg cfg_scale_0.5_1.0.json \
    --output heter_density_scaling_pos_zdim10_dp \
    --loss real_rfft \
    --batch 64 \
    --zdim 10 \
    --nclass 2 \
    --check
