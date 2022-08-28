function done = MNItrans (toolbox_dir, template)
cd (toolbox_dir);
addpath ('CTseg');
addpath ('spm12');
addpath ('spm12/toolbox/mb');
addpath ('spm12/toolbox/Shoot');
addpath ('spm12/toolbox/Longitudinal');

% CTseg template
p_mu      = fullfile('CTseg/mu_CTseg.nii');
n_mu      = nifti(p_mu);
M_mu      = n_mu.mat;

% atlas
n_atlas  = nifti(template);
M_atlas  = n_atlas.mat;
dm_atlas = n_atlas.dat.dim(1:3);
% affine from mni to mu
load(fullfile('CTseg/Mmni.mat'), 'Mmni');

% make affine warp
M_kl       = Mmni\M_atlas;
M          = M_mu*M_kl;
y          = spm_CTseg_util('affine',n_mu.dat.dim(1:3), inv(M));
y          = reshape(y, [n_mu.dat.dim(1:3), 1, 3]);
p_y_mni2mu = fullfile('template/y_mu2mni.nii');
spm_CTseg_util('write_nii',p_y_mni2mu,single(y),M_atlas,'mu-to-mni affine');
out = 'done';
