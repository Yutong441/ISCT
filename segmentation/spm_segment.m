function out= spm_segment (toolbox_dir, pth_ct, dir_out)
cd (toolbox_dir);
addpath ('CTseg');
addpath ('spm12');
addpath ('spm12/toolbox/mb');
addpath ('spm12/toolbox/Shoot');
addpath ('spm12/toolbox/Longitudinal');
[res, vol] = spm_CTseg(pth_ct, dir_out, true, true, true, true, 1);

pth_mu = 'CTseg/mu_CTseg.nii';
pth_y = dir (strcat (dir_out, '/y_*.nii'));
pth_y = strcat (dir_out, '/', pth_y(1).name);

file_list = {'c01', 'c02', 'c03', 'ss'};
for num = 1:length(file_list)
        disp (strcat (dir_out, '/', file_list{num}, '_*.nii'));
        warp_file = dir (strcat (dir_out, '/', file_list{num}, '*'));
        warp_file = strcat (dir_out, '/', warp_file(1).name);

        matlabbatch = {};
        matlabbatch{1}.spm.util.defs.comp{1}.inv.comp{1}.def     = {pth_y};
        matlabbatch{1}.spm.util.defs.comp{1}.inv.space           = {pth_mu};
        matlabbatch{1}.spm.util.defs.out{1}.pull.fnames          = {warp_file};
        matlabbatch{1}.spm.util.defs.out{1}.pull.savedir.saveusr = {dir_out};
        matlabbatch{1}.spm.util.defs.out{1}.pull.interp          = 1;
        matlabbatch{1}.spm.util.defs.out{1}.pull.mask            = 1;
        matlabbatch{1}.spm.util.defs.out{1}.pull.fwhm            = [0 0 0];
        matlabbatch{1}.spm.util.defs.out{1}.pull.prefix          = 'wpull_';
        spm_jobman('run',matlabbatch);
end
out = 'done';
