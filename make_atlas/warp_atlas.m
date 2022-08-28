function out = wrap_atlas (toolbox_dir, template, dir_out)
cd (toolbox_dir);
addpath ('CTseg');
addpath ('spm12');
addpath ('spm12/toolbox/mb');
addpath ('spm12/toolbox/Shoot');
addpath ('spm12/toolbox/Longitudinal');

file_list = dir (strcat (template, '/*.nii')) ;
pth_mu = 'CTseg/mu_CTseg.nii';
pth_y = 'template/y_mni2mu.nii';

for num = 1:length(file_list)
        warp_file = file_list(num);
        warp_file = strcat (template, '/', warp_file(1).name);
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

out='done';
