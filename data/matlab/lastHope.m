clear;
clc;
%%
miPyPath = 'C:\Users\stone\AppData\Local\Programs\Python\Python312\python.exe';
lenPyPath = 'D:\Python\python.exe';
pyenv('Version', lenPyPath)
model = py.csiFormer.load_model();

function [csi_est] = csiInfer(model, csi_ls, pre_csi,csi_perfect)
    disp('in func')

    csi_ls = py.numpy.array(cat(ndims(csi_ls)+1, real(csi_ls), imag(csi_ls)));
    pre_csi = py.numpy.array(cat(ndims(pre_csi)+1, real(pre_csi), imag(pre_csi)));
    csi_perfect = py.numpy.array(cat(ndims(csi_perfect)+1, real(csi_perfect), imag(csi_perfect)));
    % csi_est = double(py.array.array('d', py.numpy.nditer(csi_perfect1)));
    % csi_est = reshape(csi_est, 52,14,2,2,2);
    % csi_est = complex(csi_est(:,:,:,:,1), csi_est(:,:,:,:,2));
    % disp(all(abs(csi_perfect(:) - csi_est(:)) < 1e-6))


    csi_est = py.csiFormer.infer2(model, csi_ls, pre_csi,csi_perfect);

    details(csi_est)

    % 转换 Python numpy 输出为 MATLAB 矩阵
    csi_est = double(py.array.array('d', py.numpy.nditer(csi_est)));
    csi_est = reshape(csi_est, 52,14,2,2,2);

    csi_est = complex(csi_est(:,:,:,:,1), csi_est(:,:,:,:,2));

    disp('out func')
end

% 保存批量数据到文件
load('../raw/valData.mat', ...
    'csiLSData',...
    'csiPreData',...
    'csiLabelData', ...
    'txSignalData',...
    'rxSignalData');

csi_ls = squeeze(complex(csiLSData(1,:,:,:,:,1),csiLSData(1,:,:,:,:,2)));
csi_pre = squeeze(complex(csiPreData(1,:,:,:,:,:,1),csiPreData(1,:,:,:,:,:,2)));
csi_perfect = squeeze(complex(csiLabelData(1,:,:,:,:,1),csiLabelData(1,:,:,:,:,2)));


csi_est = csiInfer(model, csi_ls, csi_pre,csi_perfect);

disp(all(abs(csi_perfect(:) - csi_est(:)) < 1e-6));

disp(mean(abs(csi_perfect(:) - csi_est(:)).^2))

