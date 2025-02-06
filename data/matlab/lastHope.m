clear;
clc;
%%
miPyPath = 'C:\Users\stone\AppData\Local\Programs\Python\Python312\python.exe';
lenPyPath = 'D:\Python\python.exe';
pyenv('Version', lenPyPath)
model = py.eqDnn.load_model();

function [equalized_signal] = eqInfer(model, csi_est, rx_signal)
    csi_est = py.numpy.array(cat(ndims(csi_est)+1, real(csi_est), imag(csi_est)));
    rx_signal = py.numpy.array(cat(ndims(rx_signal)+1, real(rx_signal), imag(rx_signal)));
    
    equalized_signal = py.eqDnn.infer(model, csi_est, rx_signal, rx_signal);

    % 转换 Python numpy 输出为 MATLAB 矩阵
    equalized_signal = double(py.array.array('d', py.numpy.nditer(equalized_signal)));
    equalized_signal = reshape(equalized_signal, 224,14,2,2);
    equalized_signal = complex(equalized_signal(:,:,:,1), equalized_signal(:,:,:,2));
end

% 保存批量数据到文件
load('../raw/eqValData.mat', ...
    'csiLSData',...
    'csiPreData',...
    'csiLabelData', ...
    'txSignalData',...
    'rxSignalData');
idx=10;
csi_perfect = squeeze(complex(csiLabelData(idx,:,:,:,:,1),csiLabelData(idx,:,:,:,:,2)));
tx = squeeze(complex(txSignalData(idx,:,:,:,1),txSignalData(idx,:,:,:,2)));
rx = squeeze(complex(rxSignalData(idx,:,:,:,1),rxSignalData(idx,:,:,:,2)));

size(csi_perfect)

tx_est = eqInfer(model,csi_perfect,rx);

mean((real(tx_est(:))-real(tx(:))).^2 + (imag(tx_est(:))-imag(tx(:))).^2)
disp(mean(abs(tx_est(:) - tx(:)).^2))

