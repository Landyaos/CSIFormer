clear;
clc;%3.1801e+09  3.1801e+09  3.1801e+09  4.2397e+09 4.1451e+09 102832043
%% python脚本加载3.4429e+09 3.6214e+09
miPyPath = 'C:\Users\stone\AppData\Local\Programs\Python\Python312\python.exe';
lenPyPath = 'D:\Python\python.exe';
pyenv('Version', lenPyPath)

csiEncoderModel = py.csiEncoder.load_model();
csiFormerModel = py.csiFormer.load_model();
% csiFormerLiteModel = py.csiFormerLite.load_model();
csiFormerStudentModel = py.csiFormerStudent.load_model();

% eqDnnModel = py.eqDnn.load_model();
eqDnnProModel = py.eqDnnPro.load_model();
eqDnnProStudentModel = py.eqDnnProStudent.load_model();
% ceDnnModel = py.ceDnn.load_model();

deeprxModel = py.deepRx.load_model();
channelformerModel = py.channelformer.load_model();

function [csi_est] = csiEncoderInfer(model, csi_ls)
    % 保存原始 csi_ls 的尺寸
    orig_shape = size(csi_ls);
    
    % 拼接实部和虚部，得到新的维度 [orig_shape, 2]
    
    csi_ls_cat = cat(ndims(csi_ls)+1, real(csi_ls), imag(csi_ls));
    
    % 转换为 Python numpy 数组
    csi_ls_py = py.numpy.array(csi_ls_cat);
    
    % 调用 Python 端的推断函数
    csi_est_py = py.csiEncoder.infer(model, csi_ls_py);
    
    % 将 Python numpy 输出转换为 MATLAB 数组
    csi_est = double(py.array.array('d', py.numpy.nditer(csi_est_py)));
    
    % 重构为与拼接后输入相同的 shape: [orig_shape, 2]
    csi_est = reshape(csi_est, [orig_shape, 2]);

    csi_est = complex(csi_est(:,:,:,:,1), csi_est(:,:,:,:,2));
end

function [csi_est] = csiFormerInfer(model, csi_ls, pre_csi)
    % 保存原始 csi_ls 的尺寸
    orig_shape = size(csi_ls);
    
    % 拼接实部和虚部，得到新的维度 [orig_shape, 2]
    csi_ls_cat = cat(ndims(csi_ls)+1, real(csi_ls), imag(csi_ls));
    
    % 转换为 Python numpy 数组
    csi_ls_py = py.numpy.array(csi_ls_cat);
    pre_csi_py = py.numpy.array(pre_csi);
    
    % 调用 Python 端的推断函数
    csi_est_py = py.csiFormer.infer(model, csi_ls_py, pre_csi_py);
    
    % 将 Python numpy 输出转换为 MATLAB 数组
    csi_est = double(py.array.array('d', py.numpy.nditer(csi_est_py)));
    
    % 重构为与拼接后输入相同的 shape: [orig_shape, 2]
    csi_est = reshape(csi_est, [orig_shape, 2]);

    csi_est = complex(csi_est(:,:,:,:,1), csi_est(:,:,:,:,2));
end

function [csi_est] = csiFormerLiteInfer(model, csi_ls, pre_csi)
    % 保存原始 csi_ls 的尺寸
    orig_shape = size(csi_ls);
    
    % 拼接实部和虚部，得到新的维度 [orig_shape, 2]
    csi_ls_cat = cat(ndims(csi_ls)+1, real(csi_ls), imag(csi_ls));
    
    % 转换为 Python numpy 数组
    csi_ls_py = py.numpy.array(csi_ls_cat);
    pre_csi_py = py.numpy.array(pre_csi);
    
    % 调用 Python 端的推断函数
    csi_est_py = py.csiFormerLite.infer(model, csi_ls_py, pre_csi_py);
    
    % 将 Python numpy 输出转换为 MATLAB 数组
    csi_est = double(py.array.array('d', py.numpy.nditer(csi_est_py)));
    
    % 重构为与拼接后输入相同的 shape: [orig_shape, 2]
    csi_est = reshape(csi_est, [orig_shape, 2]);

    csi_est = complex(csi_est(:,:,:,:,1), csi_est(:,:,:,:,2));
end

function [csi_est] = csiFormerStudentInfer(model, csi_ls, pre_csi)
    % 保存原始 csi_ls 的尺寸
    orig_shape = size(csi_ls);
    
    % 拼接实部和虚部，得到新的维度 [orig_shape, 2]
    csi_ls_cat = cat(ndims(csi_ls)+1, real(csi_ls), imag(csi_ls));
    
    % 转换为 Python numpy 数组
    csi_ls_py = py.numpy.array(csi_ls_cat);
    pre_csi_py = py.numpy.array(pre_csi);
    
    % 调用 Python 端的推断函数
    csi_est_py = py.csiFormerStudent.infer(model, csi_ls_py, pre_csi_py);
    
    % 将 Python numpy 输出转换为 MATLAB 数组
    csi_est = double(py.array.array('d', py.numpy.nditer(csi_est_py)));
    
    % 重构为与拼接后输入相同的 shape: [orig_shape, 2]
    csi_est = reshape(csi_est, [orig_shape, 2]);

    csi_est = complex(csi_est(:,:,:,:,1), csi_est(:,:,:,:,2));
end

function [csi_est] = channelformerInfer(model, csi_ls)
    % 保存原始 csi_ls 的尺寸
    orig_shape = size(csi_ls);
    
    % 拼接实部和虚部，得到新的维度 [orig_shape, 2]
    csi_ls_cat = cat(ndims(csi_ls)+1, real(csi_ls), imag(csi_ls));
    
    % 转换为 Python numpy 数组
    csi_ls_py = py.numpy.array(csi_ls_cat);
    
    % 调用 Python 端的推断函数
    csi_est_py = py.channelformer.infer(model, csi_ls_py);
    
    % 将 Python numpy 输出转换为 MATLAB 数组
    csi_est = double(py.array.array('d', py.numpy.nditer(csi_est_py)));
    
    % 重构为与拼接后输入相同的 shape: [orig_shape, 2]
    csi_est = reshape(csi_est, [orig_shape, 2]);

    csi_est = complex(csi_est(:,:,:,:,1), csi_est(:,:,:,:,2));
end

function [csi_est] = ceDnnInfer(model, csi_ls)
    % 保存原始 csi_ls 的尺寸
    orig_shape = size(csi_ls);  % [nsubc, nsym, ntx, nrx]
    
    % 拼接实部和虚部，新增加的最后一维为2：即 [orig_shape, 2]
    csi_ls_cat = cat(ndims(csi_ls)+1, real(csi_ls), imag(csi_ls));
    
    % 转换为 Python numpy 数组
    csi_ls_py = py.numpy.array(csi_ls_cat);
    
    % 调用 Python 端的推断函数
    csi_est_py = py.ceDnn.infer(model, csi_ls_py);
    
    % 将 Python numpy 输出转换为 MATLAB 数组（此时数据是一维向量）
    csi_est = double(py.array.array('d', py.numpy.nditer(csi_est_py)));
    
    % 重构为与拼接后输入相同的 shape: [orig_shape, 2]
    csi_est = reshape(csi_est, [orig_shape, 2]);

    csi_est = complex(csi_est(:,:,:,:,1), csi_est(:,:,:,:,2));
end

function [equalized_signal] = eqDnnInfer(model, csi_est, rx_signal)
    % 获取参数维度
    [nsubc, nsym, ntx, nrx] = size(csi_est);  % [nsubc, nsym, ntx, nrx]
    
    % 拼接 csi_est 的实部和虚部，在最后一维增加一个维度2
    csi_est_cat = cat(ndims(csi_est)+1, real(csi_est), imag(csi_est));
    % 同理，拼接 rx_signal 的实部和虚部
    rx_signal_cat = cat(ndims(rx_signal)+1, real(rx_signal), imag(rx_signal));
    
    % 转换为 Python numpy 数组
    csi_est_py = py.numpy.array(csi_est_cat);
    rx_signal_py = py.numpy.array(rx_signal_cat);
    
    % 调用 Python 端的推断函数
    eq_sigal_py = py.eqDnn.infer(model, csi_est_py, rx_signal_py);
    
    % 将 Python numpy 输出转换为 MATLAB 数组（此时数据为一维向量）
    eq_sigal = double(py.array.array('d', py.numpy.nditer(eq_sigal_py)));
    
    % 重构为 [nsubc, nsym, ntx, 2]
    eq_sigal = reshape(eq_sigal, [nsubc, nsym, ntx, 2]);
    
    % 合成复数矩阵：将最后一维中的第1和第2部分分别作为实部和虚部
    equalized_signal = complex(eq_sigal(:,:,:,1), eq_sigal(:,:,:,2));
end

function [equalized_signal] = eqDnnProInfer(model, csi_est, rx_signal)
    % 获取参数维度
    [nsubc, nsym, ntx, nrx] = size(csi_est);  % [nsubc, nsym, ntx, nrx]
    
    % 拼接 csi_est 的实部和虚部，在最后一维增加一个维度2
    csi_est_cat = cat(ndims(csi_est)+1, real(csi_est), imag(csi_est));
    % 同理，拼接 rx_signal 的实部和虚部
    rx_signal_cat = cat(ndims(rx_signal)+1, real(rx_signal), imag(rx_signal));
    
    % 转换为 Python numpy 数组
    csi_est_py = py.numpy.array(csi_est_cat);
    rx_signal_py = py.numpy.array(rx_signal_cat);
    
    % 调用 Python 端的推断函数
    eq_sigal_py = py.eqDnnPro.infer(model, csi_est_py, rx_signal_py);
    
    % 将 Python numpy 输出转换为 MATLAB 数组（此时数据为一维向量）
    eq_sigal = double(py.array.array('d', py.numpy.nditer(eq_sigal_py)));
    
    % 重构为 [nsubc, nsym, ntx, 2]
    eq_sigal = reshape(eq_sigal, [nsubc, nsym, ntx, 2]);
    
    % 合成复数矩阵：将最后一维中的第1和第2部分分别作为实部和虚部
    equalized_signal = complex(eq_sigal(:,:,:,1), eq_sigal(:,:,:,2));
end

function [equalized_signal] = eqDnnProStudentInfer(model, csi_est, rx_signal)
    % 获取参数维度
    [nsubc, nsym, ntx, nrx] = size(csi_est);  % [nsubc, nsym, ntx, nrx]
    
    % 拼接 csi_est 的实部和虚部，在最后一维增加一个维度2
    csi_est_cat = cat(ndims(csi_est)+1, real(csi_est), imag(csi_est));
    % 同理，拼接 rx_signal 的实部和虚部
    rx_signal_cat = cat(ndims(rx_signal)+1, real(rx_signal), imag(rx_signal));
    
    % 转换为 Python numpy 数组
    csi_est_py = py.numpy.array(csi_est_cat);
    rx_signal_py = py.numpy.array(rx_signal_cat);
    
    % 调用 Python 端的推断函数
    eq_sigal_py = py.eqDnnProStudent.infer(model, csi_est_py, rx_signal_py);
    
    % 将 Python numpy 输出转换为 MATLAB 数组（此时数据为一维向量）
    eq_sigal = double(py.array.array('d', py.numpy.nditer(eq_sigal_py)));
    
    % 重构为 [nsubc, nsym, ntx, 2]
    eq_sigal = reshape(eq_sigal, [nsubc, nsym, ntx, 2]);
    
    % 合成复数矩阵：将最后一维中的第1和第2部分分别作为实部和虚部
    equalized_signal = complex(eq_sigal(:,:,:,1), eq_sigal(:,:,:,2));
end


function [equalized_signal] = deeprxInfer(model, csi_ls, tx_signal, rx_signal)
    % 获取参数维度
    [nsubc, nsym, ntx, nrx] = size(csi_ls);  % [nsubc, nsym, ntx, nrx]
    
    % 拼接 csi_est 的实部和虚部，在最后一维增加一个维度2
    csi_ls_cat = cat(ndims(csi_ls)+1, real(csi_ls), imag(csi_ls));
    % 同理，拼接 rx_signal 的实部和虚部
    rx_signal_cat = cat(ndims(rx_signal)+1, real(rx_signal), imag(rx_signal));

    tx_signal_cat = cat(ndims(tx_signal)+1, real(tx_signal), imag(tx_signal));

    % 转换为 Python numpy 数组
    csi_ls_py = py.numpy.array(csi_ls_cat);
    rx_signal_py = py.numpy.array(rx_signal_cat);
    tx_signal_py = py.numpy.array(tx_signal_cat);

    
    % 调用 Python 端的推断函数
    eq_sigal_py = py.deepRx.infer(model, csi_ls_py, tx_signal_py, rx_signal_py);
    
    % 将 Python numpy 输出转换为 MATLAB 数组（此时数据为一维向量）
    eq_sigal = double(py.array.array('d', py.numpy.nditer(eq_sigal_py)));
    
    % 重构为 [nsubc, nsym, ntx, 2]
    eq_sigal = reshape(eq_sigal, [nsubc, nsym, ntx, 2]);
    
    % 合成复数矩阵：将最后一维中的第1和第2部分分别作为实部和虚部
    equalized_signal = complex(eq_sigal(:,:,:,1), eq_sigal(:,:,:,2));
end

%% 参数设置
% 系统参数配置
numSubc = 256;                                                                           % FFT 长度
numGuardBands = [16;15];                                                                 % 左右保护带
numPilot = (numSubc-sum(numGuardBands)-1)/4;                                             % 每根天线的导频子载波
numTx = 2;                                                                               % 发射天线数量
numRx = 2;                                                                               % 接收天线数量
numSym = 14;                                                                             % 每帧 OFDM 符号数
numStream = 2;                                                                           % 数据流个数
cpLength = numSubc/4;                                                                    % 循环前缀长度


% 调制参数配置
M = 4;                                                                                   % QPSK 调制（M=4）

% 信道模型配置
sampleRate = 15.36e6;                                                                     % 采样率
t_rms = 2e-6/sqrt(2);                                                                    % 均方根时延
power_r = 2;                                                                             % 导频功率
delta_f = sampleRate/numSubc;                                                            % 子载波间隔
pathDelays = [0, 30, 70, 90, 110, 190, 410] * 1e-9;                                       % 路径时延
averagePathGains = [0, -1.0, -2.0, -3.0, -8.0, -17.2, -20.8];                             % 平均路径增益
maxDopplerShift = 5.5;                                                                    % 最大多普勒频移

% 信号导频分布配置
validSubcIndices = setdiff((numGuardBands(1)+1):(numSubc-numGuardBands(2)), numSubc/2+1);
numValidSubc = length(validSubcIndices);

% 导频子载波配置
pilotIndicesAnt1 = (numGuardBands(1)+1:4:numSubc-numGuardBands(2))';                        % 发射天线1的导频子载波位置
pilotIndicesAnt2 = (numGuardBands(1)+2:4:numSubc-numGuardBands(2))';                        % 发射天线2的导频子载波位置
pilotIndicesAnt2(end) = pilotIndicesAnt1(end)-1;
pilotIndicesAnt1 = pilotIndicesAnt1(pilotIndicesAnt1~=numSubc/2+1);                         % DC子载波不允许设为导频，因此去除
pilotIndicesAnt2 = pilotIndicesAnt2(pilotIndicesAnt1~=numSubc/2+1);                         % DC子载波不允许设为导频，因此去除


% 构造 PilotCarrierIndices (3D 矩阵, NPilot-by-NSym-by-NT)
pilotIndices = zeros(numPilot, numSym, numTx);
pilotIndices(:, :, 1) = repmat(pilotIndicesAnt1, 1, numSym); % 天线 1
pilotIndices(:, :, 2) = repmat(pilotIndicesAnt2, 1, numSym); % 天线 2

% 数据子载波配置
dataIndices = setdiff((numGuardBands(1)+1):(numSubc-numGuardBands(2)),[unique(pilotIndices); numSubc/2+1]);
numDataSubc = length(dataIndices);
numFrameSymbols = numDataSubc * numSym * numTx;

% 已知 validSubcIndices 与 dataIndices（均为原始FFT子载波编号）
% 求 dataIndices 在 validSubcIndices 中的相对位置
[~, valid2DataIndices] = ismember(dataIndices, validSubcIndices);

% OFDM解调器
ofdmDemod = comm.OFDMDemodulator('FFTLength', numSubc, ...
                                  'NumGuardBandCarriers', numGuardBands, ...
                                  'RemoveDCCarrier', true,...
                                  'NumSymbols', numSym, ...
                                  'PilotOutputPort', true, ...
                                  'PilotCarrierIndices', pilotIndices, ...
                                  'CyclicPrefixLength', cpLength, ...
                                  'NumReceiveAntennas', numRx);
% OFDM调制器 两种方式均可
ofdmMod = comm.OFDMModulator('FFTLength', numSubc, ...
                             'NumGuardBandCarriers', numGuardBands, ...
                             'InsertDCNull', true,...
                             'NumSymbols', numSym, ...
                             'PilotInputPort', true, ...
                             'PilotCarrierIndices', pilotIndices, ...
                             'CyclicPrefixLength', cpLength, ...
                             'NumTransmitAntennas', numTx);


minSeed = 0;
maxSeed = 2^32 - 1;   % 3.4625e+09 2.4608e+09 4.0359e+09 2.7777e+09  3.4992e+09 2.1401e+09  2.3742e+09

% 3.3813e+09 3.4070e+09

seed = randi([minSeed, maxSeed]);
seed = 102832043;
% 每个SNR统计子帧的数量
numCountFrame = 25;   

seed
% 信道模型
mimoChannel = comm.MIMOChannel(...
    'SampleRate', sampleRate, ...
    'SpatialCorrelationSpecification', 'None',...
    'PathDelays', pathDelays, ...
    'AveragePathGains', averagePathGains, ...
    'MaximumDopplerShift', maxDopplerShift, ...
    'NumTransmitAntennas', numTx, ...
    'NumReceiveAntennas', numRx, ...
    'FadingDistribution', 'Rayleigh', ...
    'PathGainsOutputPort', true,...
    'RandomStream', 'mt19937ar with seed', ...  % 使用固定种子的随机数流
    'Seed', seed);   % 开启路径增益输出

mimoChannelInfo = info(mimoChannel);
pathFilters = mimoChannelInfo.ChannelFilterCoefficients;
toffset = mimoChannelInfo.ChannelFilterDelay;

%% 评价体系
% snr数组
snrValues = 0:3:30;

% SER 数据对比
ser_ideal_zf = zeros(length(snrValues), 3);
ser_ideal_mmse = zeros(length(snrValues), 3);
ser_ideal_eqDnnPro = zeros(length(snrValues), 3);

ser_ls_zf = zeros(length(snrValues), 3);
ser_ls_mmse = zeros(length(snrValues), 3);
ser_ls_eqDnnPro = zeros(length(snrValues), 3);

ser_mmse_zf = zeros(length(snrValues), 3);
ser_mmse_mmse = zeros(length(snrValues), 3);

ser_csiEncoder_mmse = zeros(length(snrValues), 3);

ser_csiFormer_zf = zeros(length(snrValues), 3);
ser_csiFormer_mmse = zeros(length(snrValues), 3);
ser_csiFormer_eqDnnPro = zeros(length(snrValues), 3);

ser_channelformer_mmse = zeros(length(snrValues), 3);

ser_deeprx = zeros(length(snrValues), 3);

ser_csiFormerStudent_eqDnnProStudent = zeros(length(snrValues), 3);


% 信道估计MSE LOSS 数据对比
mse_csi_ls = zeros(length(snrValues), 1);
mse_csi_mmse = zeros(length(snrValues), 1);
mse_csi_csiEncoder = zeros(length(snrValues), 1);
mse_csi_csiFormer = zeros(length(snrValues), 1);
mse_csi_csiFormerStudent = zeros(length(snrValues), 1);

mse_csi_channelformer = zeros(length(snrValues), 1);


% 信道均衡MSE LOSS 数据对比
% TODO
                                     
csiPreTemp = zeros(3, numValidSubc, numSym, numTx, numRx);

for idx = 1:length(snrValues)
    snr = snrValues(idx);

    % errorRate对象创建
    er_ideal_zf = comm.ErrorRate;
    er_ideal_mmse = comm.ErrorRate;
    er_ideal_eqDnn = comm.ErrorRate;
    
    er_ls_zf = comm.ErrorRate;
    er_ls_mmse = comm.ErrorRate;
    er_ls_eqDnnPro = comm.ErrorRate;

    er_mmse_zf = comm.ErrorRate;
    er_mmse_mmse = comm.ErrorRate;

    er_csiEncoder_mmse = comm.ErrorRate;

    er_csiFormer_zf = comm.ErrorRate;
    er_csiFormer_mmse = comm.ErrorRate;
    er_csiFormer_eqDnnPro = comm.ErrorRate;
    
    er_csiFormerStudent_eqDnnProStudent = comm.ErrorRate;

    er_channelformer_mmse = comm.ErrorRate;
    er_deeprx = comm.ErrorRate;


    loss_ls = 0;            % 用于累计LS     的MSE LOSS    
    loss_mmse = 0;          % 用于累计MMSE   的MSE LOSS
    loss_lmmse = 0;         % 用于累计LMMSE  的MSE LOSS
    loss_csiFormer = 0;     % 用于累计AI     的MSE LOSS
    loss_csiEncoder = 0;    % 用于累计AI     的MSE LOSS
    loss_csiFormerStudent = 0; 
    loss_channelformer = 0;

    for frame = -1:numCountFrame
        % 数据符号生成
        txStream = randi([0 M-1], numFrameSymbols, 1); 
        dataSignal = pskmod(txStream, M);  % 调制后的符号为复数形式
        dataSignal = reshape(dataSignal, numDataSubc, numSym, numTx);
        pilotSignal = repmat(1+1i, numPilot, numSym, numTx);
        % OFDM 调制
        txSignal = ofdmMod(dataSignal, pilotSignal);
        % 通过信道模型获取接收信号和路径增益[总样本数, N_path, numTransmitAntennas, numReceiveAntennas]
        [airSignal, pathGains] = mimoChannel(txSignal);    
        % 去滤波器时延
        airSignal = [airSignal((toffset+1):end,:); zeros(toffset,2)];
        % 噪声
        [airSignal, noiseVar] = awgn(airSignal, snr, "measured");
        noiseVar = noiseVar * numSubc;

        % OFDM 解调
        [rxDataSignal, rxPilotSignal] = ofdmDemod(airSignal);
        % 完美信道估计
        hValid = ofdmChannelResponse(pathGains, pathFilters, numSubc, cpLength, validSubcIndices, toffset);
        hPerfect = hValid(valid2DataIndices,:,:,:);

        % 计算导频处信道估计
        csi_ls = zeros(numSubc, numSym, numTx, numRx);
        for tx = 1:numTx
            for rx = 1:numRx
                csi_ls(pilotIndices(:,1,tx),:,tx,rx) = rxPilotSignal(:,:,tx,rx) ./ pilotSignal(:, :, tx);
            end
        end

        csiPreTemp(1,:,:,:,:,:) = csiPreTemp(2,:,:,:,:,:);
        csiPreTemp(2,:,:,:,:,:) = csiPreTemp(3,:,:,:,:,:);
        csiPreTemp(3,:,:,:,:,1) = real(csi_ls(validSubcIndices,:,:,:));
        csiPreTemp(3,:,:,:,:,2) = imag(csi_ls(validSubcIndices,:,:,:));

        if frame < 1
            continue;
        end

        % 发射信号收集
        originSignal = zeros(numSubc, numSym, numTx);
        originSignal(dataIndices,:,:) = dataSignal;
        for tx = 1:numTx
            originSignal(pilotIndices(:,1,tx),:,tx) = pilotSignal(:, :, tx);
        end   
        % 接收信号采集
        finalSignal = zeros(numSubc, numSym, numRx);
        finalSignal(dataIndices,:,:) = rxDataSignal(:,:,:);
        for rx = 1:numRx
            for tx = 1:numTx
                finalSignal(pilotIndices(:,1,tx),:,rx) = rxPilotSignal(:,:,tx,rx);
            end
        end
        
        %% AI信道估计与均衡
        % 计算导频处信道估计
        csi_ls = zeros(numSubc, numSym, numTx, numRx);
        for tx = 1:numTx
            for rx = 1:numRx
                csi_ls(pilotIndices(:,1,tx),:,tx,rx) = rxPilotSignal(:,:,tx,rx) ./ pilotSignal(:, :, tx);
            end
        end

        %% 信道估计
        % csiEncoder估计
        csiEncoder_valid = csiEncoderInfer(csiEncoderModel, csi_ls(validSubcIndices,:,:,:));
        csiEncoder_data = csiEncoder_valid(valid2DataIndices, :,:,:);
        loss_csiEncoder = loss_csiEncoder + immse(csiEncoder_data, hPerfect);

        % csiFormer估计
        csiFormer_valid = csiFormerInfer(csiFormerModel,csi_ls(validSubcIndices,:,:,:), csiPreTemp(1:2,:,:,:,:,:));
        % csiFormer_valid = csiFormerStudentInfer(csiFormerStudentModel,csi_ls(validSubcIndices,:,:,:), csiPreTemp(1:2,:,:,:,:,:));
        csiFormer_data = csiFormer_valid(valid2DataIndices,:,:,:);
        loss_csiFormer = loss_csiFormer + immse(csiFormer_data, hPerfect);
        
        % csiFormerStudent估计
        csiFormerStudent_valid = csiFormerStudentInfer(csiFormerStudentModel,csi_ls(validSubcIndices,:,:,:), csiPreTemp(1:2,:,:,:,:,:));
        csiFormerStudent_data = csiFormerStudent_valid(valid2DataIndices,:,:,:);
        loss_csiFormerStudent = loss_csiFormerStudent + immse(csiFormerStudent_data, hPerfect);
        
        % channelformer估计
        channelformer_valid = channelformerInfer(channelformerModel,csi_ls(validSubcIndices,:,:,:));
        channelformer_data = channelformer_valid(valid2DataIndices,:,:,:);
        loss_channelformer = loss_channelformer + immse(channelformer_data, hPerfect);

        % LS信道估计
        csi_ls_ce = lsChannelEst(rxPilotSignal, pilotSignal, dataIndices, pilotIndices);
        loss_ls = loss_ls + immse(csi_ls_ce, hPerfect);

        % MMSE信道估计
        csi_mmse = mmseChannelEst(rxPilotSignal, pilotSignal, dataIndices, pilotIndices, t_rms, delta_f, power_r, noiseVar);
        loss_mmse = loss_mmse + immse(csi_mmse, hPerfect);

        
        %% 信道均衡

        % deeprx 均衡
        eqSignal_deeprx = deeprxInfer(deeprxModel, csi_ls(validSubcIndices,:,:,:), originSignal(validSubcIndices,:,:), finalSignal(validSubcIndices,:,:));
        eqSignal_deeprx = eqSignal_deeprx(valid2DataIndices,:,:);
        rxStream_deeprx = pskdemod(reshape(eqSignal_deeprx, [], 1), M);
        ser_deeprx(idx, :) = er_deeprx(txStream, rxStream_deeprx);

        % 完美信道 eqDnnPro 信道均衡 
        %                         eqDnnInfer(eqDnnModel, hValid, finalSignal(validSubcIndices,:,:));
        eqSignal_ideal_eqDnnPro = eqDnnProInfer(eqDnnProModel, hValid, finalSignal(validSubcIndices,:,:));
        eqSignal_ideal_eqDnnPro = eqSignal_ideal_eqDnnPro(valid2DataIndices,:,:);
        rxStream_ideal_eqDnnPro = pskdemod(reshape(eqSignal_ideal_eqDnnPro, [], 1), M);
        ser_ideal_eqDnnPro(idx, :) = er_ideal_eqDnn(txStream, rxStream_ideal_eqDnnPro);
        
        % 完美信道 MMSE均衡
        eqSignal_ideal_mmse = ofdmEqualize(rxDataSignal,reshape(hPerfect,[],numTx,numRx), noiseVar, Algorithm="mmse");
        rxStream_ideal_mmse = pskdemod(reshape(eqSignal_ideal_mmse, [], 1), M);
        ser_ideal_mmse(idx, :) = er_ideal_mmse(txStream, rxStream_ideal_mmse);
        
        % AI csiEncoder MMSE 信道均衡
        eqSignal_csiEncoder_mmse = myMMSEequalize(csiEncoder_data, rxDataSignal, noiseVar);
        rxStream_Encoder_mmse = pskdemod(reshape(eqSignal_csiEncoder_mmse, [], 1), M);
        ser_csiEncoder_mmse(idx, :) = er_csiEncoder_mmse(txStream, rxStream_Encoder_mmse);

        % AI csiFormer ZF 信道均衡
        % eqSignal_csiFormer_ls = myZFequalize(csiFormer_data, rxDataSignal);
        % rxStream_csiFormer_ls = pskdemod(reshape(eqSignal_csiFormer_ls, [], 1), M);
        % ser_csiFormer_zf(idx, :) = er_csiFormer_zf(txStream, rxStream_csiFormer_ls);

        % AI csiFormer MMSE 信道均衡
        eqSignal_csiFormer_mmse = myMMSEequalize(csiFormer_data, rxDataSignal, noiseVar);
        rxStream_csiFormer_mmse = pskdemod(reshape(eqSignal_csiFormer_mmse, [], 1), M);
        ser_csiFormer_mmse(idx, :) = er_csiFormer_mmse(txStream, rxStream_csiFormer_mmse);

        % AI channelformer+ MMSE 信道均衡
        eqSignal_channelformer_mmse = myMMSEequalize(channelformer_data, rxDataSignal, noiseVar);
        rxStream_channelformer_mmse = pskdemod(reshape(eqSignal_channelformer_mmse, [], 1), M);
        ser_channelformer_mmse(idx, :) = er_channelformer_mmse(txStream, rxStream_channelformer_mmse);

        % AI csiFormer+eqDnn 信道均衡
        eqSignal_csiFormer_eqDnnPro = eqDnnProInfer(eqDnnProModel, csiFormer_valid, finalSignal(validSubcIndices,:,:));
        % eqSignal_csiFormer_eqDnnPro = eqDnnProStudentInfer(eqDnnProStudentModel, csiFormer_valid, finalSignal(validSubcIndices,:,:));
        eqSignal_csiFormer_eqDnnPro = eqSignal_csiFormer_eqDnnPro(valid2DataIndices,:,:);
        rxStream_csiFormer_eqDnnPro = pskdemod(reshape(eqSignal_csiFormer_eqDnnPro, [], 1), M);
        ser_csiFormer_eqDnnPro(idx, :) = er_csiFormer_eqDnnPro(txStream, rxStream_csiFormer_eqDnnPro);

        % AI csiFormerStudent+eqDnnProStudent 信道均衡
        eqSignal_csiFormerStudent_eqDnnProStudent = eqDnnProStudentInfer(eqDnnProStudentModel, csiFormer_valid, finalSignal(validSubcIndices,:,:));
        eqSignal_csiFormerStudent_eqDnnProStudent = eqSignal_csiFormerStudent_eqDnnProStudent(valid2DataIndices,:,:);
        rxStream_csiFormerStudent_eqDnnProStudent = pskdemod(reshape(eqSignal_csiFormerStudent_eqDnnProStudent, [], 1), M);
        ser_csiFormerStudent_eqDnnProStudent(idx, :) = er_csiFormerStudent_eqDnnProStudent(txStream, rxStream_csiFormerStudent_eqDnnProStudent);
        

        % % LS信道 ZF均衡
        % rxSignal_ls_zf = myZFequalize(csi_ls, rxDataSignal);
        % rxStream_ls_zf = pskdemod(reshape(rxSignal_ls_zf, [],1), M);
        % ser_ls_zf(idx, :) = er_ls_zf(txStream, rxStream_ls_zf);

        % LS信道 MMSE均衡
        rxSignal_ls_mmse = myMMSEequalize(csi_ls_ce, rxDataSignal, noiseVar);
        rxStream_ls_mmse = pskdemod(reshape(rxSignal_ls_mmse,[],1),M);
        ser_ls_mmse(idx, :) = er_ls_mmse(txStream, rxStream_ls_mmse);

        %
        % rxSignal_ls_eqDnnPro = eqDnnProInfer(eqDnnProModel, csi_ls, finalSignal(validSubcIndices,:,:));
        % rxStream_ls_eqDnnPro = pskdemod(reshape(rxSignal_ls_eqDnnPro,[],1),M);
        % ser_ls_eqDnnPro(idx, :) = er_ls_eqDnnPro(txStream, rxStream_ls_mmse);

        % % MMSE信道 ZF均衡
        % rxSignal_mmse_zf = myZFequalize(csi_mmse, rxDataSignal);
        % rxStream_mmse_zf = pskdemod(reshape(rxSignal_mmse_zf,[],1),M);
        % ser_mmse_zf(idx, :) = er_mmse_zf(txStream, rxStream_mmse_zf);

        % MMSE信道 MMSE均衡
        rxSignal_mmse_mmse = myMMSEequalize(csi_mmse, rxDataSignal, noiseVar);
        rxStream_mmse_mmse = pskdemod(reshape(rxSignal_mmse_mmse,[],1),M);
        ser_mmse_mmse(idx, :) = er_mmse_mmse(txStream, rxStream_mmse_mmse);

    end
    % 对每种算法在当前 SNR 的所有帧计算平均 MSE
    mse_csi_ls(idx) = loss_ls / numCountFrame;
    mse_csi_mmse(idx) = loss_mmse / numCountFrame;
    mse_csi_csiFormer(idx) = loss_csiFormer / numCountFrame;
    mse_csi_csiEncoder(idx) = loss_csiEncoder / numCountFrame;
    mse_csi_csiFormerStudent(idx) = loss_csiFormerStudent / numCountFrame;
    mse_csi_channelformer(idx) = loss_channelformer / numCountFrame;
end

%% 保存数据
% 生成时间戳字符串，例如 '20250211_153045'
timestampStr = string(datetime('now','Format','yyyyMMdd_HHmmss'));

% 拼接文件名，例如 '20250211_153045.mat'
filename = timestampStr+'.mat';

save(filename, ...
    'seed', ...
    'ser_ideal_zf', ...
    'ser_ideal_mmse', ...
    'ser_ideal_eqDnnPro',...
    'ser_ls_zf', ...
    'ser_ls_mmse', ...
    'ser_ls_eqDnnPro', ...
    'ser_mmse_zf', ...
    'ser_mmse_mmse', ...
    'ser_csiEncoder_mmse', ...
    'ser_csiFormer_zf', ...
    'ser_csiFormer_mmse', ...
    'ser_csiFormer_eqDnnPro', ...
    'ser_csiFormerStudent_eqDnnProStudent', ...
    'mse_csi_ls', ...
    'mse_csi_mmse', ...
    'mse_csi_csiEncoder', ...
    'mse_csi_csiFormer', ...
    'mse_csi_csiFormerStudent', ...
    'ser_deeprx', ...
    'ser_channelformer_mmse', ...
    'mse_csi_channelformer');

%% 图形绘制部分
% 手动定义对比鲜明的颜色矩阵
colors = [
    0.0000, 0.4470, 0.7410;  % 蓝色 (Blue)
    0.9290, 0.6940, 0.1250;  % 黄色 (Yellow)
    0.4660, 0.6740, 0.1880;  % 绿色 (Green)
    0.6350, 0.0780, 0.1840;  % 红色 (Red)    
    0.3010, 0.7450, 0.9330;  % 青色 (Cyan)
    0.8500, 0.3250, 0.0980;  % 橙色 (Orange)
    0.4940, 0.1840, 0.5560;  % 紫色 (Purple)

];

% --- 图1：信道估计 MSE LOSS 曲线对比 ---
figure;
hold on;
plot(snrValues, mse_csi_ls,        '-o', 'Color', colors(1,:), 'LineWidth', 1, 'DisplayName', 'LS');
plot(snrValues, mse_csi_mmse,      '-d', 'Color', colors(2,:), 'LineWidth', 1, 'DisplayName', 'MMSE');
plot(snrValues, mse_csi_csiEncoder, '-s', 'Color', colors(3,:), 'LineWidth', 1, 'DisplayName', 'CSIEncoder');
plot(snrValues, mse_csi_channelformer, '-d', 'Color', colors(5,:), 'LineWidth', 1, 'DisplayName', 'Channelformer');
plot(snrValues, mse_csi_csiFormer,  '-p', 'Color', colors(4,:), 'LineWidth', 1, 'DisplayName', 'CSIFormer');


grid on;
xlabel('SNR (dB)');
ylabel('MSE with h_{Perfect}');
title('MSE vs. SNR for Different Channel Estimation Algorithms');
legend('Location', 'best');
hold off;


% --- 图2：信道估计  SER 误码率曲线对比 ---
figure;
hold on;

plot(snrValues, ser_ls_mmse(:,1),        '-o', 'Color', colors(1,:),  'LineWidth', 1, 'DisplayName', 'LS');
plot(snrValues, ser_csiEncoder_mmse(:,1),  '-*', 'Color', colors(2,:),  'LineWidth', 0.5, 'DisplayName', 'CSIEncoder');
plot(snrValues, ser_channelformer_mmse(:,1),  '-s', 'Color', colors(5,:),  'LineWidth', 0.5, 'DisplayName', 'Channelformer');
plot(snrValues, ser_csiFormer_mmse(:,1), '-p', 'Color', colors(4,:), 'LineWidth', 1.5, 'DisplayName', 'CSIFormer');
plot(snrValues, ser_ideal_mmse(:,1),     '-d', 'Color', colors(3,:),  'LineWidth', 1, 'DisplayName', 'Ideal')


grid on;
xlabel('SNR (dB)');
ylabel('Symbol Error Rate (SER)');
title('SER vs. SNR for Different Channel Estimation Algorithms');
legend('Location', 'best');
set(gca, 'YScale', 'log');  % Y轴使用对数刻度
hold off;

% --- 图3：信道均衡 SER 误码率曲线 ---
figure;
hold on;

plot(snrValues, ser_ls_mmse(:,1),        '-p', 'Color', colors(1,:),  'LineWidth', 1, 'DisplayName', 'LS+MMSE');
plot(snrValues, ser_mmse_mmse(:,1),      '-*', 'Color', colors(2,:),  'LineWidth', 1, 'DisplayName', 'LMMSE+MMSE');
plot(snrValues, ser_ideal_mmse(:,1),     '-s', 'Color', colors(3,:),  'LineWidth', 1, 'DisplayName', 'Ideal+MMSE');
plot(snrValues, ser_deeprx(:,1),     '-d', 'Color', colors(5,:),  'LineWidth', 1, 'DisplayName', 'DeepRx');
plot(snrValues, ser_ideal_eqDnnPro(:,1), '-v', 'Color', colors(4,:),  'LineWidth', 1.5, 'DisplayName', 'Ideal+EQAttentiion');


grid on;
xlabel('SNR (dB)');
ylabel('Symbol Error Rate (SER)');
title('SER vs. SNR for Different Channel Equalization Algorithms');
legend('Location', 'best');
set(gca, 'YScale', 'log');  % Y轴使用对数刻度
hold off;

% --- 图4：信道估计与信道均衡 SER 误码率曲线 ---
figure;
hold on;

plot(snrValues, ser_ls_mmse(:,1),        '-p', 'Color', colors(1,:),  'LineWidth', 1.5, 'DisplayName', 'LS+MMSE');
plot(snrValues, ser_mmse_mmse(:,1),      '-*', 'Color', colors(2,:),  'LineWidth', 1.5, 'DisplayName', 'MMSE+MMSE');
plot(snrValues, ser_deeprx(:,1),     '-d', 'Color', colors(7,:),  'LineWidth', 1, 'DisplayName', 'DeepRx')
plot(snrValues, ser_csiFormer_mmse(:,1), '--s', 'Color', colors(6,:), 'LineWidth', 1.5, 'DisplayName', 'CSIFormer+MMSE');
plot(snrValues, ser_csiFormer_eqDnnPro(:,1),'--d', 'Color', colors(4,:), 'LineWidth', 1.5, 'DisplayName', 'JointCEEQ');
plot(snrValues, ser_ideal_mmse(:,1),     '-s', 'Color', colors(5,:),  'LineWidth', 1.5, 'DisplayName', 'Ideal+MMSE');
plot(snrValues, ser_ideal_eqDnnPro(:,1),    '-v', 'Color', colors(3,:),  'LineWidth', 1.5, 'DisplayName', 'Ideal+EQAttention');

grid on;
xlabel('SNR (dB)');
ylabel('Symbol Error Rate (SER)');
title('SER vs. SNR for Different Channel Estimation and Equalization Algorithms');
legend('Location', 'best');
set(gca, 'YScale', 'log');  % Y轴使用对数刻度
hold off;


% --- 图5：模型压缩与推理加速 MSE LOSS 曲线对比 ---
figure;
hold on;
plot(snrValues, mse_csi_ls, '-o', 'Color', colors(1,:), 'LineWidth', 1, 'DisplayName', 'LS');
plot(snrValues, mse_csi_csiEncoder, '-^', 'Color', colors(2,:), 'LineWidth', 1, 'DisplayName', 'CSIEncoder');
plot(snrValues, mse_csi_channelformer, '-s', 'Color', colors(5,:), 'LineWidth', 1, 'DisplayName', 'Channelformer');
plot(snrValues, mse_csi_csiFormer,  '-d', 'Color', colors(3,:), 'LineWidth', 1, 'DisplayName', 'CSIFormer');
plot(snrValues, mse_csi_csiFormerStudent,  '-p', 'Color', colors(4,:), 'LineWidth', 1, 'DisplayName', 'CSIFormerStudent');

grid on;
xlabel('SNR (dB)');
ylabel('MSE with h_{Perfect}');
title('MSE vs. SNR for Different Channel Estimation Algorithms');
legend('Location', 'best');
hold off;

% --- 图6：模型压缩与推理加速 SER 误码率曲线 ---
figure;
hold on;
plot(snrValues, ser_ls_mmse(:,1),        '-p', 'Color', colors(1,:),  'LineWidth', 0.5, 'DisplayName', 'LS+MMSE');
plot(snrValues, ser_csiFormer_mmse(:,1), '-s', 'Color', colors(2,:), 'LineWidth', 0.5, 'DisplayName', 'CSIFormer+MMSE');
plot(snrValues, ser_deeprx(:,1),    '-v', 'Color', colors(6,:),  'LineWidth', 0.5, 'DisplayName', 'DeepRx');
plot(snrValues, ser_csiFormerStudent_eqDnnProStudent(:,1),'-d', 'Color', colors(4,:), 'LineWidth', 1.5, 'DisplayName', 'JointCEEQStudent');
plot(snrValues, ser_csiFormer_eqDnnPro(:,1),'-d', 'Color', colors(3,:), 'LineWidth', 0.5, 'DisplayName', 'JointCEEQTeacher');
plot(snrValues, ser_ideal_eqDnnPro(:,1),    '-v', 'Color', colors(5,:),  'LineWidth', 0.5, 'DisplayName', 'Ideal+EQAttention');

grid on;
xlabel('SNR (dB)');
ylabel('Symbol Error Rate (SER)');
title('SER vs. SNR for Different Channel Estimation and Equalization Algorithms');
legend('Location', 'best');
set(gca, 'YScale', 'log');  % Y轴使用对数刻度
hold off;




%% 自定义函数
function [hEst] = lsChannelEst(rxPilotSignal,refPilotSignal, dataIndices, pilotIndices)
    % MIMO-OFDM 信道估计函数
    % 输入：
    %   rxPilotSignal: 接收导频信号(numPilotSubc x numSym x numTx x numRx)
    %   refPilotSignal: 参考导频信号 (numPilotSubc x numSym x numTx)
    %   dataIndices: 数据符号索引
    %   pilotIndices: 导频符号索引(numPilotSubc x numSym x numTx)
    %   CEC.algorithm: 估计算法 ('LS' 或 'MMSE')
    %   CEC.interpType: 插值类型 ('nearest', 'linear', 'cubic', 'spline')
    %   CEC.freqWindow: 频域平均窗口大小;
    %   CEC.timeWindow: 时域平均窗口大小;
    %   
    % 输出：
    %   H_est: 估计的信道响应矩阵 (numSubc x numSym x numTx x numRx)

    % 提取信号维度
    numDataSubc = length(dataIndices);
    [~, numSym, numTx, numRx] = size(rxPilotSignal);

    % 初始化估计的信道响应矩阵和噪声功率
    hEst = zeros(numDataSubc, numSym, numTx, numRx);

    for tx = 1:numTx
        for rx = 1:numRx
            for sym=1:numSym
                % 获取当前发射 - 接收天线对的导频信号
                pilotRxSignal = rxPilotSignal(:,sym,tx,rx);
                pilotRefSignal = refPilotSignal(:, sym, tx);
                H_ls = pilotRxSignal ./ pilotRefSignal; 
                hEst(:, sym, tx, rx)  = interp1(pilotIndices(:,sym,tx), H_ls, dataIndices, 'linear', 'extrap');
            end
        end
    end
end

function hAvg = pilotAveraging(H_ls, freqWindow, timeWindow)
    [numPilotSubc, numSym] = size(H_ls);
    hAvg = zeros(size(H_ls));

    halfFreqWindow = floor(freqWindow / 2);
    halfTimeWindow = floor(timeWindow / 2);

    for sym = 1:numSym
        for subc = 1:numPilotSubc
            startSubc = max(1, subc - halfFreqWindow);
            endSubc = min(numPilotSubc, subc + halfFreqWindow);
            startSym = max(1, sym - halfTimeWindow);
            endSym = min(numSym, sym + halfTimeWindow);

            window = H_ls(startSubc:endSubc, startSym:endSym);
            hAvg(subc, sym) = mean(window(:));
        end
    end
end

function hEst = mmseChannelEst(rxPilotSignal, refPilotSignal, dataIndices, pilotIndices, t_rms, delta_f, power_r, noiseVar)

    D = 1j*2*pi*t_rms;
    df_1 = squeeze(pilotIndices(:,1,1)) - squeeze(pilotIndices(:,1,1))';
    rf_1 = 1./(1+D*df_1*delta_f);
    Rhp_1 = rf_1;
    Rpp_1 = rf_1 + eye(size(df_1,1))*noiseVar/power_r;

    df_2 = squeeze(pilotIndices(:,1,2)) - squeeze(pilotIndices(:,1,2))';
    rf_2 = 1./(1+D*df_2*delta_f);
    Rhp_2 = rf_2;
    Rpp_2 = rf_2 + eye(size(df_2,1))*noiseVar/power_r;

    % 提取信号维度
    numDataSubc = length(dataIndices);
    [~, numSym, numTx, numRx] = size(rxPilotSignal);

    % 初始化估计的信道响应矩阵和噪声功率
    hEst = zeros(numDataSubc, numSym, numTx, numRx);

    for tx = 1:numTx
        for rx = 1:numRx
            for k = 1:numSym
                pilotRxSignal = rxPilotSignal(:,k,tx,rx);
                pilotRefSignal = refPilotSignal(:, k, tx);
                H_ls = pilotRxSignal ./ pilotRefSignal; % []
                if tx==1
                   H_MMSE = Rhp_1*inv(Rpp_1)*H_ls;  
                else
                   H_MMSE = Rhp_2*inv(Rpp_2)*H_ls;
                end
                hEst(:,k,tx,rx) = interp1(pilotIndices(:,k,tx), H_MMSE, dataIndices, 'linear', 'extrap');
            end
        end
    end
end


function [eqSignal, csi] = myZFequalize(H, rxSignal)
    % myZFequalize 自定义零迫（ZF）均衡函数
    %
    % 输入：
    %   H: 信道矩阵，尺寸为 [nsc, nsym, ntx, nrx]
    %      - nsc: 子载波数量
    %      - nsym: OFDM 符号数量
    %      - ntx: 发射天线数
    %      - nrx: 接收天线数
    %
    %   rxSignal: 接收信号，尺寸为 [nsc, nsym, nrx]
    %
    % 输出：
    %   eqSignal: 均衡后估计的发送符号，尺寸为 [nsc, nsym, ntx]
    %   csi: 软信道状态信息，尺寸为 [nsc, nsym, ntx]
    %
    % 算法说明：
    % 对于每个子载波和每个 OFDM 符号，
    % 1. 从 H 中提取对应的信道矩阵，原始尺寸为 [ntx, nrx]，
    %    转置后变为 [nrx, ntx]，符合 y = H * x 的模型。
    % 2. 计算伪逆： x_hat = pinv(H_sub) * y
    % 3. 同时计算 CSI，例如取 (H_sub^H * H_sub) 的对角线并取倒数，
    %    作为各个发射分量的信道“质量”指标。
    
    % 获取各维度大小
    [nsc, nsym, ntx, nrx] = size(H);
    eqSignal = zeros(nsc, nsym, ntx);
    csi = zeros(nsc, nsym, ntx);
    for i = 1:nsc
        for j = 1:nsym
            % 提取当前子载波、符号的信道矩阵（原始尺寸 [ntx, nrx]）
            H_ij = squeeze(H(i, j, :, :));
            % 为符合 y = H_sub * x 模型，转置为 [nrx, ntx]
            H_sub = H_ij.';  
            % 提取接收信号
            y = squeeze(rxSignal(i, j, :));
            
            % 计算ZF均衡矩阵：W_ZF = (H_sub^H * H_sub)^{-1} * H_sub^H
            % 当 H_sub' * H_sub 存在奇异或病态问题时需加保护措施
            temp = H_sub' * H_sub;
            % 对角元素保护
            diag_temp = diag(temp);
            diag_temp(diag_temp < eps) = eps;
            temp = temp - diag(diag(temp)) + diag(diag_temp);
            
            W_zf = inv(temp) * H_sub';
            x_hat = W_zf * y;
            eqSignal(i, j, :) = x_hat;
            
            % 计算 CSI，反映各发射分量的有效增益
            Hhh = H_sub' * H_sub;
            diagVal = diag(Hhh);
            diagVal(diagVal < eps) = eps;
            csi(i, j, :) = 1 ./ diagVal;
        end
    end
end


function [out, csi] = myMMSEequalize(H, rxsignal, noiseVar)
% myMMSEequalize 自定义 MMSE 均衡器
%
% 输入：
%   H: 信道矩阵，尺寸为 [nsc, nsym, ntx, nrx]
%      - nsc: 子载波数量
%      - nsym: OFDM 符号数量
%      - ntx: 发射天线数
%      - nrx: 接收天线数
%
%   rxsignal: 接收信号，尺寸为 [nsc, nsym, nrx]
%
%   noiseVar: 噪声方差，可以为标量或行向量（长度为 nsym）
%
% 输出：
%   out: 均衡后估计的发送符号，尺寸为 [nsc, nsym, ntx]
%   csi: 软信道状态信息，尺寸为 [nsc, nsym, ntx]
%
% 算法说明：
% 对于每个子载波 i 和每个 OFDM 符号 j：
% 1. 提取信道矩阵 H(i,j,:,:)（尺寸 [ntx, nrx]），转置后变为 H_sub (nrx×ntx)
%    对于 y = H_sub * x 模型，y 为 rxsignal(i,j,:)（nrx×1），x 为发射信号 (ntx×1)
% 2. 计算 MMSE 均衡器：
%       x_hat = inv( H_sub^H * H_sub + noiseVar * I ) * H_sub^H * y
% 3. 同时可以计算 CSI，比如取对角线的倒数：
%       csi(i,j,:) = 1./diag(H_sub^H * H_sub + noiseVar * I)
%
% 注意：为了防止除零，需要对非常小的对角线值加以限制。

% 获取尺寸
[nsc, nsym, ntx, nrx] = size(H);

% 初始化输出
out = zeros(nsc, nsym, ntx);
csi = zeros(nsc, nsym, ntx);

% 如果 noiseVar 为标量，则扩展为 nsym 的行向量
if isscalar(noiseVar)
    noiseVarVec = repmat(noiseVar, 1, nsym);
else
    noiseVarVec = noiseVar;
end

% 对每个子载波和每个 OFDM 符号循环
for i = 1:nsc
    for j = 1:nsym
        % 提取当前子载波和符号的信道矩阵，尺寸为 [ntx, nrx]
        H_ij = squeeze(H(i, j, :, :)); % [ntx, nrx]
        % 转置 H，使其符合 y = H_sub * x 模型，得到 H_sub 尺寸 [nrx, ntx]
        H_sub = H_ij.'; 
        
        % 提取当前子载波和符号的接收信号 y，尺寸为 [nrx, 1]
        y = squeeze(rxsignal(i, j, :));
        
        % 构造单位矩阵，尺寸为 [ntx, ntx]
        I = eye(ntx);
        
        % 计算 MMSE 均衡器矩阵：
        %   W = inv( H_sub^H * H_sub + noiseVar * I ) * H_sub^H
        % 注意：noiseVar 对应当前符号 j
        W = pinv(H_sub' * H_sub + noiseVarVec(j) * I) * H_sub';
        
        % 估计发送信号
        x_hat = W * y;
        out(i, j, :) = x_hat;
        
        % 计算 CSI：这里取 H_sub^H * H_sub + noiseVar * I 的对角线元素，并取倒数
        % 对角线元素代表各个发射天线分量的“有效”增益，噪声项可视为权重修正
        Hhh = H_sub' * H_sub + noiseVarVec(j) * I;
        diagVal = diag(Hhh);
        % 防止除零
        diagVal(diagVal < eps) = eps;
        csi(i, j, :) = 1 ./ diagVal;
    end
end

end
