clear;
clc;
%% python脚本加载
miPyPath = 'C:\Users\stone\AppData\Local\Programs\Python\Python312\python.exe';
lenPyPath = 'D:\Python\python.exe';
pyenv('Version', lenPyPath)
% csiModel = py.csiFormer.load_model();
% ceeqModel = py.ceeqFormer.load_model();
eqDnnModel = py.eqDnn.load_model();
ceDnnModel = py.ceDnn.load_model();

function [csi_est] = csiInfer(model, csi_ls, pre_csi)
    % 保存原始 csi_ls 的尺寸
    orig_shape = size(csi_ls);
    
    % 拼接实部和虚部，得到新的维度 [orig_shape, 2]
    csi_ls_cat = cat(ndims(csi_ls)+1, real(csi_ls), imag(csi_ls));
    pre_csi_cat = cat(ndims(pre_csi)+1, real(pre_csi), imag(pre_csi));
    
    % 转换为 Python numpy 数组
    csi_ls_py = py.numpy.array(csi_ls_cat);
    pre_csi_py = py.numpy.array(pre_csi_cat);
    
    % 调用 Python 端的推断函数
    csi_est_py = py.csiFormer.infer(model, csi_ls_py, pre_csi_py);
    
    % 将 Python numpy 输出转换为 MATLAB 数组
    csi_est = double(py.array.array('d', py.numpy.nditer(csi_est_py)));
    
    % 根据原始尺寸（拼接后应为 [orig_shape, 2]）重构 csi_est 的 shape
    csi_est = reshape(csi_est, [orig_shape, 2]);
end

function [csi_est] = ceInfer(model, csi_ls)
    % 保存原始 csi_ls 的尺寸
    orig_shape = size(csi_ls);  % 例如: [52, 14, 2, 2]
    
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

% function [equalized_signal] = ceeqInfer(model, csi_ls, pre_csi, rx_signal)
%     csi_ls = py.numpy.array(cat(ndims(csi_ls)+1, real(csi_ls), imag(csi_ls)));
%     pre_csi = py.numpy.array(cat(ndims(pre_csi)+1, real(pre_csi), imag(pre_csi)));
%     rx_signal = py.numpy.array(cat(ndims(rx_signal)+1, real(rx_signal), imag(rx_signal)));
% 
%     equalized_signal = py.ceeqFormer.infer(model, csi_ls, pre_csi, rx_signal);
% 
%     % 转换 Python numpy 输出为 MATLAB 矩阵
%     equalized_signal = double(py.array.array('d', py.numpy.nditer(equalized_signal)));
%     equalized_signal = reshape(equalized_signal, 52,14,2,2);
%     equalized_signal = complex(equalized_signal(:,:,:,1), equalized_signal(:,:,:,2));
% end

function [equalized_signal] = eqInfer(model, csi_est, rx_signal)
    csi_est = py.numpy.array(cat(ndims(csi_est)+1, real(csi_est), imag(csi_est)));
    rx_signal = py.numpy.array(cat(ndims(rx_signal)+1, real(rx_signal), imag(rx_signal)));
    
    equalized_signal = py.eqDnn.infer(model, csi_est, rx_signal);

    % 转换 Python numpy 输出为 MATLAB 矩阵
    equalized_signal = double(py.array.array('d', py.numpy.nditer(equalized_signal)));
    equalized_signal = reshape(equalized_signal, 224,14,2,2);
    equalized_signal = complex(equalized_signal(:,:,:,1), equalized_signal(:,:,:,2));
end

%% 参数设置
% 系统参数配置
numSubc = 256;                                            % FFT 长度
numGuardBands = [16;15];                                  % 左右保护带
numPilot = (numSubc-sum(numGuardBands)-1)/4;              % 每根天线的导频子载波
numTx = 2;                                                % 发射天线数量
numRx = 2;                                                % 接收天线数量
numSym = 14;                                              % 每帧 OFDM 符号数
numStream = 2;                                            % 数据流个数
cpLength = numSubc/4;                                            % 循环前缀长度

% 调制参数配置
M = 4;                                                    % QPSK 调制（M=4）

% 信道模型配置
sampleRate = 16e6;                                        % 采样率
pathDelays = [0, 30, 70, 90, 110, 190, 410] * 1e-9;       % 路径时延
averagePathGains = [0, -1.0, -2.0, -3.0, -8.0, -17.2, -20.8];                             % 平均路径增益
maxDopplerShift = 5.5;                                    % 最大多普勒频移

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
% OFDM调制器 
ofdmMod = comm.OFDMModulator('FFTLength', numSubc, ...
                             'NumGuardBandCarriers', numGuardBands, ...
                             'InsertDCNull', true,...
                             'NumSymbols', numSym, ...
                             'PilotInputPort', true, ...
                             'PilotCarrierIndices', pilotIndices, ...
                             'CyclicPrefixLength', cpLength, ...
                             'NumTransmitAntennas', numTx);

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
    'PathGainsOutputPort', true);   % 开启路径增益输出

%% 评价体系
snrValues = 0:5:30;
% errorRate对象创建
errorRatePerfectLS = comm.ErrorRate;
errorRatePerfectMMSE = comm.ErrorRate;
errorRateLSZF = comm.ErrorRate;
errorRateLSMMSE = comm.ErrorRate;
errorRateMMSEZF = comm.ErrorRate;
errorRateMMSEMMSE = comm.ErrorRate;
errorRateAI = comm.ErrorRate;
errorRateEQ = comm.ErrorRate;

% SER 数据对比
serPerfectZF = zeros(length(snrValues), 3);
serPerfectMMSE = zeros(length(snrValues), 3);
serLSZF = zeros(length(snrValues), 3);
serLSMMSE = zeros(length(snrValues), 3);
serMMSEZF = zeros(length(snrValues), 3);
serMMSEMMSE = zeros(length(snrValues), 3);
serAI = zeros(length(snrValues), 3);
serAIEQ = zeros(length(snrValues), 3);

% 信道估计MSE LOSS 数据对比
csiLossPerfectLS = zeros(length(snrValues), 1);
csiLossPerfectMMSE = zeros(length(snrValues), 1);
csiLossPerfectLMMSE = zeros(length(snrValues), 1);
csiLossAI = zeros(length(snrValues), 1);

% 每个SNR统计子帧的数量
numCountFrame = 200;                                        
csiPreTemp = zeros(3, numValidSubc, numSym, numTx, numRx);

for idx = 1:length(snrValues)
    snr = snrValues(idx);
    % 重置工具对象
    reset(mimoChannel)
    
    reset(errorRatePerfectLS);
    reset(errorRatePerfectMMSE);
    reset(errorRateLSZF);
    reset(errorRateLSMMSE);
    reset(errorRateMMSEZF);
    reset(errorRateMMSEMMSE);
    reset(errorRateAI);
    reset(errorRateEQ);
    
    csiEst_LOSS_MMSE = 0;  % 用于累计MMSE   的MSE LOSS
    csiEst_LOSS_LS = 0;    % 用于累计LS     的MSE LOSS
    csiEst_LOSS_LMMSE = 0; % 用于累计LMMSE  的MSE LOSS
    csiEst_LOSS_AI = 0;    % 用于累计AI     的MSE LOSS
    
    for frame = -1:numCountFrame
        % 数据符号生成
        txSymStream = randi([0 M-1], numFrameSymbols, 1); 
        dataSignal = pskmod(txSymStream, M);  % 调制后的符号为复数形式
        dataSignal = reshape(dataSignal, numDataSubc, numSym, numTx);
        pilotSignal = repmat(1+1i, numPilot, numSym, numTx);
        
        originSignal = zeros(numSubc, numSym, numTx);
        originSignal(dataIndices,:,:) = dataSignal;
        for tx = 1:numTx
            originSignal(pilotIndices(:,1,tx),:,tx) = pilotSignal(:, :, tx);
        end    

        % OFDM 调制
        txSignal = ofdmMod(dataSignal, pilotSignal);
        % 通过信道模型获取接收信号和路径增益[总样本数, N_path, numTransmitAntennas, numReceiveAntennas]
        [airSignal, pathGains] = mimoChannel(txSignal);
        mimoChannelInfo = info(mimoChannel);
        pathFilters = mimoChannelInfo.ChannelFilterCoefficients;
        toffset = mimoChannelInfo.ChannelFilterDelay;        
        % 去滤波器时延
        airSignal = [airSignal((toffset+1):end,:); zeros(toffset,2)];
        % 噪声
        [airSignal, noiseVar] = awgn(airSignal, snr, "measured");
        % OFDM 解调
        [rxDataSignal, rxPilotSignal] = ofdmDemod(airSignal);

        % 完美CSI矩阵Nsc x Nsym x Nt x Nr
        hAll = ofdmChannelResponse(pathGains, pathFilters, numSubc, cpLength, 1:numSubc, toffset);
        hValid = hAll(validSubcIndices,:,:,:);
        hPerfect = hAll(dataIndices,:,:,:);

        csiPreTemp(1,:,:,:,:,:) = csiPreTemp(2,:,:,:,:,:);
        csiPreTemp(2,:,:,:,:,:) = csiPreTemp(3,:,:,:,:,:);
        csiPreTemp(3,:,:,:,:,1) = real(hValid);
        csiPreTemp(3,:,:,:,:,2) = imag(hValid);

        if frame < 1
            continue;
        end

        % 接收信号采集
        finalSignal = zeros(numSubc, numSym, numRx);
        finalSignal(dataIndices,:,:) = rxDataSignal(:,:,:);
        for rx = 1:numRx
            for tx = 1:numTx
                finalSignal(pilotIndices(:,1,tx),:,rx) = rxPilotSignal(:,:,tx,rx);
            end
        end
        rx_pilot_signal = zeros(numSubc, numSym, numRx);
        for rx = 1:numRx
            for tx = 1:numTx
                rx_pilot_signal(pilotIndices(:,1,tx),:,rx) = rxPilotSignal(:,:,tx,rx);
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
        % csi_ai = csiInfer(csiModel,csi_ls(validSubcIndices,:,:,:), csiPreTemp(1:2,:,:,:,:));
        csi_ai = ceInfer(ceDnnModel, csi_ls(validSubcIndices,:,:,:));
        csi_ai = csi_ai(valid2DataIndices,:,:,:);
        csiEst_LOSS_AI = csiEst_LOSS_AI + mean(abs(hPerfect(:) - csi_ai(:)).^2);

        % AI信道估计 ZF均衡
        hReshaped = reshape(csi_ai,[],numTx,numRx);
        eqSignalAIZF = ofdmEqualize(rxDataSignal,hReshaped, noiseVar, Algorithm="mmse");
        eqSignalAIZF = reshape(eqSignalAIZF, [], 1);
        rxSymAI = pskdemod(eqSignalAIZF, M);
        serAI(idx, :) = errorRateAI(txSymStream, rxSymAI);
        
        %% AI联合信道估计与均衡
        % eqAISignal = ceeqInfer(ceeqModel, csi_ls(validSubcIndices,:,:,:), csiPreTemp(1:2,:,:,:,:), finalSignal(validSubcIndices,:,:));
        eqAISignal = eqInfer(eqDnnModel, hValid, finalSignal(validSubcIndices,:,:));
        eqAISignal = eqAISignal(valid2DataIndices,:,:);
        eqAISignal = reshape(eqAISignal, [], 1);
        rxSymAIEQ = pskdemod(eqAISignal, M);
        serAIEQ(idx,:) = errorRateEQ(txSymStream, rxSymAIEQ);

        %% 传统信道估计
        
        % LS信道估计
        CEC.algorithm = 'ls';
        hLS = channelEstimate(rxPilotSignal, pilotSignal, dataIndices, pilotIndices);
        csiEst_LOSS_LS = csiEst_LOSS_LS + mean(abs(hPerfect(:) - hLS(:)).^2);

        % MMSE信道估计
        CEC.algorithm = 'ls';
        hMMSE = channelEstimate(rxPilotSignal, pilotSignal, dataIndices, pilotIndices);
        csiEst_LOSS_MMSE = csiEst_LOSS_MMSE + mean(abs(hPerfect(:) - hMMSE(:)).^2);

        % LMMSE信道估计
        CEC.algorithm = 'ls';
        hLMMSE = channelEstimate(rxPilotSignal, pilotSignal, dataIndices, pilotIndices);
        csiEst_LOSS_LMMSE = csiEst_LOSS_LMMSE + mean(abs(hPerfect(:) - hLMMSE(:)).^2);
        
        %% 信道均衡

        % 完美信道 ZF均衡
        hReshaped = reshape(hPerfect,[],numTx,numRx);
        eqSignalPerfectZF = ofdmEqualize(rxDataSignal,hReshaped, noiseVar, Algorithm="zf");
        eqSignalPerfectZF = reshape(eqSignalPerfectZF, [], 1);

        % 完美信道 MMSE均衡
        hReshaped = reshape(hPerfect,[],numTx,numRx);
        eqSignalPerfectMMSE = ofdmEqualize(rxDataSignal,hReshaped, noiseVar, Algorithm="mmse");
        eqSignalPerfectMMSE = reshape(eqSignalPerfectMMSE, [], 1);  
        
        % LS信道 ZF均衡
        hReshaped = reshape(hLS,[],numTx,numRx);
        eqSignalLSZF = ofdmEqualize(rxDataSignal,hReshaped, noiseVar, Algorithm="zf");
        eqSignalLSZF = reshape(eqSignalLSZF, [], 1);  
        % LS信道 MMSE均衡
        hReshaped = reshape(hLS,[],numTx,numRx);
        eqSignalLSMMSE = ofdmEqualize(rxDataSignal,hReshaped, noiseVar, Algorithm="mmse");
        eqSignalLSMMSE = reshape(eqSignalLSMMSE, [], 1);  
        % MMSE信道 ZF均衡
        hReshaped = reshape(hMMSE,[],numTx,numRx);
        eqSignalMMSEZF = ofdmEqualize(rxDataSignal,hReshaped, noiseVar, Algorithm="zf");
        eqSignalMMSEZF = reshape(eqSignalMMSEZF, [], 1);  
        % MMSE信道 MMSE均衡
        hReshaped = reshape(hMMSE,[],numTx,numRx);
        eqSignalMMSEMMSE = ofdmEqualize(rxDataSignal,hReshaped, noiseVar, Algorithm="mmse");
        eqSignalMMSEMMSE = reshape(eqSignalMMSEMMSE, [], 1);  

        % 误符号率计算
        rxSymPerfectLS = pskdemod(eqSignalPerfectZF, M);
        serPerfectZF(idx, :) = errorRatePerfectLS(txSymStream, rxSymPerfectLS);
    
        rxSymPerfectMMSE = pskdemod(eqSignalPerfectMMSE, M);
        serPerfectMMSE(idx, :) = errorRatePerfectMMSE(txSymStream, rxSymPerfectMMSE);
    
        rxSymLSZF = pskdemod(eqSignalLSZF, M);
        serLSZF(idx, :) = errorRateLSZF(txSymStream, rxSymLSZF);
    
        rxSymLSMMSE = pskdemod(eqSignalLSMMSE, M);
        serLSMMSE(idx, :) = errorRateLSMMSE(txSymStream, rxSymLSMMSE);
    
        rxSymMMSEZF = pskdemod(eqSignalMMSEZF, M);
        serMMSEZF(idx, :) = errorRateMMSEZF(txSymStream, rxSymMMSEZF);
    
        rxSymMMSEMMSE = pskdemod(eqSignalMMSEMMSE, M);
        serMMSEMMSE(idx, :) = errorRateMMSEMMSE(txSymStream, rxSymMMSEMMSE);

    end
    % 对每种算法在当前 SNR 的所有帧计算平均 MSE
    csiLossPerfectLS(idx) = csiEst_LOSS_LS / numCountFrame;
    csiLossPerfectMMSE(idx) = csiEst_LOSS_MMSE / numCountFrame;
    csiLossPerfectLMMSE(idx) = csiEst_LOSS_LMMSE / numCountFrame;
    csiLossAI(idx) = csiEst_LOSS_AI / numCountFrame;
end

% % 测试数据集保存
% save('../raw/compareData.mat',...
%     'rxSignalData',...
%     'txSignalData',...
%     'csiData',...
%     '-v7.3')
% 
% save('./metric/CE_SER_METRIC.mat',...
%     'noiseVarlData',...
%     'txSymStreamData',...
%     'errorPerfectZF', ...
%     'errorPerfectMMSE', ...
%     'errorLSZF', ...
%     'errorLSMMSE', ...
%     'errorMMSEZF', ...
%     'errorMMSEMMSE',...
%     'msePerfectLS', ...
%     'msePerfectMMSE', ...
%     'msePerfectLMMSE',...
%     '-v7.3')
%%

disp(snrValues)
disp(serPerfectZF(:, 1))
disp(serLSZF(:, 1))
disp(serAIEQ(:, 1))


%% 图形1：SER误码率图像

figure;
hold on;
% 绘制每种算法的误符号率曲线
plot(snrValues, serPerfectZF(:, 1), '-o', 'LineWidth', 1.5, 'DisplayName', 'Perfect ZF');
plot(snrValues, serPerfectMMSE(:, 1), '-s', 'LineWidth', 1.5, 'DisplayName', 'Perfect MMSE');
plot(snrValues, serLSZF(:, 1), '-d', 'LineWidth', 1.5, 'DisplayName', 'LS ZF');
plot(snrValues, serLSMMSE(:, 1), '-^', 'LineWidth', 1.5, 'DisplayName', 'LS MMSE');
% plot(snrValues, errorMMSEZF(:, 1), '-v', 'LineWidth', 1.5, 'DisplayName', 'MMSE ZF');
% plot(snrValues, errorMMSEMMSE(:, 1), '-p', 'LineWidth', 1.5, 'DisplayName', 'MMSE MMSE');
plot(snrValues, serAI(:, 1), '-p', 'LineWidth', 1.5, 'DisplayName', 'AI ZF');
plot(snrValues, serAIEQ(:, 1), '-v', 'LineWidth', 1.5, 'DisplayName', 'AI EQ');


% 设置图形属性
grid on;
xlabel('SNR (dB)');
ylabel('Symbol Error Rate (SER)');
title('SER vs. SNR for Different Channel Estimation and Equalization Algorithms');
legend('Location', 'best');
set(gca, 'YScale', 'log');  % 将 Y 轴设置为对数坐标

% 显示图形
hold off;

% %% 图形1：信道估计 MSE LOSS图像
figure;
hold on;

% 绘制每种算法的信道估计MSELOSS图像
plot(snrValues, csiLossPerfectLS, '-o', 'LineWidth', 1.5, 'DisplayName', 'LS');
% plot(snrValues, msePerfectMMSE, '-s', 'LineWidth', 1.5, 'DisplayName', 'MMSE');
% plot(snrValues, msePerfectLMMSE, '-^', 'LineWidth', 1.5, 'DisplayName', 'LMMSE');
plot(snrValues, csiLossAI, '-o', 'LineWidth', 1.5, 'DisplayName', 'AI');


% 设置图形属性
grid on;
xlabel('SNR (dB)');
ylabel('MSE with h_{Perfect}');
title('Channel Estimation MSE vs. SNR');
legend('Location', 'best');
hold off;

%% 自定义函数
function [H_est] = channelEstimate(rxPilotSignal, refPilotSignal, dataIndices, pilotIndices)
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
    H_est = zeros(numDataSubc, numSym, numTx, numRx);

    for tx = 1:numTx
        for rx = 1:numRx
            for sym=1:numSym
                % 获取当前发射 - 接收天线对的导频信号
                pilotRxSignal = rxPilotSignal(:,sym,tx,rx);
                pilotRefSignal = refPilotSignal(:, sym, tx);
                H_ls = pilotRxSignal ./ pilotRefSignal; 
                H_est(:, sym, tx, rx)  = interp1(pilotIndices(:,sym,tx), H_ls, dataIndices, 'linear', 'extrap');
            end
        end
    end
end

function H_avg = pilotAveraging(H_ls, freqWindow, timeWindow)
    [numPilotSubc, numSym] = size(H_ls);
    H_avg = zeros(size(H_ls));

    halfFreqWindow = floor(freqWindow / 2);
    halfTimeWindow = floor(timeWindow / 2);

    for sym = 1:numSym
        for subc = 1:numPilotSubc
            startSubc = max(1, subc - halfFreqWindow);
            endSubc = min(numPilotSubc, subc + halfFreqWindow);
            startSym = max(1, sym - halfTimeWindow);
            endSym = min(numSym, sym + halfTimeWindow);

            window = H_ls(startSubc:endSubc, startSym:endSym);
            H_avg(subc, sym) = mean(window(:));
        end
    end
end

function [out, csi] = myZFequalize(H, rxsignal)
% myZFequalize 自定义零迫（ZF）均衡函数
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
% 输出：
%   out: 均衡后估计的发送符号，尺寸为 [nsc, nsym, ntx]
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

% 初始化输出变量
out = zeros(nsc, nsym, ntx);
csi = zeros(nsc, nsym, ntx);

% 对每个子载波和每个 OFDM 符号进行循环
for i = 1:nsc
    for j = 1:nsym
        % 提取当前子载波和符号的信道矩阵
        % H(i,j,:,:) 的尺寸为 [ntx, nrx]
        % 为了使其符合 y = H_sub * x 模型，将其转置为 [nrx, ntx]
        H_sub = squeeze(H(i, j, :, :)).';  % 结果尺寸为 [nrx, ntx]
        
        % 提取当前子载波和符号的接收信号
        % rxsignal(i,j,:) 的尺寸为 [nrx, 1]
        y = squeeze(rxsignal(i, j, :));
        
        % 计算零迫均衡
        % 若 H_sub 为满列秩，则 pinv(H_sub) = inv(H_sub' * H_sub) * H_sub'
        x_hat = pinv(H_sub) * y;
        out(i, j, :) = x_hat;
        
        % 计算 CSI 信息：
        % 这里简单计算 H_sub^H * H_sub，然后取其对角线（每个发射分量的能量）
        % 并取倒数，作为信道质量指标。注意防止除零错误。
        Hhh = H_sub' * H_sub;    % 尺寸为 [ntx, ntx]
        diag_val = diag(Hhh);     % 取对角线，得到 [ntx, 1] 向量
        % 防止除数为 0
        diag_val(diag_val < eps) = eps;
        csi(i, j, :) = 1 ./ diag_val;
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
