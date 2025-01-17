clear;
clc;
%%
miPyPath = 'C:\Users\stone\AppData\Local\Programs\Python\Python312\python.exe';
lenPyPath = 'D:\Python\python.exe';
pyenv('Version', lenPyPath)
model = py.csiFormer.load_model();

function [equalizdSignal] = equalizerInfer(model, tx_pilot, rx_pilot, pre_csi, rx_signal)
    tx_pilot = py.numpy.array(cat(ndims(tx_pilot)+1, real(tx_pilot), imag(tx_pilot)));
    rx_pilot = py.numpy.array(cat(ndims(rx_pilot)+1, real(rx_pilot), imag(rx_pilot)));
    pre_csi = py.numpy.array(cat(ndims(pre_csi)+1, real(pre_csi), imag(pre_csi)));
    rx_signal = py.numpy.array(cat(ndims(rx_signal)+1, real(rx_signal), imag(rx_signal)));
    
    equalizdData = py.infer.infer3(model, tx_pilot, rx_pilot, pre_csi, rx_signal);
    % 转换 Python numpy 输出为 MATLAB 矩阵
    equalizdSignal = double(py.array.array('d', py.numpy.nditer(equalizdData)));
    equalizdSignal = reshape(equalizdSignal, 52,14,2,2);
    equalizdSignal = complex(equalizdSignal(:,:,:,1), equalizdSignal(:,:,:,2));
end

function [csi_est] = csiInfer(model, csi_ls, pre_csi)
    csi_ls = py.numpy.array(cat(ndims(csi_ls)+1, real(csi_ls), imag(csi_ls)));
    pre_csi = py.numpy.array(cat(ndims(pre_csi)+1, real(pre_csi), imag(pre_csi)));
    
    csi_est = py.csiFormer.infer(model, csi_ls, pre_csi);
    % 转换 Python numpy 输出为 MATLAB 矩阵
    csi_est = double(py.array.array('d', py.numpy.nditer(csi_est)));
    csi_est = reshape(csi_est, 52,14,2,2,2);
    csi_est = complex(csi_est(:,:,:,:,1), csi_est(:,:,:,:,2));
end


%% 参数设置
% 系统参数配置
numSubFrame = 10;                                         % 子帧数量
snrValues = 0:6:24;                                       % 信噪比范围
numSubc = 64;                                             % FFT 长度
numGuardBands = [6;6];                                    % 左右保护带
numPilot = 4;                                             % 每根天线的导频子载波
numTx = 2;                                                % 发射天线数量
numRx = 2;                                                % 接收天线数量
numSym = 14;                                              % 每帧 OFDM 符号数
numStream = 2;                                            % 数据流个数
cpLength = 16;                                            % 循环前缀长度

% 调制参数配置
M = 2;                                                    % QPSK 调制（M=4）


% 信道模型配置
sampleRate = 15.36e6;                                     % 采样率
pathDelays = [0 0.5e-6];                                  % 路径时延
averagePathGains = [0 -2];                                % 平均路径增益
maxDopplerShift = 200;                                    % 最大多普勒频移

% 信道估计配置
CEC.pilotAverage = 'UserDefined';
CEC.freqWindow = 3;
CEC.timeWindow = 3;
CEC.interpType = 'linear';
CEC.algorithm = 'ls';
CEstimateAlgs = ['ls', 'mmse', 'lmmse'];

%信道均衡配置
CEqualizeAlgs = ['zf', 'mmse'];

% 信号导频分布配置
validSubcIndices = (numGuardBands(1)+1):(numSubc-numGuardBands(2));
numValidSubc = length(validSubcIndices);

% 导频子载波配置
pilotIndicesAnt1 = [7; 26; 40; 57]; % 天线 1 导频索引
pilotIndicesAnt2 = [8; 27; 41; 58]; % 天线 2 导频索引
% 构造 PilotCarrierIndices (3D 矩阵, NPilot-by-NSym-by-NT)
pilotIndices = zeros(numPilot, numSym, numTx);
pilotIndices(:, :, 1) = repmat(pilotIndicesAnt1, 1, numSym); % 天线 1
pilotIndices(:, :, 2) = repmat(pilotIndicesAnt2, 1, numSym); % 天线 2
pilotIndiceMask = unique([pilotIndicesAnt1,pilotIndicesAnt2]);

% 数据子载波配置
dataIndices = setdiff((numGuardBands(1)+1):(numSubc-numGuardBands(2)),unique(pilotIndices));
numDataSubc = length(dataIndices);
numSubFrameSym = numDataSubc * numSym * numTx;

dataIndiceMask = dataIndices - numGuardBands(1);

% OFDM调制器
ofdmMod = comm.OFDMModulator('FFTLength', numSubc, ...
                             'NumGuardBandCarriers', numGuardBands, ...
                             'NumSymbols', numSym, ...
                             'PilotInputPort', true, ...
                             'PilotCarrierIndices', pilotIndices, ...
                             'CyclicPrefixLength', cpLength, ...
                             'NumTransmitAntennas', numTx);
% OFDM解调器
ofdmDemod = comm.OFDMDemodulator('FFTLength', numSubc, ...
                                  'NumGuardBandCarriers', numGuardBands, ...
                                  'NumSymbols', numSym, ...
                                  'PilotOutputPort', true, ...
                                  'PilotCarrierIndices', pilotIndices, ...
                                  'CyclicPrefixLength', cpLength, ...
                                  'NumReceiveAntennas', numRx);
% 信道模型
% mimoChannel = comm.MIMOChannel(...
%     'SampleRate', sampleRate, ...
%     'SpatialCorrelationSpecification', 'None',...
%     'PathDelays', pathDelays, ...
%     'AveragePathGains', averagePathGains, ...
%     'MaximumDopplerShift', maxDopplerShift, ...
%     'NumTransmitAntennas', numTx, ...
%     'NumReceiveAntennas', numRx, ...
%     'FadingDistribution', 'Rayleigh', ...
%     'RandomStream', 'mt19937ar with seed', ...
%     'Seed', 123, ... % 固定随机种子
%     'PathGainsOutputPort', true);   % 开启路径增益输出

% 简单信道
mimoChannel = comm.MIMOChannel(...
    'SampleRate', sampleRate, ...
    'SpatialCorrelationSpecification', 'None',...
    'NumTransmitAntennas', numTx, ...
    'NumReceiveAntennas', numRx, ...
    'FadingDistribution', 'Rayleigh', ...
    'RandomStream', 'mt19937ar with seed', ...
    'Seed', 123, ... % 固定随机种子
    'PathGainsOutputPort', true);   % 开启路径增益输出

% 评价体系
% errorRate对象创建
errorRatePerfectLS = comm.ErrorRate;
errorRatePerfectMMSE = comm.ErrorRate;
errorRateLSZF = comm.ErrorRate;
errorRateLSMMSE = comm.ErrorRate;
errorRateMMSEZF = comm.ErrorRate;
errorRateMMSEMMSE = comm.ErrorRate;


err = comm.ErrorRate;
% SER 数据对比
errorPerfectZF = zeros(length(snrValues), 3);
errorPerfectMMSE = zeros(length(snrValues), 3);
errorLSZF = zeros(length(snrValues), 3);
errorLSMMSE = zeros(length(snrValues), 3);
errorMMSEZF = zeros(length(snrValues), 3);
errorMMSEMMSE = zeros(length(snrValues), 3);

% 信道估计MSE LOSS 数据对比
msePerfectLS = zeros(length(snrValues), 1);
msePerfectMMSE = zeros(length(snrValues), 1);
msePerfectLMMSE = zeros(length(snrValues), 1);

noiseVarlData = zeros(length(snrValues));

txSignalData = zeros(length(snrValues), numSubFrame, numValidSubc, numSym, numTx, 2);
rxSignalData = zeros(length(snrValues), numSubFrame, numValidSubc, numSym, numRx, 2);
csiData = zeros(length(snrValues), numSubFrame, numValidSubc, numSym, numTx, numRx, 2);
txSymStreamData = zeros(length(snrValues), numSubFrame, numSubFrameSym,1);

pre_csi = zeros(2, numValidSubc, numSym, numTx, numRx);

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
    
    snrEstimateMSE_MMSE = 0; % 用于累计MMSE的MSE
    snrEstimateMSE_LS = 0;   % 用于累计LS的MSE
    snrEstimateMSE_LMMSE = 0; % 用于累计LMMSE的MSE
    
    for frame = 1:numSubFrame
        % 数据符号生成&调制
        txSymStream = randi([0 M-1], numSubFrameSym, 1); 
        txSymStreamData(idx, frame,:,:) = txSymStream;
        dataSignal = pskmod(txSymStream, M);  
        dataSignal = reshape(dataSignal, numDataSubc, numSym, numTx);
        % 导频符号生成
        pilotQPSKSymbols = [1+1i, 1+1i, 1+1i, 1+1i];
        pilotSignal = pilotQPSKSymbols(randi(length(pilotQPSKSymbols), numPilot, numSym, numTx));
        % 发射信号采集
        originSignal = zeros(numSubc, numSym, numTx);
        originSignal(dataIndices, :, :) = dataSignal;
        for tx = 1:numTx
            for sym = 1:numSym
                originSignal(pilotIndices(:,sym,tx),sym,tx) = pilotSignal(:, sym, tx);
            end
        end    
                % 发射信号采集
        arg3 = zeros(numSubc, numSym, numTx);
        arg3(dataIndices, :, :) = dataSignal;
        arg = arg3(validSubcIndices,:,:);
        % 发射信号采集
        ttxSignal = zeros(numSubc, numSym, numTx);
        for tx = 1:numTx
            for sym = 1:numSym
                arg4(pilotIndices(:,sym,tx),sym,tx) = pilotSignal(:, sym, tx);
            end
        end  
        ttxPilot = arg4(validSubcIndices,:,:);
        % OFDM调制
        txSignal = ofdmMod(dataSignal, pilotSignal); % 结果为 (80 × 14 × 2)，包含循环前缀的时域信号
        % 信道模型：传输信号&路径增益
        [transmitSignal, pathGains] = mimoChannel(txSignal); % pathGains: [总样本数, N_path, numTx, numRx]
        % 噪声模型：传输信号&噪声功率
        [transmitSignal, noiseVar] = awgn(transmitSignal, snr, "measured");
        % OFDM解调: 接收数据信号&接收导频信号 NPilot-by-NSym-by-NT-by-NR
        [rxDataSignal, rxPilotSignal] = ofdmDemod(transmitSignal);

        % 完美CSI
        mimoChannelInfo = info(mimoChannel);
        pathFilters = mimoChannelInfo.ChannelFilterCoefficients;
        toffset = mimoChannelInfo.ChannelFilterDelay;
        h = ofdmChannelResponse(pathGains, pathFilters, numSubc, cpLength, validSubcIndices, toffset); % Nsc x Nsym x Nt x Nr
        hPerfect = h(dataIndiceMask,:,:,:);
        
        pre_csi(1,:,:,:,:) = pre_csi(2,:,:,:,:);
        pre_csi(2,:,:,:,:) = h;

        if frame < 3

            continue;
        end

        % 接收信号采集
        rxSignal = zeros(numSubc, numSym, numRx);
        rxSignal(dataIndices,:,:) = rxDataSignal(:,:,:);
        for rx = 1:numRx
            for tx = 1:numTx
                for sym = 1:numSym
                rxSignal(pilotIndices(:,sym,tx),sym,rx) = rxPilotSignal(:,sym,tx,rx);
                end
            end
        end 
        
        arg1 = zeros(numSubc, numSym, numRx);
        arg1(dataIndices,:,:) = rxDataSignal(:,:,:);
        rrxSignal = arg1(validSubcIndices,:,:);

        arg2 = zeros(numSubc, numSym, numRx);
        for rx = 1:numRx
            for tx = 1:numTx
                for sym = 1:numSym
                arg2(pilotIndices(:,sym,tx),sym,rx) = rxPilotSignal(:,sym,tx,rx);
                end
            end
        end 
        rrxPilot = arg2(validSubcIndices,:,:);

        equalizedData = equalizerInfer(model,ttxPilot, rrxPilot, pre_csi, rrxSignal);
        equalizedData = equalizedData(dataIndiceMask,:,:);
        eqSignalll = reshape(equalizedData, [], 1);  
        % 误符号率计算
        rxSymPerfectLS = pskdemod(eqSignalll, M);
        err = comm.ErrorRate;
        disp(err(txSymStream, rxSymPerfectLS))

        % 测试数据集采集
        noiseVarlData(idx, frame) = noiseVar;
        txSignalData(idx, frame,:,:,:,1) = real(originSignal(validSubcIndices,:,:));
        txSignalData(idx, frame,:,:,:,2) = imag(originSignal(validSubcIndices,:,:));
        rxSignalData(idx, frame,:,:,:,1) = real(rxSignal(validSubcIndices,:,:));
        rxSignalData(idx, frame,:,:,:,2) = imag(rxSignal(validSubcIndices,:,:));
        csiData(idx, frame, :,:,:,:,1) = real(h);
        csiData(idx, frame, :,:,:,:,2) = imag(h);

        %% 信道估计
        
        % LS信道估计
        CEC.algorithm = 'ls';
        hLS = channelEstimate(rxPilotSignal, pilotSignal, dataIndices, pilotIndices, CEC);
        snrEstimateMSE_LS = snrEstimateMSE_LS + mean(abs(hPerfect(:) - hLS(:)).^2);

        % MMSE信道估计
        CEC.algorithm = 'ls';
        hMMSE = channelEstimate(rxPilotSignal, pilotSignal, dataIndices, pilotIndices, CEC);
        msePerfectMMSE(idx) = mean(abs(hPerfect(:) - hMMSE(:)).^2);
        snrEstimateMSE_MMSE = snrEstimateMSE_MMSE + mean(abs(hPerfect(:) - hMMSE(:)).^2);

        % LMMSE信道估计
        CEC.algorithm = 'ls';
        hLMMSE = channelEstimate(rxPilotSignal, pilotSignal, dataIndices, pilotIndices, CEC);
        msePerfectLMMSE(idx) = mean(abs(hPerfect(:) - hLMMSE(:)).^2);
        snrEstimateMSE_LMMSE = snrEstimateMSE_LMMSE + mean(abs(hPerfect(:) - hLMMSE(:)).^2);

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
        errorPerfectZF(idx, :) = errorRatePerfectLS(txSymStream, rxSymPerfectLS);
    
        rxSymPerfectMMSE = pskdemod(eqSignalPerfectMMSE, M);
        errorPerfectMMSE(idx, :) = errorRatePerfectMMSE(txSymStream, rxSymPerfectMMSE);
    
        rxSymLSZF = pskdemod(eqSignalLSZF, M);
        errorLSZF(idx, :) = errorRateLSZF(txSymStream, rxSymLSZF);
    
        rxSymLSMMSE = pskdemod(eqSignalLSMMSE, M);
        errorLSMMSE(idx, :) = errorRateLSMMSE(txSymStream, rxSymLSMMSE);
    
        rxSymMMSEZF = pskdemod(eqSignalMMSEZF, M);
        errorMMSEZF(idx, :) = errorRateMMSEZF(txSymStream, rxSymMMSEZF);
    
        rxSymMMSEMMSE = pskdemod(eqSignalMMSEMMSE, M);
        errorMMSEMMSE(idx, :) = errorRateMMSEMMSE(txSymStream, rxSymMMSEMMSE);

    end
    % 对每种算法在当前 SNR 的所有帧计算平均 MSE
    msePerfectLS(idx) = snrEstimateMSE_LS / numSubFrame;
    msePerfectMMSE(idx) = snrEstimateMSE_MMSE / numSubFrame;
    msePerfectLMMSE(idx) = snrEstimateMSE_LMMSE / numSubFrame;
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

%% 图形1：SER误码率图像

figure;
hold on;
% 绘制每种算法的误符号率曲线
plot(snrValues, errorPerfectZF(:, 1), '-o', 'LineWidth', 1.5, 'DisplayName', 'Perfect ZF');
% plot(snrValues, errorPerfectMMSE(:, 1), '-s', 'LineWidth', 1.5, 'DisplayName', 'Perfect MMSE');
plot(snrValues, errorLSZF(:, 1), '-d', 'LineWidth', 1.5, 'DisplayName', 'LS ZF');
% plot(snrValues, errorLSMMSE(:, 1), '-^', 'LineWidth', 1.5, 'DisplayName', 'LS MMSE');
% plot(snrValues, errorMMSEZF(:, 1), '-v', 'LineWidth', 1.5, 'DisplayName', 'MMSE ZF');
% plot(snrValues, errorMMSEMMSE(:, 1), '-p', 'LineWidth', 1.5, 'DisplayName', 'MMSE MMSE');
% plot(snrValues, errorAIMMSE(:, 1), '-p', 'LineWidth', 1.5, 'DisplayName', 'AIMMSE');
% plot(snrValues, errorAIPROMMSE(:, 1), '-p', 'LineWidth', 1.5, 'DisplayName', 'AIPROMMSE');
% plot(snrValues, errorAIPROMAX(:, 1), '-p', 'LineWidth', 1.5, 'DisplayName', 'AIPROMAX');
% 设置图形属性
grid on;
xlabel('SNR (dB)');
ylabel('Symbol Error Rate (SER)');
title('SER vs. SNR for Different Channel Estimation and Equalization Algorithms');
legend('Location', 'best');
set(gca, 'YScale', 'log');  % 将 Y 轴设置为对数坐标

% 显示图形
hold off;

% % %% 图形1：信道估计 MSE LOSS图像
% figure;
% hold on;
% 
% % 绘制每种算法的信道估计MSELOSS图像
% plot(snrValues, msePerfectLS, '-o', 'LineWidth', 1.5, 'DisplayName', 'LS');
% % plot(snrValues, msePerfectMMSE, '-s', 'LineWidth', 1.5, 'DisplayName', 'MMSE');
% % plot(snrValues, msePerfectLMMSE, '-^', 'LineWidth', 1.5, 'DisplayName', 'LMMSE');
% % plot(snrValues, mseAI, '-o', 'LineWidth', 1.5, 'DisplayName', 'mseAI');
% % plot(snrValues, mseAIPRO, '-o', 'LineWidth', 1.5, 'DisplayName', 'mseAIPRO');
% 
% 
% % 设置图形属性
% grid on;
% xlabel('SNR (dB)');
% ylabel('MSE with h_{Perfect}');
% title('Channel Estimation MSE vs. SNR');
% legend('Location', 'best');
% hold off;

%% 自定义函数
function [H_est, noiseVar] = channelEstimate(rxPilotSignal,refPilotSignal, dataIndices, pilotIndices, CEC)
    % MIMO-OFDM 信道估计函数
    % 输入：
    %   rxDataSignal: 接收信号矩阵 (numDataSubc x numSym x numRx)
    %   rxPilotSignal: 接收导频信号(numPilotSubc x numSym x numTx x numRx)
    %   refPilotSignal: 参考导频信号 (numPilotSubc x numSym x numTx)
    %   dataIndices: 数据符号索引
    %   pilotIndices: 导频符号索引(numPilotSubc x numSym x numTx)
    %   CEC.algorithm: 估计算法 ('LS' 或 'MMSE')
    %   CEC.interpType: 插值类型 ('nearest', 'linear', 'cubic', 'spline')
    %   CEC.freqWindow: 频域平均窗口大小;
    %   CEC.timeWindow: 时域平均窗口大小;T
    %   
    % 输出：
    %   H_est: 估计的信道响应矩阵 (numSubc x numSym x numTx x numRx)
    %   noiseVar: 噪声平均功率

    % 提取信号维度
    numDataSubc = length(dataIndices);
    [~, numSym, numTx, numRx] = size(rxPilotSignal);

    % 初始化估计的信道响应矩阵和噪声功率
    H_est = zeros(numDataSubc, numSym, numTx, numRx);
    noiseVar = zeros(numTx, numRx);

    % 验证算法参数
    validAlgorithms = {'LS', 'MMSE'};
    if ~any(strcmpi(CEC.algorithm, validAlgorithms))
        error('Invalid algorithm. Must be one of: LS, MMSE');
    end

    % 验证插值类型
    validInterpTypes = {'nearest', 'linear', 'cubic','spline'};
    if ~any(strcmpi(CEC.interpType, validInterpTypes))
        error('Invalid interpolation type. Must be one of: nearest, linear, cubic, spline');
    end

    for tx = 1:numTx
        for rx = 1:numRx
            % 获取当前发射 - 接收天线对的导频信号
            pilotRxSignal = rxPilotSignal(:,:,tx,rx);
            pilotRefSignal = refPilotSignal(:, :, tx);

            % 根据算法进行信道估计
            if strcmpi(CEC.algorithm, 'LS')
                % 最小二乘估计
                H_ls = pilotRxSignal./ pilotRefSignal;
            elseif strcmpi(CEC.algorithm, 'MMSE')
                error('not today');
                % % 简单假设信道自相关矩阵为单位矩阵
                % R_hh = eye(size(pilotRefSignal, 1) * size(pilotRefSignal, 2));
                % % 噪声功率谱密度暂时假设为1，实际需要估计
                % noisePower = 1;
                % R_nn = noisePower * eye(size(pilotRefSignal, 1) * size(pilotRefSignal, 2));
                % R_yh = R_hh;
                % mmseDenominator = R_yh' * inv(R_yh * R_yh' + R_nn) * R_yh;
                % H_ls = mmseDenominator \ (pilotRxSignal(:));
                % H_ls = reshape(H_ls, size(pilotRxSignal));
            end
            
            % 导频平均
            H_avg = pilotAveraging(H_ls, CEC.freqWindow, CEC.timeWindow);
            H_avg = H_ls;
            % 信道插值
            [X, Y] = meshgrid(1:numSym, dataIndices);
            [Xp, Yp] = meshgrid(1:numSym, squeeze(pilotIndices(:,1,tx)));
            H_est(:, :, tx, rx)  = griddata(Xp, Yp, H_avg, X, Y);

            % % 噪声估计
            % dataRxSignal = rxSignal(dataIndices(:, :, tx), :, rx);
            % dataRefSignal = refPilotSignal(:, :, tx);
            % estimatedSignal = dataRefSignal.* H_est(dataIndices(:, :, tx), tx, rx);
            % noise = dataRxSignal - estimatedSignal;
            % noiseVar(tx, rx) = mean(abs(noise(:)).^2);
        end
    end

    % 计算平均噪声功率
    noiseVar = mean(noiseVar(:));
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
