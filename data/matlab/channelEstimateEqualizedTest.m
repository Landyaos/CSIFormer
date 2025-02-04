clear;
clc;
%%
miPyPath = 'C:\Users\stone\AppData\Local\Programs\Python\Python312\python.exe';
lenPyPath = 'D:\Python\python.exe';
pyenv('Version', lenPyPath)
csiModel = py.csiFormer.load_model();
ceeqModel = py.ceeqFormer.load_model();

function [csi_est] = csiInfer(model, csi_ls, pre_csi)
    csi_ls = py.numpy.array(cat(ndims(csi_ls)+1, real(csi_ls), imag(csi_ls)));
    pre_csi = py.numpy.array(cat(ndims(pre_csi)+1, real(pre_csi), imag(pre_csi)));
    csi_est = py.csiFormer.infer(model, csi_ls, pre_csi);
    % 转换 Python numpy 输出为 MATLAB 矩阵
    csi_est = double(py.array.array('d', py.numpy.nditer(csi_est)));
    csi_est = reshape(csi_est, 52,14,2,2,2);
    csi_est = complex(csi_est(:,:,:,:,1), csi_est(:,:,:,:,2));
end

function [equalized_signal] = ceeqInfer(model, csi_ls, pre_csi, rx_signal)
    csi_ls = py.numpy.array(cat(ndims(csi_ls)+1, real(csi_ls), imag(csi_ls)));
    pre_csi = py.numpy.array(cat(ndims(pre_csi)+1, real(pre_csi), imag(pre_csi)));
    rx_signal = py.numpy.array(cat(ndims(rx_signal)+1, real(rx_signal), imag(rx_signal)));
    
    equalized_signal = py.ceeqFormer.infer(model, csi_ls, pre_csi, rx_signal);

    % 转换 Python numpy 输出为 MATLAB 矩阵
    equalized_signal = double(py.array.array('d', py.numpy.nditer(equalized_signal)));
    equalized_signal = reshape(equalized_signal, 52,14,2,2);
    equalized_signal = complex(equalized_signal(:,:,:,1), equalized_signal(:,:,:,2));
end


%% 参数设置
% 系统参数配置
snrValues = 0:4:20;                                      % 信噪比范围
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
% pathDelays = [0 50 120 200 230 500 1600 2300 5000 7000] * 1e-9; 
% averagePathGains = [-1.0 -1.0 -1.0 -1.0 -1.0 -1.5 -1.5 -1.5 -3.0 -5.0]; 
maxDopplerShift = 250;                                    % 最大多普勒频移

% 信道估计配置
CEC.pilotAverage = 'UserDefined';
CEC.freqWindow = 3;
CEC.timeWindow = 3;
CEC.interpType = 'linear';
CEC.algorithm = 'ls';

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
% 导频集合
pilotQPSKSymbols = [1+1i, 1+1i, 1+1i, 1+1i];

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
mimoChannel = comm.MIMOChannel(...
    'SampleRate', sampleRate, ...
    'SpatialCorrelationSpecification', 'None',...
    'PathDelays', pathDelays, ...
    'AveragePathGains', averagePathGains, ...
    'MaximumDopplerShift', maxDopplerShift, ...
    'NumTransmitAntennas', numTx, ...
    'NumReceiveAntennas', numRx, ...
    'FadingDistribution', 'Rayleigh', ...
    'RandomStream', 'mt19937ar with seed', ...
    'Seed', 123, ... % 固定随机种子
    'PathGainsOutputPort', true);   % 开启路径增益输出

% % % 简单信道
% mimoChannel = comm.MIMOChannel(...
%     'SampleRate', sampleRate, ...
%     'SpatialCorrelationSpecification', 'None',...
%     'NumTransmitAntennas', numTx, ...
%     'NumReceiveAntennas', numRx, ...
%     'FadingDistribution', 'Rayleigh', ...
%     'RandomStream', 'mt19937ar with seed', ...
%     'Seed', 123, ... % 固定随机种子
%     'PathGainsOutputPort', true);   % 开启路径增益输出

% 评价体系
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
numCountFrame = 10;                                        
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
    csiEst_LOSS_AI = 0;    % 用于累计AI  的MSE LOSS
    
    for frame = -1:numCountFrame
        % 数据符号生成&调制
        txSymStream = randi([0 M-1], numSubFrameSym, 1); 
        dataSignal = pskmod(txSymStream, M);  
        dataSignal = reshape(dataSignal, numDataSubc, numSym, numTx);

        pilotSignal = pilotQPSKSymbols(randi(length(pilotQPSKSymbols), numPilot, numSym, numTx));
        % 发射信号采集
        originSignal = zeros(numSubc, numSym, numTx);
        originSignal(dataIndices, :, :) = dataSignal;
        for tx = 1:numTx
            originSignal(pilotIndices(:,1,tx),:,tx) = pilotSignal(:, :, tx);

        end    

        % OFDM 调制
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
        hValidSubc = ofdmChannelResponse(pathGains, pathFilters, numSubc, cpLength, validSubcIndices, toffset); % Nsc x Nsym x Nt x Nr
        hPerfect = hValidSubc(dataIndiceMask,:,:,:);
        
        csiPreTemp(1,:,:,:,:) = csiPreTemp(2,:,:,:,:);
        csiPreTemp(2,:,:,:,:) = csiPreTemp(3,:,:,:,:);
        csiPreTemp(3,:,:,:,:) = hValidSubc;

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

        %% AI信道估计与均衡
        % 计算导频处信道估计
        csi_ls = zeros(numSubc, numSym, numTx, numRx);
        for tx = 1:numTx
            for rx = 1:numRx
                csi_ls(pilotIndices(:,1,tx),:,tx,rx) = rxPilotSignal(:,:,tx,rx) ./ pilotSignal(:, :, tx);
            end
        end
        csi_ls = csi_ls(validSubcIndices,:,:,:);
        csi_ai = csiInfer(csiModel,csi_ls, csiPreTemp(1:2,:,:,:,:));
        csi_ai = csi_ai(dataIndiceMask,:,:,:);
        csiEst_LOSS_AI = csiEst_LOSS_AI + mean(abs(hPerfect(:) - csi_ai(:)).^2);
        % AI信道估计 ZF均衡
        hReshaped = reshape(csi_ai,[],numTx,numRx);
        eqSignalAIZF = ofdmEqualize(rxDataSignal,hReshaped, noiseVar, Algorithm="mmse");
        eqSignalAIZF = reshape(eqSignalAIZF, [], 1);
        rxSymAI = pskdemod(eqSignalAIZF, M);
        serAI(idx, :) = errorRateAI(txSymStream, rxSymAI);
        
        %% AI联合信道估计与均衡
        eqAISignal = ceeqInfer(ceeqModel, csi_ls, csiPreTemp(1:2,:,:,:,:), finalSignal(validSubcIndices,:,:));
        eqAISignal = eqAISignal(dataIndiceMask, :,:);
        eqAISignal = reshape(eqAISignal, [], 1);
        rxSymAIEQ = pskdemod(eqAISignal, M);
        serAIEQ(idx,:) = errorRateEQ(txSymStream, rxSymAIEQ);

        %% 传统信道估计
        
        % LS信道估计
        CEC.algorithm = 'ls';
        hLS = channelEstimate(rxPilotSignal, pilotSignal, dataIndices, pilotIndices, CEC);
        csiEst_LOSS_LS = csiEst_LOSS_LS + mean(abs(hPerfect(:) - hLS(:)).^2);

        % MMSE信道估计
        CEC.algorithm = 'ls';
        hMMSE = channelEstimate(rxPilotSignal, pilotSignal, dataIndices, pilotIndices, CEC);
        csiEst_LOSS_MMSE = csiEst_LOSS_MMSE + mean(abs(hPerfect(:) - hMMSE(:)).^2);

        % LMMSE信道估计
        CEC.algorithm = 'ls';
        hLMMSE = channelEstimate(rxPilotSignal, pilotSignal, dataIndices, pilotIndices, CEC);
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
