%% 参数设置
% 系统参数配置
numSubFrame = 10;                                         % 子帧数量
snrValues = 0:5:30;                                       % 信噪比范围
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
pathDelays = [0 0.5e-6 1.2e-6];                           % 路径时延
averagePathGains = [0 -2 -5];                             % 平均路径增益
maxDopplerShift = 50;                                     % 最大多普勒频移

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

% 数据子载波配置
dataIndices = setdiff((numGuardBands(1)+1):(numSubc-numGuardBands(2)),unique(pilotIndices));
numDataSubc = length(dataIndices);
numSubFrameSym = numDataSubc * numSym * numTx;


%% 模型推理结果解析
load('inferResults.mat',...
    '',...
    '',...
    )

load('../raw/compareData.mat',...
    'rxSignalData',...
    'txSignalData',...
    'csiData',...
    '-v7.3')

load('CE_SER_METRIC.mat',...
    'errorPerfectZF', ...
    'errorPerfectMMSE', ...
    'errorLSZF', ...
    'errorLSMMSE', ...
    'errorMMSEZF', ...
    'errorMMSEMMSE',...
    'msePerfectLS', ...
    'msePerfectMMSE', ...
    'msePerfectLMMSE')

errorRateAIZF = comm.ErrorRate;
errorRateAIMMSE = comm.ErrorRate;
errorRateAIPROZF = comm.ErrorRate;
errorRateAIPROMMSE = comm.ErrorRate;
errorRateAIPROMAX = comm.ErrorRate;

errorAIZF = zeros(length(snrValues), 3);
errorAIMMSE = zeros(length(snrValues), 3);
errorAIPROZF = zeros(length(snrValues), 3);
errorAIPROMMSE = zeros(length(snrValues), 3);
errorAIPROMAX = zeros(length(snrValues), 3);

mseAI = zeros(length(snrValues), 1);

for idx = 1:length(snrValues)
    snr = snrValues(idx);
    for frame = 1:numSubFrame

        txSignal = complex(txSignalData(:,:,:,:,:,1),txSignalData(:,:,:,:,:,1));
        txDataSignal = zeros();
        

    end
end
%% 图形1：SER误码率图像
figure(1);
hold on;

% 绘制每种算法的误符号率曲线
plot(snrValues, errorPerfectZF(:, 1), '-o', 'LineWidth', 1.5, 'DisplayName', 'Perfect ZF');
plot(snrValues, errorPerfectMMSE(:, 1), '-s', 'LineWidth', 1.5, 'DisplayName', 'Perfect MMSE');
plot(snrValues, errorLSZF(:, 1), '-d', 'LineWidth', 1.5, 'DisplayName', 'LS ZF');
plot(snrValues, errorLSMMSE(:, 1), '-^', 'LineWidth', 1.5, 'DisplayName', 'LS MMSE');
plot(snrValues, errorMMSEZF(:, 1), '-v', 'LineWidth', 1.5, 'DisplayName', 'MMSE ZF');
plot(snrValues, errorMMSEMMSE(:, 1), '-p', 'LineWidth', 1.5, 'DisplayName', 'MMSE MMSE');

% 设置图形属性
grid on;
xlabel('SNR (dB)');
ylabel('Symbol Error Rate (SER)');
title('SER vs. SNR for Different Channel Estimation and Equalization Algorithms');
legend('Location', 'best');
set(gca, 'YScale', 'log');  % 将 Y 轴设置为对数坐标

% 显示图形
hold off;

%% 图形1：信道估计 MSE LOSS图像
figure(2);
hold on;

% 绘制每种算法的信道估计MSELOSS图像
plot(snrValues, msePerfectLS, '-o', 'LineWidth', 1.5, 'DisplayName', 'LS');
plot(snrValues, msePerfectMMSE, '-s', 'LineWidth', 1.5, 'DisplayName', 'MMSE');
plot(snrValues, msePerfectLMMSE, '-^', 'LineWidth', 1.5, 'DisplayName', 'LMMSE');
% 设置图形属性
grid on;
xlabel('SNR (dB)');
ylabel('MSE with h_{Perfect}');
title('Channel Estimation MSE vs. SNR');
legend('Location', 'best');
hold off;

