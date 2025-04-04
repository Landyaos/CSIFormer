clear;
clc;%1.0908e+09
filename = 'raw/20250213_204633.mat';
% filename = '20250330_152257.mat'; %
% filename = '20250330_101645.mat';
filename = '20250330_100844.mat'; %%%
% filename = '20250330_094826.mat';
filename = '20250330_094523.mat';
% filename = '20250330_093717.mat'; %%
% filename = '20250330_183615.mat'; %%
filename = '20250331_072836.mat'; %%

snrValues = 0:3:30;

load(filename, ...
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
seed
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
plot(snrValues, mse_csi_mmse,      '-d', 'Color', colors(2,:), 'LineWidth', 1, 'DisplayName', 'LMMSE');
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
plot(snrValues, ser_mmse_mmse(:,1),        '-o', 'Color', colors(6,:),  'LineWidth', 1, 'DisplayName', 'LMMSE');
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
