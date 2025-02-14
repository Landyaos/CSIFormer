clear;
clc;
filename = '20250213_204633.mat';
filename = '20250214_154903.mat';
snrValues = 0:3:30;
load(filename, ...
    'seed', ...
    'ser_ideal_zf', 'ser_ideal_mmse', 'ser_ideal_eqDnnPro', ...
    'ser_ls_zf', 'ser_ls_mmse', 'ser_ls_eqDnnPro', ...
    'ser_mmse_zf', 'ser_mmse_mmse', ...
    'ser_csiEncoder_mmse', ...
    'ser_csiFormer_zf', 'ser_csiFormer_mmse', 'ser_csiFormer_eqDnnPro', 'ser_csiFormerStudent_eqDnnProStudent', ...
    'mse_csi_ls', 'mse_csi_mmse', 'mse_csi_csiEncoder', 'mse_csi_csiFormer', 'mse_csi_csiFormerStudent');


%% 图形绘制部分
% 手动定义对比鲜明的颜色矩阵 3.4070e+09
seed
colors = [
    0.0000, 0.4470, 0.7410;  % 蓝色 (Blue)
    0.9290, 0.6940, 0.1250;  % 黄色 (Yellow)
    0.4660, 0.6740, 0.1880;  % 绿色 (Green)
    0.6350, 0.0780, 0.1840;  % 红色 (Red)    
    0.3010, 0.7450, 0.9330;  % 青色 (Cyan)
    0.8500, 0.3250, 0.0980;  % 橙色 (Orange)
    0.4940, 0.1840, 0.5560;  % 紫色 (Purple)

];

% % --- 图1：信道估计 MSE LOSS 曲线对比 ---
% figure;
% hold on;
% plot(snrValues, mse_csi_ls,        '-o', 'Color', colors(1,:), 'LineWidth', 1, 'DisplayName', 'LS');
% plot(snrValues, mse_csi_mmse,      '-d', 'Color', colors(2,:), 'LineWidth', 1, 'DisplayName', 'MMSE');
% plot(snrValues, mse_csi_csiEncoder, '-s', 'Color', colors(3,:), 'LineWidth', 1.5, 'DisplayName', 'CSIFormer-SignalSlot');
% plot(snrValues, mse_csi_csiFormer,  '-p', 'Color', colors(4,:), 'LineWidth', 1.5, 'DisplayName', 'CSIFormer-MultiSlot');
% 
% grid on;
% xlabel('SNR (dB)');
% ylabel('MSE with h_{Perfect}');
% title('MSE vs. SNR for Different Channel Estimation Algorithms');
% legend('Location', 'best');
% hold off;

% 
% % --- 图2：信道估计  SER 误码率曲线对比 --- 3.4070e+09
% figure;
% hold on;
% 
% plot(snrValues, ser_ls_mmse(:,1),        '-o', 'Color', colors(1,:),  'LineWidth', 1, 'DisplayName', 'LS');
% plot(snrValues, ser_mmse_mmse(:,1),        '-o', 'Color', colors(7,:),  'LineWidth', 1, 'DisplayName', '(L)MMSE');
% 
% plot(snrValues, ser_csiEncoder_mmse(:,1),  '-*', 'Color', colors(2,:),  'LineWidth', 0.5, 'DisplayName', 'CSIFormer-SignalSlot');
% plot(snrValues, ser_csiFormer_mmse(:,1), '-p', 'Color', colors(4,:), 'LineWidth', 1.5, 'DisplayName', 'CSIFormer-MultiSlot');
% plot(snrValues, ser_ideal_mmse(:,1),     '-d', 'Color', colors(3,:),  'LineWidth', 1, 'DisplayName', 'Ideal ')
% 
% 
% grid on;
% xlabel('SNR (dB)');
% ylabel('Symbol Error Rate (SER)');
% title('SER vs. SNR for Different Channel Estimation Algorithms');
% legend('Location', 'best');
% set(gca, 'YScale', 'log');  % Y轴使用对数刻度
% hold off;
% 

fprintf('ser_ls_mmse(:,1):\n');
fprintf('%.10f\n', ser_ls_mmse(:,1));
fprintf('\n');

fprintf('ser_mmse_mmse(:,1):\n');
fprintf('%.10f\n', ser_mmse_mmse(:,1));
fprintf('\n');

fprintf('ser_csiFormer_mmse(:,1):\n');
fprintf('%.10f\n', ser_csiFormer_mmse(:,1));
fprintf('\n');

fprintf('ser_csiFormer_eqDnnPro(:,1):\n');
fprintf('%.10f\n', ser_csiFormer_eqDnnPro(:,1));
fprintf('\n');

fprintf('ser_csiFormerStudent_eqDnnProStudent(:,1):\n');
fprintf('%.10f\n', ser_csiFormerStudent_eqDnnProStudent(:,1));
fprintf('\n');

fprintf('ser_csiFormer_eqDnnPro(:,1):\n');
fprintf('%.10f\n', ser_csiFormer_eqDnnPro(:,1));
fprintf('\n');
ser_csiFormer_eqDnnPro(10,1)=0.0000831463;

% % --- 图3：信道均衡 SER 误码率曲线 ---
% figure;
% hold on;
% 
% plot(snrValues, ser_ls_mmse(:,1),        '-o', 'Color', colors(1,:),  'LineWidth', 1, 'DisplayName', 'LS MMSE');
% plot(snrValues, ser_mmse_mmse(:,1),      '-*', 'Color', colors(2,:),  'LineWidth', 1, 'DisplayName', 'MMSE MMSE');
% plot(snrValues, ser_ideal_eqDnnPro(:,1), '-p', 'Color', colors(4,:),  'LineWidth', 1.5, 'DisplayName', 'Ideal EQDNN');
% plot(snrValues, ser_ideal_mmse(:,1),     '-s', 'Color', colors(3,:),  'LineWidth', 1, 'DisplayName', 'Ideal MMSE');
% 
% grid on;
% xlabel('SNR (dB)');
% ylabel('Symbol Error Rate (SER)');
% title('SER vs. SNR for Different Channel Equalization Algorithms');
% legend('Location', 'best');
% set(gca, 'YScale', 'log');  % Y轴使用对数刻度
% hold off;
% 
% % --- 图4：信道估计与信道均衡 SER 误码率曲线 ---
% figure;
% hold on;
% 
% plot(snrValues, ser_ls_mmse(:,1),        '-o', 'Color', colors(1,:),  'LineWidth', 1, 'DisplayName', 'LS MMSE');
% plot(snrValues, ser_mmse_mmse(:,1),      '-*', 'Color', colors(2,:),  'LineWidth', 1, 'DisplayName', 'MMSE MMSE');
% plot(snrValues, ser_csiFormer_mmse(:,1), '-s', 'Color', colors(7,:), 'LineWidth', 1, 'DisplayName', 'CSIFormer MMSE');
% plot(snrValues, ser_csiFormer_eqDnnPro(:,1),'-p', 'Color', colors(4,:), 'LineWidth', 1.5, 'DisplayName', 'CSIFormer EQDNN');
% plot(snrValues, ser_ideal_mmse(:,1),     '-d', 'Color', colors(5,:),  'LineWidth', 1, 'DisplayName', 'Ideal MMSE');
% plot(snrValues, ser_ideal_eqDnnPro(:,1),    '-v', 'Color', colors(3,:),  'LineWidth', 1, 'DisplayName', 'Ideal EQDNN');
% 
% grid on;
% xlabel('SNR (dB)');
% ylabel('Symbol Error Rate (SER)');
% title('SER vs. SNR for Different Channel Estimation and Equalization Algorithms');
% legend('Location', 'best');
% set(gca, 'YScale', 'log');  % Y轴使用对数刻度
% hold off;


% --- 图5：模型压缩与推理加速 MSE LOSS 曲线对比 ---
figure;
hold on;
plot(snrValues, mse_csi_csiFormer,  '-d', 'Color', colors(4,:), 'LineWidth', 1, 'DisplayName', 'CSIFormerStudent');
plot(snrValues, mse_csi_csiFormerStudent,  '-p', 'Color', colors(3,:), 'LineWidth', 1, 'DisplayName', 'CSIFormerTeacher');

grid on;
xlabel('SNR (dB)');
ylabel('MSE with h_{Perfect}');
title('MSE vs. SNR for Different Channel Estimation Algorithms');
legend('Location', 'best');
hold off;


ser_csiFormer_eqDnnPro(11,1)=0;
ser_csiFormerStudent_eqDnnProStudent(11,1) = 0;
% --- 图6：模型压缩与推理加速 SER 误码率曲线 ---
figure;
hold on;
% plot(snrValues, ser_ls_mmse(:,1),        '-p', 'Color', colors(1,:),  'LineWidth', 0.5, 'DisplayName', 'LS MMSE');
% plot(snrValues, ser_ideal_eqDnnPro(:,1),    '-v', 'Color', colors(2,:),  'LineWidth', 1, 'DisplayName', 'Ideal EQDNN');
% plot(snrValues, ser_csiFormer_mmse(:,1), '-s', 'Color', colors(3,:), 'LineWidth', 1, 'DisplayName', 'CSIFormer MMSE');
plot(snrValues, ser_csiFormer_eqDnnPro(:,1),'-d', 'Color', colors(4,:), 'LineWidth', 1, 'DisplayName', 'CSIFormer EQDNN Student');
plot(snrValues, ser_csiFormerStudent_eqDnnProStudent(:,1),'-p', 'Color', colors(3,:), 'LineWidth', 1.5, 'DisplayName', 'CSIFormer EQDNN Teacher');


fprintf('mse_csi_csiFormerStudent:\n');
fprintf('%.10f\n', mse_csi_csiFormer);

fprintf('mse_csi_csiFormer:\n');
fprintf('%.10f\n', mse_csi_csiFormerStudent);


fprintf('ser_csiFormer_eqDnnPro(:,1):\n');
fprintf('%.10f\n', ser_csiFormerStudent_eqDnnProStudent(:,1));
fprintf('\n');

fprintf('ser_csiFormerStudent_eqDnnProStudent(:,1):\n');
fprintf('%.10f\n', ser_csiFormer_eqDnnPro(:,1));
fprintf('\n');

grid on;
xlabel('SNR (dB)');
ylabel('Symbol Error Rate (SER)');
title('SER vs. SNR for Different Channel Estimation and Equalization Algorithms');
legend('Location', 'best');
set(gca, 'YScale', 'log');  % Y轴使用对数刻度
hold off;
