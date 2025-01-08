% 假设参数
numsubc = 72;   % 总子载波数
numsym = 14;    % 符号数
numdatasubc = 50; % 数据子载波数

% 随机生成 h_ls (导频信道估计值)
h_ls = rand(numsubc, numsym) + 1j*rand(numsubc, numsym);

% 生成导频位置 pilot_indices
[pilot_subc, pilot_sym] = meshgrid(1:6:numsubc, 1:numsym); % 每6个子载波插入导频
pilot_indices = [pilot_subc(:), pilot_sym(:)];

% 提取导频的信道值
pilot_values = h_ls(sub2ind(size(h_ls), pilot_subc(:), pilot_sym(:)));

% 生成目标位置 H_indices (数据子载波的网格)
[data_subc, data_sym] = meshgrid(linspace(1, numsubc, numdatasubc), 1:numsym);
H_indices = [data_subc(:), data_sym(:)];

% 使用 griddata 进行插值
h_interpolated = griddata(pilot_indices(:, 1), pilot_indices(:, 2), pilot_values, ...
                          H_indices(:, 1), H_indices(:, 2), 'linear');

% 将结果恢复到矩阵形式
h_interp_matrix = reshape(h_interpolated, numdatasubc, numsym);

% 可视化结果
figure;
subplot(1, 2, 1);
imagesc(abs(h_ls));
title('原始信道 (h_{ls})');
xlabel('符号索引'); ylabel('子载波索引');
colorbar;

subplot(1, 2, 2);
imagesc(abs(h_interp_matrix));
title('插值后的信道');
xlabel('符号索引'); ylabel('数据子载波索引');
colorbar;
