classdef IdealMIMOChannel < matlab.System
    % IdealMIMOChannel 自定义理想MIMO信道模型
    % 该模型模拟了一个理想的MIMO信道，去除了噪声，且输入符号矩阵按时隙为单位

    properties
        NumTransmitAntennas   % 发射天线数量
        NumReceiveAntennas    % 接收天线数量
        NumSubcarriers        % 子载波数
        NumSymbols            % 每个子载波上的符号数
    end
    
    properties(Access = private)
        H % 信道矩阵，维度为 (numSubcarriers, numSymbols, numRx, numTx)
    end
    
    methods
        % 构造函数
        function obj = IdealMIMOChannel(varargin)
            setProperties(obj, nargin, varargin{:});
        end
        
        % 重写 setupImpl 方法，初始化信道矩阵
        function setupImpl(obj)
            % 初始化信道矩阵 H
            obj.H = (randn(obj.NumSubcarriers, obj.NumSymbols, obj.NumReceiveAntennas, obj.NumTransmitAntennas) ...
                   + 1i * randn(obj.NumSubcarriers, obj.NumSymbols, obj.NumReceiveAntennas, obj.NumTransmitAntennas)) / sqrt(2);
        end
        
        % 信道模型的核心处理方法
        function [y, H] = stepImpl(obj, x)
            % x: 输入信号 (numSubcarriers x symbols x numTx)
            % y: 输出信号 (numSubcarriers x symbols x numRx)
            % H: 当前符号的信道矩阵 (numSubcarriers x numRx x numTx)

            % 获取输入的维度
            [numSubcarriers, numSymbols, ~] = size(x);
            
            % 初始化输出信号矩阵
            y = zeros(numSubcarriers, numSymbols, obj.NumReceiveAntennas);
            
            % 对每个符号进行信号传播
            for j = 1:numSymbols
                % 提取第 j 个符号的信道矩阵 H_j (numSubcarriers x numRx x numTx)
                H_j = squeeze(obj.H(:, j, :, :));
                
                % 对每个子载波执行信道传播
                for i = 1:numSubcarriers
                    % 信号传播：y(i, j, :) = H_j(i, :, :) * x(i, j, :)
                    y(i, j, :) = squeeze(H_j(i, :, :)) * squeeze(x(i, j, :))';
                end
            end
            
            % 返回当前符号的信道矩阵 H
            H = obj.H;
        end
    end
end
