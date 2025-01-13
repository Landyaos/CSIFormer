clear all;

close all;

fprintf('OFDM�ŵ����Ʒ���\n\n');

carrier_count=64;%-----------�ز���Ŀ

num_symbol=50;%--------------OFDM���Ÿ���

Guard=8;%--------------------ѭ��ǰ׺

pilot_Inter=8;%--------------��Ƶ���

modulation_mode=4;%---------���Ʒ�ʽ

SNR=[1:1:12];%-------------�����ȡֵ

NumLoop=10;%-----------------ѭ������

num_bit_err=zeros(length(SNR),NumLoop);

num_bit_err_dft=zeros(length(SNR),NumLoop);

num_bit_err_ls=zeros(length(SNR),NumLoop);

num_bit_err_mmse=zeros(length(SNR),NumLoop);

MSE=zeros(length(SNR),NumLoop);

MSE1=zeros(length(SNR),NumLoop);

MSE2=zeros(length(SNR),NumLoop);

%%%%%%%%%%%%%%%%%%%%%%%������ѭ��%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for c1=1:length(SNR)

fprintf('\n\n\n���������Ϊ%f\n\n',SNR(c1));

for num1=1:NumLoop

%---------------�������͵�������С���������������������������-

BitsLen=carrier_count*num_symbol;

BitsTx=randi([0 3],1,BitsLen);

%---------------���ŵ���---------------------------------------

Modulated_Sequence=qammod(BitsTx,modulation_mode);

%---------------��Ƶ��ʽ---------------------------------------

% pilot_len=carrier_count;
% 
% pilot_symbols=round(rand(1,pilot_len));
% 
% for i=1:pilot_len
% 
% if pilot_symbols(1,i)==0
% 
% pilot_symbols(1,i)=pilot_symbols(1,i)-1;
% 
% else
% 
% pilot_symbols(1,i)=pilot_symbols(1,i);
% 
% end
% 
% end
% 
% pilot_symbols=pilot_symbols';
 pilot_symbols=qammod(randi([0 3],64,1),4);
%----------------���㵼Ƶ��������Ŀ----------------------------

num_pilot=ceil(num_symbol/pilot_Inter);

if rem(num_symbol,pilot_Inter)==0

num_pilot=num_pilot+1;

end

num_data=num_symbol+num_pilot;

%----------------��Ƶλ�ü���----------------------------------

pilot_Indx=zeros(1,num_pilot);

Data_Indx=zeros(1,num_pilot*(pilot_Inter+1));

for i=1:num_pilot-1

pilot_Indx(1,i)=(i-1)*(pilot_Inter+1)+1;

end

pilot_Indx(1,num_pilot)=num_data;

for j=0:num_pilot

Data_Indx(1,(1+j*pilot_Inter):(j+1)*pilot_Inter)=(2+j*(pilot_Inter+1)):((j+1)*(pilot_Inter+1));

end

Data_Indx=Data_Indx(1,1:num_symbol);

%----------------��Ƶ����-------------------------------------

piloted_ofdm_syms=zeros(carrier_count,num_data);

piloted_ofdm_syms(:,Data_Indx)=reshape(Modulated_Sequence,carrier_count,num_symbol);

piloted_ofdm_syms(:,pilot_Indx)=repmat(pilot_symbols,1,num_pilot);

%----------------IFFT�任��������������������������������������

time_signal=sqrt(carrier_count)*ifft(piloted_ofdm_syms);

%----------------��ѭ��ǰ׺------------------------------------

add_cyclic_signal=[time_signal((carrier_count-Guard+1:carrier_count),:);time_signal];

Tx_data_trans=reshape(add_cyclic_signal,1,(carrier_count+Guard)*num_data);

%----------------�ŵ�����--------------------------------------

Tx_signal_power=sum(abs(Tx_data_trans(:)).^2)/length(Tx_data_trans(:));

noise_var=Tx_signal_power/(10^(SNR(c1)/10));

Rx_data=awgn(Tx_data_trans,SNR(c1),'measured');

%----------------�źŽ��ա�ȥѭ��ǰ׺��FFT�任-----------------

Rx_signal=reshape(Rx_data,(carrier_count+Guard),num_data);

Rx_signal_matrix=zeros(carrier_count,num_data);

Rx_signal_matrix=Rx_signal(Guard+1:end,:);

Rx_carriers=fft(Rx_signal_matrix)/sqrt(carrier_count);

%----------------��Ƶ��������ȡ--------------------------------

Rx_pilot=Rx_carriers(:,pilot_Indx);

Rx_fre_data=Rx_carriers(:,Data_Indx);

%----------------��Ƶλ���ŵ���ӦLS����------------------------

pilot_patt=repmat(pilot_symbols,1,num_pilot);

pilot_esti=Rx_pilot./pilot_patt;

%----------------LS���Ƶ����Բ�ֵ������������������������������

int_len=pilot_Indx;

len=1:num_data;

for ii=1:carrier_count

channel_H_ls(ii,:)=interp1(int_len,pilot_esti(ii,1:(num_pilot)),len,'spline');

end

channel_H_data_ls=channel_H_ls(:,Data_Indx);

%----------------LS�����з������ݵĹ���ֵ----------------------

Tx_data_estimate_ls=Rx_fre_data.*conj(channel_H_data_ls)./(abs(channel_H_data_ls).^2);

%----------------DFT����--------------------------------------

Tx_pilot_estimate_ifft=ifft(pilot_esti);

padding_zero=zeros(64,num_pilot);

Tx_pilot_estimate_ifft_padding_zero=[Tx_pilot_estimate_ifft;padding_zero];

Tx_pilot_estimate_dft=fft(Tx_pilot_estimate_ifft_padding_zero);

%----------------DFT���Ƶ����Բ�ֵ������������������������������

int_len=pilot_Indx;

len=1:num_data;

for ii=1:carrier_count

channel_H_dft(ii,:)=interp1(int_len,Tx_pilot_estimate_dft(ii,1:(num_pilot)),len,'linear');

end

channel_H_data_dft=channel_H_dft(:,Data_Indx);

%----------------DFT�����з������ݵĹ���ֵ----------------------

Tx_data_estimate_dft=Rx_fre_data.*conj(channel_H_data_dft)./(abs(channel_H_data_dft).^2);
%----------------MMSE����--------------------------------------
beta=1;
trms=1e-6;
t_max=8;
N=carrier_count;
Rhh=zeros(N,N);
for k=1:N
    for l=1:N
        Rhh(k,l)=(1-exp((-1)*t_max*((1/trms)+1i*2*pi*(k-l)/N)))./(trms*(1-exp((-1)*t_max/trms))*((1/trms)+1i*2*pi*(k-l)/N));
    end
end

Q=Rhh*(eye(N)/(Rhh+(beta/SNR(c1))*eye(N)));
Hmmse=Q*pilot_esti;

%----------------MMSE�����з������ݵĹ���ֵ----------------------
int_len=pilot_Indx;

len=1:num_data;

for ii=1:carrier_count

channel_H_mmse(ii,:)=interp1(int_len,Hmmse(ii,1:(num_pilot)),len,'linear');

end

channel_H_data_mmse=channel_H_mmse(:,Data_Indx);

Tx_data_estimate_mmse=Rx_fre_data.*conj(channel_H_data_mmse)./(abs(channel_H_data_mmse).^2);
%----------------DFT���Ž��------------------------------------

demod_in_dft=Tx_data_estimate_dft(:).';

demod_out_dft=qamdemod(demod_in_dft,modulation_mode);

%----------------LS���Ž��------------------------------------

demod_in_ls=Tx_data_estimate_ls(:).';

demod_out_ls=qamdemod(demod_in_ls,modulation_mode);


%----------------LS���Ž��------------------------------------

demod_in_mmse=Tx_data_estimate_mmse(:).';

demod_out_mmse=qamdemod(demod_in_mmse,modulation_mode);
%----------------�����ʵļ���----------------------------------

for i=1:length(BitsTx)

if demod_out_dft(i)~=BitsTx(i)

num_bit_err_dft(c1,num1)=num_bit_err_dft(c1,num1)+1;

end

if demod_out_ls(i)~=BitsTx(i)

num_bit_err_ls(c1,num1)=num_bit_err_ls(c1,num1)+1;

end

if demod_out_mmse(i)~=BitsTx(i)

num_bit_err_mmse(c1,num1)=num_bit_err_mmse(c1,num1)+1;

end

end

end

end

BER_dft=mean(num_bit_err_dft.')/length(BitsTx);

BER_ls=mean(num_bit_err_ls.')/length(BitsTx);

BER_mmse=mean(num_bit_err_mmse.')/length(BitsTx);
%%%%%%%%%%%%%%%%%%%a%%%%%%%%������ѭ��������%%%%%%%%%%%%%%%%%%%%%%%%%%%%

figure

semilogy(SNR,BER_dft,'-go',SNR,BER_ls,'-k+',SNR,BER_mmse,'-b*');
legend('dft','ls','mmse');
title('OFDMϵͳ��LS��DFT��MMSE�ŵ�����');
xlabel('SNR'),ylabel('BER')