%% 
% close all force;
% all clc;all clear;
load('class_SY_GLP_PP.mat');
% X=log(X);
% X = X(:,1:12800);
figure
plot(X(1:20,:)','r-o');             % ��ĥ �� 1
hold on;
plot(X(21:56,:)','black-*');        % ���� �� 2
hold on;
plot(X(57:end,:)','g--');           % ʪĥ �� 3


grid on

% hold on;
% plot(X(85:end,:)','--g');         % �� 3

% %% ��һ��
% close all force;
% all clc;all clear;
% load('class_ZD_1_GLP.mat')
% X=20*log10(X);
% X = NormalizeL2(X,1);
% 
% % [X_train,inputps] = mapminmax(X');
% % X_test = mapminmax('apply',X2',inputps);
% % X = X_train';X2 = X_test';
% % 
% 
% plot(X(1:20,:)','r-o');             % ��ĥ �� 1
% hold on;
% plot(X(21:56,:)','black-*');        % ���� �� 2
% hold on;
% plot(X(57:end,:)','g');             % ʪĥ �� 3
% grid on
