clc;
clear all;

%% Probability density function
x = [-15:.1:15];
normW1 = normpdf(x,0,sqrt(7.5));
normW2 = normpdf(x,7,sqrt(7));
%Plot the pdf.
figure;
plot(x,normW1,'r','LineWidth',2);
hold on;
plot(x,normW2,'g','LineWidth',2);
% Add title and axis labels
xlabel('x')
ylabel('P(x|wi)')
title('Gaussian/Normal Probability Distribution')
dim = [.2 .5 .3 .3];


%% Prior Ration P(W2)/P(W1)=1, Prior probality is equal.
figure
ax1 = subplot(3,1,1); % top subplot
plot(x,normW1,'r','LineWidth',2);
hold on;
plot(x,normW2,'g','LineWidth',2);
hold on
line([3.528 3.528],ylim,'LineWidth',2);
% Add title and axis labels
xlabel('x')
ylabel('P(x|wi)')
title('Gaussian /Normal Probability Distribution')


%% Prior Ration P(W2)/P(W1)=1, Prior probality is equal.
ax2 = subplot(3,1,2); % top subplot
%Plot the pdf.
plot(x,normW1,'r','LineWidth',2);
hold on;
plot(x,normW2,'g','LineWidth',2);
hold on
line([4.249 4.249],ylim,'LineWidth',2);
% Add title and axis labels
xlabel('x')
ylabel('P(x|wi)')
title('Gaussian /Normal Probability Distribution')


%% Prior Ration P(W2)/P(W1)=1, Prior probality is equal.
ax3 = subplot(3,1,3); % top subplot
%Plot the pdf.
plot(x,normW1,'r','LineWidth',2);
hold on;
plot(x,normW2,'g','LineWidth',2);
hold on
line([2.102 2.102],ylim,'LineWidth',2);
% Add title and axis labels
xlabel('x')
ylabel('P(x|wi)')
title('Gaussian /Normal Probability Distribution')



