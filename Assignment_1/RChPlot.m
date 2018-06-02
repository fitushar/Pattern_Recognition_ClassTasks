function RChPlot(R,Q,titlestring)
% RChPlot(R,Q,titlestring)
%
% Creates a new figure with the graphs of the ROC curve provided in the
% matrix R and of the convex hull provided in the matrix Q.
% titlestring: the title of the figure
% If Q=[], only the ROC curve is drawn.

h=figure;
if(isempty(Q))
    plot(R(:,1),R(:,2),'-k.');
    legend('ROC curve','Location','southeast');
else
   hold on;
   plot(R(:,1),R(:,2),'-k.','MarkerSize',10);
   plot(Q(:,1),Q(:,2),'-ko','LineWidth',2,'MarkerSize',10);
   legend('ROC curve','Convex Hull','Location','southeast');
end
set(gca,'XTick',0.0:0.2:1.0);
set(gca,'YTick',0.0:0.2:1.0);
set(gca,'XTickLabel',{'0.0';'0.2';'0.4';'0.6';'0.8';'1.0'});
set(gca,'YTickLabel',{'0.0';'0.2';'0.4';'0.6';'0.8';'1.0'});
xlabel('FPR','FontSize',12,'FontWeight','bold');
ylabel('TPR','FontSize',12,'FontWeight','bold');
set(gca,'FontSize',12,'FontWeight','bold');
set(gca,'Box','on','Layer','top');
title(titlestring);
axis equal tight;
set(h,'NumberTitle','on','Name',titlestring);
