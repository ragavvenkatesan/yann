clc
clear all
close all
%%
results_file = 'results/results.txt';
errors_file = 'results/error.txt';
cost_file = 'results/cost.txt';
confusion_file = 'results/confusion.txt';
%%
figure;
subplot(2,2,1);
result = load(results_file);
labels = result(:,2);
predictions = result(:,3);
nll = result(:,4:5);
[X,Y,T,AUC] = perfcurve(labels,nll(:,2),1);
plot(X,Y);
hold on;
xlabel('False positive rate')
ylabel('True positive rate')
title('ROC Curve');
legend(sprintf('AUC = %0.4f',AUC));
%%
error = load(errors_file);
subplot(2,2,2);
plot(error(:,1));
hold on;
plot(error(:,2));
hold on;
legend('Validation Error', 'Training Error');
xlabel('Epoch Number');
ylabel('Number of samples that are wrongly predicted');
title('Error Rates');

%% 
subplot(2,2,3);
cost = load(cost_file);
plot(cost);
title('Cost Rates');
%% 
confusion = load(confusion_file);
confusion_mat = confusion ./max(confusion(:));
subplot(2,2,4);
imagesc(confusion_mat);            %# Create a colored plot of the matrix values
colormap(flipud(gray));  %# Change the colormap to gray (so higher values are
                         %#   black and lower values are white)

textStrings = num2str(confusion_mat(:),'%0.2f');  %# Create strings from the matrix values
textStrings = strtrim(cellstr(textStrings));  %# Remove any space padding
[x,y] = meshgrid(1:length(confusion_mat));   %# Create x and y coordinates for the strings
hStrings = text(x(:),y(:),textStrings(:),...      %# Plot the strings
                'HorizontalAlignment','center');
midValue = mean(get(gca,'CLim'));  %# Get the middle value of the color range
textColors = repmat(confusion_mat(:) > midValue,1,3);  %# Choose white or black for the
                                             %#   text color of the strings so
                                             %#   they can be easily seen over
                                             %#   the background color
set(hStrings,{'Color'},num2cell(textColors,2));  %# Change the text colors
title('Confusion Matrix');
