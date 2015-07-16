clc
clear all
close all
%%
results_file = '../results/results_mnist.txt';
errors_file = '../results/error_mnist.txt';
cost_file = '../results/cost_mnist.txt';
result = load(results_file);
labels = result(:,2);
predictions = result(:,3);
nll = result(:,[4:5]);

%%
[X,Y,T,AUC] = perfcurve(labels,nll(:,2),1);
plot(X,Y);
hold on;
xlabel('False positive rate')
ylabel('True positive rate')
title('ROC Curve');
legend(sprintf('AUC = %0.4f',AUC));
%%
error = load(errors_file);
figure;
plot(error);
title('Error Rates');

figure;
cost = load(cost_file);
plot(cost);
title('Cost Rates');