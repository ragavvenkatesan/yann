% License: MIT
% (c) 2017 Ragav Venkatesan
%
% Code to convert a (downlaoded from online) dataset to proper format
% usable by yann to setup the dataset in its own internal format.
% As seen here in this folder. (These are processed mats.
%
% Download test_32x32.mat, train_32x32.mat and extra_32x32.mat from 
% http://ufldl.stanford.edu/housenumbers/ website, format 2.
% Then run this code. 
%
% batch_sizing and number of batches are parameters that can be 
% set in the second stage of code.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all
load('test_32x32.mat');
x_test = shiftdim(X,3);
x_test = x_test(:,:);
x_test = double(x_test);
y_test = y;
clear X;
clear y;

load('train_32x32.mat');
x_train = shiftdim(X,3);
x_train = x_train(:,:);
x_train = double(x_train);
y_train = y;
clear X;
clear y;

load('extra_32x32.mat');
x_valid = shiftdim(X,3);
x_valid = x_valid(:,:);
x_valid = double(x_valid);
y_valid = y;
clear X;
clear y;

x = [x_train; x_test; x_valid];
y = [y_train; y_test; y_valid];

clearvars -except x y 
% save ('data.mat' , '-v7.3');
mkdir('train');
mkdir('test');
mkdir('valid');

%% 
% Going to throw away 420 samples.
throw_away = 420; 
batch_size = 500;
test_size = 130;
train_size = 1000;

data = x (1:length(x) - throw_away,:);
labels = y (1:length(y) - throw_away) - 1; % because labels go from 1-10

total_batches = length(labels) / batch_size;
remain = total_batches - test_size; 

remain = remain - train_size;
valid_size = remain; 

clear x
clear y;
%% 

x = data(  1:train_size * batch_size ,:);
y = labels(1:train_size * batch_size);
dump( 'train',10, batch_size, x, y );

x = data(  train_size * batch_size + 1 : train_size * batch_size + test_size * batch_size ,:);
y = labels(train_size * batch_size + 1 : train_size * batch_size + test_size * batch_size);
dump( 'test',10, batch_size, x, y );

x = data(  (train_size + test_size) * batch_size + 1 : end ,:);
y = labels((train_size + test_size) * batch_size + 1 : end);
dump( 'valid',10, batch_size, x, y );
