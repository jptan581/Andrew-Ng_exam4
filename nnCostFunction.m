function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));   %Theta1为25*401矩阵
Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));         %Theta2为10*26矩阵
m = size(X, 1);
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

X = [ones(m,1),X];         %X为5000*401矩阵
z2 = X * Theta1';          %z2为5000*25矩阵
a2 = sigmoid(z2);          
a2 = [ones(m,1),a2];       %a2为5000*26矩阵
z3 = a2 * Theta2';         %z3为5000*10矩阵
a3 = sigmoid(z3);          %a3为5000*10矩阵

ylabel = zeros(num_labels,m);  %ylabel为10*5000矩阵
for i=1:m
    ylabel(y(i),i) = 1;
end

y1 = sum(ylabel.*log(a3)');
y2 = sum((1 - ylabel).*log(1-a3)');
J = -1/m * sum(y1 + y2);

theta1_all = sum(Theta1(:,2:end).^2);
theta2_all = sum(Theta2(:,2:end).^2);
J = J +lambda/2/m * (sum(theta1_all)+sum(theta2_all));

z2 = [ones(m,1),z2];
Delta1 = zeros(size(Theta1));
Delta2 = zeros(size(Theta2));

delta3 = a3' - ylabel;                                %detal3为10*5000
delta2 = (Theta2'*delta3).*sigmoidGradient(z2)';      %detal2为26*5000
Delta1 = delta2(2:end,:) * X;                         %Delta1为25*401矩阵
Delta2 = delta3 * a2;                                 %Delta2为10*26矩阵

% for t = 1:m
%     delta3 = a3(t,:)' - ylabel(:,t);                          %delta3为10*1矩阵，列循环5000次
%     delta2 = (Theta2'*delta3).*sigmoidGradient(z2(t,:)');     %delta2为26*1矩阵，列循环5000次
%     Delta1 = Delta1 + delta2(2:end) * X(t,:);                 %Delta1为25*401矩阵
%     Delta2 = Delta2 + delta3 * a2(t,:);                       %Delta2为10*26矩阵
% end

Theta1_grad = Delta1/m;                                                      %Theta1_grad第一列
Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + lambda/m*Theta1(:,2:end);      %Theta1_grad
Theta2_grad = Delta2 /m;
Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + lambda/m*Theta2(:,2:end);

grad = [Theta1_grad(:) ; Theta2_grad(:)];
end
