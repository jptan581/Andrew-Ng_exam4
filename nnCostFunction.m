function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));   %Theta1Ϊ25*401����
Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));         %Theta2Ϊ10*26����
m = size(X, 1);
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

X = [ones(m,1),X];         %XΪ5000*401����
z2 = X * Theta1';          %z2Ϊ5000*25����
a2 = sigmoid(z2);          
a2 = [ones(m,1),a2];       %a2Ϊ5000*26����
z3 = a2 * Theta2';         %z3Ϊ5000*10����
a3 = sigmoid(z3);          %a3Ϊ5000*10����

ylabel = zeros(num_labels,m);  %ylabelΪ10*5000����
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

delta3 = a3' - ylabel;                                %detal3Ϊ10*5000
delta2 = (Theta2'*delta3).*sigmoidGradient(z2)';      %detal2Ϊ26*5000
Delta1 = delta2(2:end,:) * X;                         %Delta1Ϊ25*401����
Delta2 = delta3 * a2;                                 %Delta2Ϊ10*26����

% for t = 1:m
%     delta3 = a3(t,:)' - ylabel(:,t);                          %delta3Ϊ10*1������ѭ��5000��
%     delta2 = (Theta2'*delta3).*sigmoidGradient(z2(t,:)');     %delta2Ϊ26*1������ѭ��5000��
%     Delta1 = Delta1 + delta2(2:end) * X(t,:);                 %Delta1Ϊ25*401����
%     Delta2 = Delta2 + delta3 * a2(t,:);                       %Delta2Ϊ10*26����
% end

Theta1_grad = Delta1/m;                                                      %Theta1_grad��һ��
Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + lambda/m*Theta1(:,2:end);      %Theta1_grad
Theta2_grad = Delta2 /m;
Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + lambda/m*Theta2(:,2:end);

grad = [Theta1_grad(:) ; Theta2_grad(:)];
end
