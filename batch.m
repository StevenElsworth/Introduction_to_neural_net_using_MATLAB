rng(0)

% Construct data
n = 20;                             % number of data points
x = linspace(0,1,n);                % known inputs
y = x + 0.25*randn(1,length(x));    % noisy outputs


% Plot data and solve least squares problem.
A = [x', ones(length(x),1)];
param = A\y';
figure()
plot(x, y, 'xk')
hold on
plot(x, param(1)*x + param(2), 'r')
legend('Data Points', 'Least Squares Fit', 'location', 'best')
title('Data and Least Squares Solution')
hold off


linear = @(x, W, b) W*x+b;
d_linear = @(a) 1;

e_R = 2; % factor by which to divide learning rate
N_R = 4; % how often to reduce learning rate



%%  METHOD 1: Stochastic Gradient Descent

Niter = 5e3;
eta = 0.05;

% Initialise weight and bias.
W_stochastic = zeros(Niter,1);  W_stochastic(1) = W; 
b_stochastic = zeros(Niter,1);  b_stochastic(1) = b;

for iterations = 2:Niter
    
    % Select training data point.
    i = randi(length(x), 1);
    
    % Forward pass
    a2 = linear(x(i), W_stochastic(iterations-1), b_stochastic(iterations-1));
    
    % Backward pass
    delta = d_linear(a2)*(a2-y(i));
    
    % Update weights
    W_stochastic(iterations) = W_stochastic(iterations-1) - eta*delta*x(i);
    b_stochastic(iterations) = b_stochastic(iterations-1) - eta*delta;
    
    if(mod(iterations, floor(Niter/N_R)) == 0)
        eta = eta/e_R;
    end
end


%% METHOD 2: Mini Batch

Niter = 5e3;
eta = 0.05;

% Initialise weight and bias.
W_mini = zeros(Niter,1);  W_mini(1) = W; 
b_mini = zeros(Niter,1);  b_mini(1) = b;

% Size of mini batch.
mini_batch = 6;

for iterations = 2:Niter
    
    % Select batch of data training points.
    perm = randperm(length(x));
    perm = perm(1:mini_batch);
    
    % Forward pass
    a2 = linear(x(perm), W_mini(iterations-1), b_mini(iterations-1));

    % Backward pass
    delta = d_linear(a2)*(a2-y(perm));

    % Update weights
    W_mini(iterations) = W_mini(iterations-1) - eta*mean(delta.*x(perm));
    b_mini(iterations) = b_mini(iterations-1) - eta*mean(delta);
    
    if(mod(iterations, floor(Niter/N_R)) == 0)
        eta = eta/e_R;
    end
end


%% METHOD 3: Full Batch

Niter = 3e3;
eta = 0.05;

% Initalise weight and bias.
W_batch = zeros(Niter,1);  W_batch(1) = W; 
b_batch = zeros(Niter,1);  b_batch(1) = b;

for iterations = 2:Niter
    
    % Forward pass
    a2 = linear(x, W_mini(iterations-1), b_mini(iterations-1));

    % Backward pass
    delta = d_linear(a2)*(a2-y);

    % Update weights
    W_batch(iterations) = W_batch(iterations-1) - eta*mean(delta.*x);
    b_batch(iterations) = b_batch(iterations-1) - eta*mean(delta);
    
    if(mod(iterations, floor(Niter/N_R)) == 0)
        eta = eta/e_R;
    end  
end


%% PLOT PATHS:

figure
plot(W_stochastic, b_stochastic)
hold on
plot(W_mini, b_mini)
plot(W_batch, b_batch)
plot(param(1), param(2), 'kx', 'markersize', 20, 'linewidth', 4)
legend('Stochastic', 'Mini Batch', '(Full) Batch')
ylabel('b')
xlabel('W')
