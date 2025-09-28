%% run_CA_Spiral_IMM_and_CAUKF.m
% -------------------------------------------------------------------------
% Runs CA-EKF, Spiral-EKF, IMM (CA+Spiral) as in RL_IMM_SpiralEKF_v5.m
% and CA-UKF as in CAUKF_5minRandom_RL_Tuned_v1.m
% Produces 3D tracks, per-axis tracking (with noisy meas), error vs time,
% and prints post-transient RMSE/Var for each axis and each filter.
% -------------------------------------------------------------------------

clear; clc; close all;

%% 1) Shared Parameters (exact config)
T        = 0.01;                  % sample period
SKIP     = 3000;                  % post-transient start (inclusive)
radarSTD = 50;
R_true   = radarSTD^2 * eye(3);
R_EKF    = R_true;

% IMM Transition Probability Matrix (same as your code)
TPM = [0.95 0.05;
       0.45 0.55];

% CA-EKF model matrices (exact)
F_e = [1 0 0 T 0 0 0.5*T^2 0 0;
       0 1 0 0 T 0 0 0.5*T^2 0;
       0 0 1 0 0 T 0 0 0.5*T^2;
       0 0 0 1 0 0 T 0 0;
       0 0 0 0 1 0 0 T 0;
       0 0 0 0 0 1 0 0 T;
       0 0 0 0 0 0 1 0 0;
       0 0 0 0 0 0 0 1 0;
       0 0 0 0 0 0 0 0 1];
H   = [eye(3), zeros(3,6)];
Hs  = [eye(3), zeros(3,3)];      % for Spiral-EKF

%% 2) Load Q_EKF (exact same file/field); assert like your code
ca_mat = 'CAEKF_5minRandom_RL_Tuned_v4.mat';
assert(isfile(ca_mat), 'Missing CA-EKF results file: %s', ca_mat);
S = load(ca_mat,'top_ca');
[~,idxE] = max([S.top_ca.survival]);
Q_EKF = S.top_ca(idxE).Q;

%% 3) Load Q_Spiral from IMM leaderboard (same file/field), with safe fallback
spiral_mat = 'IMM_SpiralEKF_RL_Tuned.mat';
if isfile(spiral_mat)
    Z = load(spiral_mat,'top_ekf');
    [~,bestIdx] = max([Z.top_ekf.survival]);
    Q_spiral = Z.top_ekf(bestIdx).Q;
else
    % Fallback to your baseline Spiral Q (same structure as in your code)
    Q_scalar = 1.5;
    Q_base   = [0.7;0.5;0.1; 1e-7; 1e-10; 1e-7] * Q_scalar;
    Q_spiral = diag(Q_base);
    warning('Spiral leaderboard file not found; using baseline Q_spiral.');
end

%% 4) Load UKF Q (exact UKF config), with safe fallback to Q0
ukf_mat = 'CAUKF_5minRandom_RL_Tuned_10_06_v1.mat';
Q0_diag = 5 * [1e-7;1e-7;1e-6; 0.2;0.12;0.07; 5e-8;5e-7;3e-7];
Q0_ukf  = diag(Q0_diag);
if isfile(ukf_mat)
    U = load(ukf_mat,'top_ukf');
    if isfield(U,'top_ukf') && ~isempty(U.top_ukf)
        [~,bestU] = max([U.top_ukf.survival]);
        Q_UKF = U.top_ukf(bestU).Q;
    else
        Q_UKF = Q0_ukf;
        warning('UKF mat missing top_ukf content; using Q0.');
    end
else
    Q_UKF = Q0_ukf;
    warning('UKF leaderboard file not found; using Q0.');
end

%% 5) Trajectory & Measurements (exact generator)
truth = final_traj_sim_5min_random();          % Nx3
meas  = truth + radarSTD * randn(size(truth)); % noisy radar-like

N = size(truth,1);
idx = max(SKIP,0)+1 : N;
t   = (0:N-1)*T;

%% 6) CA-EKF run (exact update)
X_e = zeros(9,1); P_e = eye(9);
est_e = zeros(N,3); err_e = zeros(N,3); pole_e = zeros(N,3);
for k=1:N
    z  = meas(k,:)';
    Xp = F_e*X_e; Pp = F_e*P_e*F_e' + Q_EKF;
    innov = z - H*Xp;
    Syy   = H*Pp*H' + R_EKF;
    K     = Pp*H'/Syy;
    X_e   = Xp + K*innov;
    P_e   = (eye(9)-K*H)*Pp;
    est_e(k,:)  = X_e(1:3)';
    err_e(k,:)  = truth(k,:) - est_e(k,:);
    pole_e(k,:) = (1 - diag(K(1:3,1:3)))';
end

%% 7) Spiral-EKF run (exact spiral_predict_update)
X_s = zeros(6,1); P_s = eye(6);
est_s = zeros(N,3); err_s = zeros(N,3); pole_s = zeros(N,3);
for k=1:N
    z = meas(k,:)';
    [Xp2,Pp2,innov2,~,pole2] = spiral_predict_update(X_s,P_s,z,Q_spiral,R_true,T);
    X_s = Xp2; P_s = Pp2;
    est_s(k,:)  = Xp2(1:3)';
    err_s(k,:)  = truth(k,:) - est_s(k,:);
    pole_s(k,:) = pole2';
end

%% 8) IMM (CA + Spiral) with exact gating/likelihood logic
mu  = [0.5;0.5];
X1  = zeros(9,1); P1 = eye(9);   % CA
X2  = zeros(6,1); P2 = eye(6);   % Spiral
est_i = zeros(N,3); err_i = zeros(N,3); pole_i = zeros(N,3);

for k=1:N
    z = meas(k,:)';

    % CA part
    X1p = F_e*X1; P1p = F_e*P1*F_e' + Q_EKF;
    innov1 = z - H*X1p;
    S1 = H*P1p*H' + R_EKF;
    K1 = P1p*H'/S1;
    X1 = X1p + K1*innov1; P1 = (eye(9)-K1*H)*P1p;
    pole_ca = 1 - diag(K1(1:3,1:3));
    x_ca    = X1(1:3);

    % Spiral part
    [Xp2,Pp2,innov2,~,pole2] = spiral_predict_update(X2,P2,z,Q_spiral,R_true,T);
    S2 = Hs*Pp2*Hs' + R_true;
    S2 = (S2 + S2')/2;                     % enforce symmetry
    [~,p] = chol(S2);
    if p>0, S2 = S2 + 1e-6*eye(3); end     % regularize

    % Likelihoods (mvnpdf fallback-safe)
    L1 = normal_like(innov1, S1);
    L2 = normal_like(innov2, S2);

    % IMM mixing
    mu = (TPM' * mu) .* [L1; L2];
    mu = mu / sum(mu);

    % Fused pole + state with your same gating
    pole_imm = mu(1)*pole_ca + mu(2)*pole2;
    x_imm    = mu(1)*x_ca    + mu(2)*Xp2(1:3);
    if any(pole_imm > pole_ca & (pole_imm - pole_ca) > pole_ca)
        x_imm = x_ca;  % same safety gate
    end

    est_i(k,:) = x_imm';
    err_i(k,:) = truth(k,:) - est_i(k,:);
    pole_i(k,:)= pole_imm';

    % carry spiral states
    X2 = Xp2; P2 = Pp2;
end

%% 9) CA-UKF run (exact as in your function)
[rm_ax_ukf,var_ax_ukf,~,~,~,X_ukf,err_ukf] = evalCandidate_UKF(...
    truth, meas, Q_UKF, R_true, T, true, SKIP);

%% 10) Post-transient stats (mean RMSE & Var) — ALL filters
[rm_e,var_e] = stats_post(err_e, idx);
[rm_s,var_s] = stats_post(err_s, idx);
[rm_i,var_i] = stats_post(err_i, idx);

fprintf('\n=== Post-transient stats (SKIP=%d) ===\n', SKIP);
prt_stats('CA-KF ', rm_e, var_e);
prt_stats('Spiral ', rm_s, var_s);
prt_stats('IMM    ', rm_i, var_i);
prt_stats('CA-UKF ', rm_ax_ukf, var_ax_ukf);

%% 11) Figures — 3D tracks
figure('Name','3D - CA-EKF','Color','w'); grid on; axis equal;
plot3(truth(:,1),truth(:,2),truth(:,3),'k-','LineWidth',1.4); hold on;
plot3(est_e(:,1),est_e(:,2),est_e(:,3),'b--','LineWidth',1.2);
legend('Truth','CA-EKF'); xlabel('X'); ylabel('Y'); zlabel('Z');
title('3D Tracking — CA-KF');

figure('Name','3D - Spiral-EKF','Color','w'); grid on; axis equal;
plot3(truth(:,1),truth(:,2),truth(:,3),'k-','LineWidth',1.4); hold on;
plot3(est_s(:,1),est_s(:,2),est_s(:,3),'g--','LineWidth',1.2);
legend('Truth','Spiral-EKF'); xlabel('X'); ylabel('Y'); zlabel('Z');
title('3D Tracking — Spiral-EKF');

figure('Name','3D - IMM (CA+Spiral)','Color','w'); grid on; axis equal;
plot3(truth(:,1),truth(:,2),truth(:,3),'k-','LineWidth',1.4); hold on;
plot3(est_i(:,1),est_i(:,2),est_i(:,3),'r--','LineWidth',1.2);
legend('Truth','IMM'); xlabel('X'); ylabel('Y'); zlabel('Z');
title('3D Tracking — IMM');

%% 12) Axis tracking (with noisy measurements) — each filter
labels = {'X','Y','Z'};

figure('Name','Axis Tracking — Spiral-EKF (with meas)','Color','w');
for a=1:3
    subplot(3,1,a); hold on; grid on;
    plot(truth(:,a),'k','LineWidth',1.2);
    plot(meas(:,a),':','Color',[0.5 0.5 0.5]);
    plot(est_s(:,a),'g--','LineWidth',1.2);
    ylabel(labels{a});
    if a==1, legend('Truth','Meas','Spiral'); end
    if a==3, xlabel('Time Step'); end
end
sgtitle('Axis Tracking — Spiral-EKF');

figure('Name','Axis Tracking — CA-EKF (with meas)','Color','w');
for a=1:3
    subplot(3,1,a); hold on; grid on;
    plot(truth(:,a),'k','LineWidth',1.2);
    plot(meas(:,a),':','Color',[0.5 0.5 0.5]);
    plot(est_e(:,a),'b--','LineWidth',1.2);
    ylabel(labels{a});
    if a==1, legend('Truth','Meas','CA-EKF'); end
    if a==3, xlabel('Time Step'); end
end
sgtitle('Axis Tracking — CA-EKF');

figure('Name','Axis Tracking — IMM (with meas)','Color','w');
for a=1:3
    subplot(3,1,a); hold on; grid on;
    plot(truth(:,a),'k','LineWidth',1.2);
    plot(meas(:,a),':','Color',[0.5 0.5 0.5]);
    plot(est_i(:,a),'r--','LineWidth',1.2);
    ylabel(labels{a});
    if a==1, legend('Truth','Meas','IMM'); end
    if a==3, xlabel('Time Step'); end
end
sgtitle('Axis Tracking — IMM');

%% 13) Error vs Time — each filter
figure('Name','Error vs Time — CA-EKF','Color','w');
for a=1:3
    subplot(3,1,a); plot(t, err_e(:,a),'b'); grid on;
    ylabel(sprintf('Err %s (m)',labels{a}));
end; xlabel('Time (s)'); sgtitle('CA-KF — Error per Axis');

figure('Name','Error vs Time — Spiral-EKF','Color','w');
for a=1:3
    subplot(3,1,a); plot(t, err_s(:,a),'g'); grid on;
    ylabel(sprintf('Err %s (m)',labels{a}));
end; xlabel('Time (s)'); sgtitle('Spiral-EKF — Error per Axis');

figure('Name','Error vs Time — IMM','Color','w');
for a=1:3
    subplot(3,1,a); plot(t, err_i(:,a),'r'); grid on;
    ylabel(sprintf('Err %s (m)',labels{a}));
end; xlabel('Time (s)'); sgtitle('IMM — Error per Axis');

%% 14) CA-UKF figures (exact UKF config)
figure('Name','3D - CA-UKF','Color','w'); grid on; axis equal;
plot3(truth(:,1),truth(:,2),truth(:,3),'k-','LineWidth',1.4); hold on;
plot3(X_ukf(:,1),X_ukf(:,2),X_ukf(:,3),'m--','LineWidth',1.2);
legend('Truth','CA-UKF'); xlabel('X'); ylabel('Y'); zlabel('Z');
title('3D Tracking — CA-UKF');

figure('Name','Axis Tracking — CA-UKF (with meas)','Color','w');
for a=1:3
    subplot(3,1,a); hold on; grid on;
    plot(truth(:,a),'k','LineWidth',1.2);
    plot(meas(:,a),':','Color',[0.5 0.5 0.5]);
    plot(X_ukf(:,a),'m--','LineWidth',1.2);
    ylabel(labels{a});
    if a==1, legend('Truth','Meas','CA-UKF'); end
    if a==3, xlabel('Step'); end
end
sgtitle('Axis Tracking — CA-UKF');

figure('Name','Error vs Time — CA-UKF','Color','w');
for a=1:3
    subplot(3,1,a); plot(t, err_ukf(:,a),'m'); grid on;
    ylabel(sprintf('Err %s (m)',labels{a}));
end; xlabel('Time (s)'); sgtitle('CA-UKF — Error per Axis');

%% 15) Optional: compare post-skip RMSE across all filters (bar plot)
figure('Name','Post-SKIP RMSE per Axis (All Filters)','Color','w');
RMSE_all = [rm_e(:), rm_s(:), rm_i(:), rm_ax_ukf(:)]; % [X;Y;Z] x 4
bar(RMSE_all);
set(gca,'XTickLabel',labels);
legend('CA-EKF','Spiral-EKF','IMM','CA-UKF','Location','northwest');
ylabel('RMSE (m)'); title(sprintf('Post-SKIP RMSE (SKIP=%d)', SKIP));
grid on;

%% ========================= Helper Functions =========================

function [rm_ax, var_ax] = stats_post(err_mat, idx)
    e = err_mat(idx,:);
    rm_ax  = sqrt(mean(e.^2,1));
    var_ax = var(e,0,1);
end

function prt_stats(name, rm, vr)
    fprintf('%s  RMSE = [%.3f %.3f %.3f]   Var = [%.3f %.3f %.3f]\n', ...
        name, rm(1),rm(2),rm(3), vr(1),vr(2),vr(3));
end

function L = normal_like(innov, S)
    % Safe Gaussian likelihood using Cholesky; mirrors mvnpdf without toolbox
    % (Used also to avoid numerical issues)
    S = (S+S')/2;
    [U,p] = chol(S,'lower');
    if p>0
        S = S + 1e-6*eye(size(S));
        U = chol(S,'lower');
    end
    y = U \ innov;
    quad = sum(y.^2);
    logdet = 2*sum(log(diag(U)));
    k = length(innov);
    L = exp(-0.5*(k*log(2*pi) + logdet + quad));
end

% ---- Spiral predict/update (exact structure you used) ----
function [Xp,Pp,innov,K,pole] = spiral_predict_update(X,P,z,Q,R,T)
    th=X(4); v=X(5); w=X(6); eps_w=1e-3;
    if abs(w)<eps_w
        Xp=[X(1)+v*cos(th)*T; X(2)+v*sin(th)*T; X(3)+v*T; th; v; w];
        F=[1 0 0 -v*sin(th)*T cos(th)*T 0;
           0 1 0  v*cos(th)*T sin(th)*T 0;
           0 0 1  0           T         0;
           0 0 0  1           0         T;
           0 0 0  0           1         0;
           0 0 0  0           0         1];
    else
        dth=th+w*T; c1=cos(dth)-cos(th); s1=sin(dth)-sin(th);
        Xp=[X(1)+v/w*c1; X(2)+v/w*s1; X(3)+v*T; dth; v; w];
        F=eye(6);
        F(1,4)= v/w*s1; F(1,5)= c1/w; F(1,6)= v/w^2*(w*T*sin(dth)-s1);
        F(2,4)=-v/w*c1; F(2,5)= s1/w; F(2,6)= v/w^2*(-w*T*cos(dth)-c1);
        F(3,5)=T; F(4,6)=T;
    end
    Pp=F*P*F'+Q;
    Hs=[eye(3),zeros(3,3)];
    innov=z-Hs*Xp;
    S=Hs*Pp*Hs'+R; S=(S+S')/2;
    [~,p] = chol(S);
    if p>0, S = S + 1e-6*eye(3); end
    K=Pp*Hs'/S;
    Xp=Xp+K*innov; Pp=(eye(size(Pp))-K*Hs)*Pp;
    pole=1-diag(K(1:3,1:3));
end

% ---- CA-UKF (exact weights & flow) ----
function [rm_ax,var_ax,score,rm_prof,innov_prof,X_est,err_mat] = evalCandidate_UKF(...
    truth, meas, Q, R, dt, keep, skip)
    if nargin<6, keep=false; end
    if nargin<7, skip=0; end
    N = size(truth,1);
    n = 9;
    X = zeros(n,1); P = eye(n);

    % UKF weights (exact)
    alpha = 1e-1; beta = 2; kappa = 0;
    lambda = alpha^2*(n+kappa)-n;
    c = n+lambda;
    Wm = [lambda/c; repmat(1/(2*c),2*n,1)];
    Wc = Wm; Wc(1) = Wc(1)+(1-alpha^2+beta);

    % Model matrices (exact)
    F = [1 0 0 dt 0 0 0.5*dt^2 0 0;
         0 1 0 0 dt 0 0 0.5*dt^2 0;
         0 0 1 0 0 dt 0 0 0.5*dt^2;
         0 0 0 1 0 0 dt 0 0;
         0 0 0 0 1 0 0 dt 0;
         0 0 0 0 0 1 0 0 dt;
         0 0 0 0 0 0 1 0 0;
         0 0 0 0 0 0 0 1 0;
         0 0 0 0 0 0 0 0 1];
    H = [eye(3), zeros(3,6)];

    err_mat = zeros(N,3);
    innov_prof = zeros(N,1);
    if keep, X_est = zeros(N,3); else, X_est = []; end

    for k=1:N
        % Sigma points
        Ps = chol(P,'lower');
        Xsig = [X, repmat(X,1,n) + sqrt(c)*Ps, repmat(X,1,n) - sqrt(c)*Ps];

        % Predict
        Xsig_p = F * Xsig;
        Xp = Xsig_p * Wm;
        Pp = Q;
        for i=1:2*n+1
            d = Xsig_p(:,i) - Xp;
            Pp = Pp + Wc(i)*(d*d');
        end

        % Predict measurement
        Ysig = H * Xsig_p;
        Yp = Ysig * Wm;
        Pyy = R; Pxy = zeros(n,3);
        for i=1:2*n+1
            dy = Ysig(:,i) - Yp;
            dx = Xsig_p(:,i) - Xp;
            Pyy = Pyy + Wc(i)*(dy*dy');
            Pxy = Pxy + Wc(i)*(dx*dy');
        end

        % Update
        z = meas(k,:)';
        innov = z - Yp;
        K = Pxy / Pyy;
        X = Xp + K*innov;
        P = Pp - K*Pyy*K';

        % Log
        if keep, X_est(k,:) = X(1:3)'; end
        err_mat(k,:)   = truth(k,:) - X(1:3)';
        innov_prof(k)  = norm(innov);
    end

    % Scores
    if skip>0 && skip<N, e = err_mat(skip+1:end,:); else, e = err_mat; end
    rm_ax  = sqrt(mean(e.^2,1));
    var_ax = var(e,0,1);
    score  = mean(rm_ax);
    rm_prof = vecnorm(err_mat,2,2);
end

% ---- Trajectory generator (same physics) ----
function traj = final_traj_sim_5min_random
    rng('shuffle'); dt=0.01; dur=300; N=dur/dt;
    box=[-1200 1200; -1200 1200; 40 700];
    t_wp=sort([0 rand(1,randi([10 20]))*dur dur]); wp=zeros(numel(t_wp),3);
    for i=1:3, wp(:,i)=box(i,1)+diff(box(i,:))*rand(size(t_wp)); end
    t=linspace(0,dur,N)'; ref=zeros(N,3);
    for i=1:3, ref(:,i)=interp1(t_wp,wp(:,i),t,'pchip'); end
    v_max=25; a_max=8; j_max=50;
    vel=zeros(N,3); acc=zeros(N,3); traj=zeros(N,3); traj(1,:)=ref(1,:);
    for k=2:N
        v_des=(ref(k,:)-ref(k-1,:))/dt;
        if norm(v_des)>v_max, v_des=v_des*v_max/norm(v_des); end
        a_des=(v_des-vel(k-1,:))/dt;
        if norm(a_des)>a_max, a_des=a_des*a_max/norm(a_des); end
        jerk=(a_des-acc(k-1,:))/dt;
        if norm(jerk)>j_max, jerk=jerk*j_max/norm(jerk); a_des=acc(k-1,:)+jerk*dt; end
        acc(k,:)=a_des; vel(k,:)=vel(k-1,:)+a_des*dt; traj(k,:)=traj(k-1,:)+vel(k,:)*dt;
    end
    for i=1:3, traj(:,i)=min(max(traj(:,i),box(i,1)),box(i,2)); end
end

