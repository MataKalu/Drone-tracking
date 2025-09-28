%% RL_SpiralEKF_5minRandom_RL_Tuned_v2.m
% ================================================================
% RL-Based Q-Tuning for 6-state Spiral-Motion EKF (improved exploration)
% Replacement uses current-trial performance of saved candidates
% ================================================================
clear; clc; close all;

%% 1) Parameters
T        = 0.01;            % sample period [s]
Trials   = 1;             % RL episodes
N_TOP    = 3;               % leaderboard size
SKIP     = 3000;            % samples to skip when scoring

% ε-greedy exploration parameters
eps0      = 0.8; epsMin = 0.3; decayE = 0.75;
pert0     = 0.07;
searchFac = 1.222; probTemp = 0.7;
rangeVar  = 0.3;
decay_F   = 0.94;
varDecay  = 0.9;
score_t_h=60;

% noise & initial Q
radarSTD  = 50;
R_true    = radarSTD^2 * eye(3);
Q0_diag   = 1 * ones(6,1);
Q0        = diag(Q0_diag);

%% 2) Leaderboard initialization
matFile = 'SpiralEKF_5minRandom_RL_Tuned.mat';
if isfile(matFile)
    load(matFile, 'top_ekf');
    if numel(top_ekf)~=N_TOP
        top_ekf = initTop(N_TOP, Q0);
    end
    disp('Loaded existing leaderboard.');
else
    top_ekf = initTop(N_TOP, Q0);
    disp('Created new leaderboard.');
end

%% 3) RL loop with current-trial replacement
epsCur = eps0;
for tr = 1:Trials
    % simulate trajectory + measurements
    truth = final_traj_sim_5min_random();
    meas  = truth + radarSTD*randn(size(truth));

    % select exploration mode
    r = rand;
    if tr <= N_TOP || r < epsCur
        mode = 'global';
    elseif r < epsCur + probTemp
        mode = 'temp';
    else
        mode = 'local';
    end

    % propose new Q candidate
    switch mode
        case 'global'
            sf    = searchFac * (1 + rangeVar * (rand(6,1)-0.5));
            logsf = log10(sf);
            qdiag = Q0_diag .* 10.^((2*rand(6,1)-1) .* logsf);
        case 'temp'
            [~,b] = min([top_ekf.score]);
            qref  = diag(top_ekf(b).Q);
            qdiag = qref .* (1 + pert0*(rand(6,1)-0.5));
        case 'local'
            idx   = randi(N_TOP);
            qref  = diag(top_ekf(idx).Q);
            qdiag = qref .* (1 + pert0*(rand(6,1)-0.5));
    end
    qdiag = max(qdiag, 1e-12);
    Qcand = diag(qdiag);

    % decay exploration parameters
    epsCur    = max(epsMin, epsCur * decayE);
    rangeVar  = max(rangeVar * varDecay, 0.1);
    searchFac = max(searchFac * decay_F, 0.5);
    saved = false;                 % <-- add this

    % evaluate new candidate
    [~,~,score_c] = runSpiralEKF_full(truth, meas, Qcand, R_true, T, false, SKIP);

    % compute current-trial scores for saved candidates
    scores_s = -inf(1, N_TOP);
    for s = 1:N_TOP
        if ~isinf(top_ekf(s).score)
            [~,~,scores_s(s)] = runSpiralEKF_full(truth, meas, top_ekf(s).Q, R_true, T, false, SKIP);
        end
    end

    % update survival counts and count wins
    wins = 0;
    for s = 1:N_TOP
        if scores_s(s) < score_c
            top_ekf(s).survival = top_ekf(s).survival + 1;
        else
            top_ekf(s).survival = max(0, top_ekf(s).survival - 1);
            wins = wins + 1;
        end
    end

    % determine replacement slot: empty first, else worst current-trial performer
    emptySlot = find(isinf([top_ekf.score]), 1);
    if ~isempty(emptySlot)
        slot = emptySlot;
    else
        [~, slot] = max(scores_s);
    end

    if ~isempty(emptySlot)
        top_ekf(emptySlot) = struct('Q',Qcand,'score',score_c,'survival',0);
        slot  = emptySlot;         % keep the right index
        saved = true;              % <-- mark that we actually saved
    elseif wins >= 2 && score_c < scores_s(slot) && score_c < score_t_h
        top_ekf(slot) = struct('Q',Qcand,'score',score_c,'survival',0);
        saved = true;              % <-- mark save
    end

    if saved
        fprintf('Trial %3d | score = %.3f | wins = %d | slot replaced = %d\n', ...
                tr, score_c, wins, slot);
    else
        fprintf('Trial %3d | score = %.3f | wins = %d | **no replacement**\n', ...
                tr, score_c, wins);
    end
end

save(matFile, 'top_ekf');
fprintf('Leaderboard saved -> %s\n', matFile);

%% 4) Post-Plots & Diagnostics (Best vs Candidate)
Q_best = top_ekf(argmax([top_ekf.survival])).Q;
[rm_b,~,~,prof_b,innov_b,X_b,P_b,err_b] = ...
    runSpiralEKF_full(truth, meas, Q_best, R_true, T, true, 0);

labels = {'X','Y','Z'};
% Axis Tracking
figure('Name','Axis Tracking');
for d = 1:3
    subplot(3,1,d);
    plot(truth(:,d),'k','LineWidth',1.2); hold on;
    plot(X_b(:,d),'b--','LineWidth',1.2);
    grid on; ylabel(labels{d});
    if d==1, legend('Truth','Best'); end
end
xlabel('Step'); sgtitle('Tracking with Best Q');

% 3-D Path
figure('Name','3-D Path');
plot3(truth(:,1),truth(:,2),truth(:,3),'k','LineWidth',1.2); hold on;
plot3(X_b(:,1),X_b(:,2),X_b(:,3),'b--','LineWidth',1.2);
axis equal; grid on; legend('Truth','Best');
xlabel('X'); ylabel('Y'); zlabel('Z');

% RMSE per Axis
figure('Name','RMSE per Axis');
bar(rm_b); set(gca,'XTickLabel',labels);
ylabel('RMSE (m)'); title('Per-Axis RMSE for Best Q');

% Innovation & RMSE-Norm
figure('Name','Innovation & RMSE-Norm');
subplot(2,1,1);
plot(innov_b,'b','LineWidth',1.2); grid on;
ylabel('|innovation|'); title('Innovation Norm');
subplot(2,1,2);
plot(prof_b,'r','LineWidth',1.2); grid on;
ylabel('||error||'); xlabel('Step'); title('Error Norm');

% Per-Axis Diagnostics
figure('Name','Per-Axis Diagnostics','Color','w');
for ax = 1:3
    subplot(3,3,(ax-1)*3+1);
    plot(squeeze(sqrt(P_b(ax,ax,:))),'b','LineWidth',1.2); grid on;
    ylabel(['\sigma_' labels{ax}]); if ax==1, title('1\sigma'); end
    subplot(3,3,(ax-1)*3+2);
    plot(abs(err_b(:,ax)),'r','LineWidth',1.2); grid on;
    ylabel(['|e_' labels{ax} '|']); if ax==1, title('|Error|'); end
    subplot(3,3,(ax-1)*3+3);
    plot(truth(:,ax),'k','LineWidth',1.2); hold on;
    plot(X_b(:,ax),'b--','LineWidth',1.2); grid on;
    ylabel(labels{ax}); if ax==1, title('Tracking'); legend('Truth','Best'); end
end
xlabel('Step'); sgtitle('Per-Axis Diagnostics for Best Q');

%% 5) Re-evaluate Best Q on 3 New Trajectories + Diagnostics
for tIdx = 1:3
    fprintf('\n=== Re-evaluation on Trajectory %d ===\n', tIdx);
    truth = final_traj_sim_5min_random();
    meas  = truth + radarSTD * randn(size(truth));
    [rm_ax,var_ax,~,prof,innov,X_est,P_tr,err] = ...
        runSpiralEKF_full(truth, meas, Q_best, R_true, T, true, SKIP);

    fprintf('RMSE per axis:     [%.2f %.2f %.2f] m\n', rm_ax);
    fprintf('Variance per axis: [%.2f %.2f %.2f] m^2\n', var_ax);

    figure('Name',sprintf('Tracking – Traj %d',tIdx));
    for d = 1:3
        subplot(3,1,d);
        plot(truth(:,d),'k','LineWidth',1.2); hold on;
        plot(X_est(:,d),'b--','LineWidth',1.2); grid on;
        ylabel(labels{d});
        if d==1, legend('Truth','Estimate'); end
    end
    xlabel('Step');

    figure('Name',sprintf('Innovation & Error Norm – Traj %d',tIdx));
    subplot(2,1,1);
    plot(innov,'r','LineWidth',1.2); grid on; ylabel('|innovation|');
    subplot(2,1,2);
    plot(prof,'b','LineWidth',1.2); grid on; ylabel('||error||'); xlabel('Step');
end

%% === Helper Functions ===
function idx = argmax(v), [~,idx] = max(v); end

function top = initTop(N, Q0)
    top = repmat(struct('Q',Q0,'score',inf,'survival',0),1,N);
end

function [rm_ax,var_ax,score,rm_prof,innov_prof,X_est,P_tr,err_hist] = runSpiralEKF_full(truth,meas,Q,R,dt,keep,skip)
    if nargin<6, keep=false; end
    if nargin<7, skip=0; end
    N = size(truth,1);
    X = zeros(6,1); P = eye(6); X_prev = X;
    err_hist = zeros(N,3); rm_prof = zeros(N,1); innov_prof = zeros(N,1);
    if keep, X_est = zeros(N,3); P_tr = zeros(6,6,N); else X_est=[]; P_tr=[]; end
    H = [eye(3) zeros(3,3)]; eps_w = 1e-3;
    for k = 1:N
        th=X(4); v=X(5); w=X(6); dth=abs(th-X_prev(4)); isStraight=(dth<1e-3);
        if abs(w)<eps_w||isStraight
            Xp=[X(1)+v*cos(th)*dt; X(2)+v*sin(th)*dt; X(3)+v*dt; th; v; w];
            F=[1 0 0 -v*sin(th)*dt cos(th)*dt 0; 0 1 0 v*cos(th)*dt sin(th)*dt 0; 0 0 1 0 dt 0; 0 0 0 1 0 dt; 0 0 0 0 1 0; 0 0 0 0 0 1];
        else
            dth2=th+w*dt; c1=cos(dth2)-cos(th); s1=sin(dth2)-sin(th);
            Xp=[X(1)+v/w*c1; X(2)+v/w*s1; X(3)+v*dt; dth2; v; w];
            F=eye(6); F(1,4)=v/w*s1;F(1,5)=c1/w;F(1,6)=v/w^2*(w*dt*sin(dth2)-s1);
            F(2,4)=-v/w*c1;F(2,5)=s1/w;F(2,6)=v/w^2*(-w*dt*cos(dth2)-c1); F(3,5)=dt; F(4,6)=dt;
        end
        Pp=F*P*F'+Q; z=meas(k,:)'; innov=z-H*Xp; S=H*Pp*H'+R; K=Pp*H'/S;
        X=Xp+K*innov; P=(eye(6)-K*H)*Pp;
        err_hist(k,:) = truth(k,:) - X(1:3)'; rm_prof(k)=norm(err_hist(k,:)); innov_prof(k)=norm(innov);
        if keep, X_est(k,:)=X(1:3)'; P_tr(:,:,k)=P; end
        X_prev = X;
    end
    if skip>0&&skip<N, e=err_hist(skip+1:end,:); else e=err_hist; end
    rm_ax = sqrt(mean(e.^2,1)); var_ax=var(e,0,1);
    abs_sum = sum(abs(e),1); score = 0.4*sum(rm_ax)+0.3*mean(var_ax);
    if ~keep, P_tr=[]; err_hist=[]; end
end

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
