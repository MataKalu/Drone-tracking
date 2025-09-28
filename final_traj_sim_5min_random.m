function traj = final_traj_sim_5min_random()
% final_traj_sim_5min_random – 5‑min randomized drone trajectory
% • Duration: 5 min at 100 Hz
% • Domain: X,Y ∈ [–2500,2500] m; Z ∈ [40,700] m
% • Drone limits: v_max=25 m/s, a_max=8 m/s², j_max=50 m/s³
% • Smooth, continuous motion with jerk limiting

rng('shuffle');
%% 1) PARAMETERS
dt            = 0.01;                % [s]
duration_min  = 5;                   % minutes
duration_s    = duration_min*60;    % seconds
N             = round(duration_s/dt);

% Spatial bounds
minX = -1200; maxX =  1200;
minY = -1200 ; maxY =  1200;
minZ =   40;   maxZ =   700;

% Physical limits
dt    = dt;
v_max = 25;   % [m/s]
a_max = 8;    % [m/s^2]
j_max = 50;   % [m/s^3]

%% 2) RANDOM WAYPOINTS\% Choose random number of waypoints (5–15)
numWP = randi([10,20]);
% Include start=0 and end=duration
t_wp = sort([0, rand(1,numWP)*duration_s, duration_s]);
x_wp = minX + (maxX-minX)*rand(size(t_wp));
y_wp = minY + (maxY-minY)*rand(size(t_wp));
z_wp = minZ + (maxZ-minZ)*rand(size(t_wp));

% Spline path through waypoints
t = linspace(0,duration_s,N)';
raw = [interp1(t_wp,x_wp,t,'pchip'), ...
       interp1(t_wp,y_wp,t,'pchip'), ...
       interp1(t_wp,z_wp,t,'pchip')];

%% 3) SMOOTH VELOCITY & ACCELERATION WITH JERK LIMITING
vel = zeros(N,3);
acc = zeros(N,3);
for k = 2:N
    % desired velocity from raw spline
    v_des = (raw(k,:) - raw(k-1,:)) / dt;
    % clip speed
    if norm(v_des) > v_max
        v_des = v_des * (v_max / norm(v_des));
    end
    % acceleration
    a_des = (v_des - vel(k-1,:)) / dt;
    if norm(a_des) > a_max
        a_des = a_des * (a_max / norm(a_des));
    end
    % jerk limiting relative to previous acc
    j = (a_des - acc(k-1,:)) / dt;
    if norm(j) > j_max
        j = j * (j_max / norm(j));
        a_des = acc(k-1,:) + j*dt;
    end
    % update states
    acc(k,:) = a_des;
    vel(k,:) = vel(k-1,:) + a_des*dt;
end

%% 4) RECOMPUTE POSITION BY INTEGRATION
traj = zeros(N,3);
traj(1,:) = raw(1,:);
for k = 2:N
    traj(k,:) = traj(k-1,:) + vel(k,:) * dt;
end

%% 5) CLAMP TO DOMAIN
traj(:,1) = min(max(traj(:,1),minX),maxX);
traj(:,2) = min(max(traj(:,2),minY),maxY);
traj(:,3) = min(max(traj(:,3),minZ),maxZ);

% 6) METRICS & DISPLAY
v_mag = vecnorm(vel,2,2);
a_mag = vecnorm(acc,2,2);
avg_v = mean(v_mag);
avg_a = mean(a_mag);
fprintf('Duration: %.1f s (%.1f min)\n',duration_s,duration_min);
fprintf('Avg velocity: %.2f m/s   Avg acceleration: %.2f m/s^2\n',avg_v,avg_a);

% % %% 7) PLOTTING
t_vec = (0:N-1)' * dt;
% % 3D Path
figure('Name','3D Trajectory','Color','w');
plot3(traj(:,1),traj(:,2),traj(:,3),'LineWidth',1.5);
grid on; axis equal;
xlabel('X (m)'); ylabel('Y (m)'); zlabel('Z (m)');
title('Randomized Smooth Trajectory');
% 
% Position vs time
figure('Name','Position vs Time','Color','w');
subplot(3,1,1); plot(t_vec,traj(:,1)); ylabel('X (m)'); grid on;
subplot(3,1,2); plot(t_vec,traj(:,2)); ylabel('Y (m)'); grid on;
subplot(3,1,3); plot(t_vec,traj(:,3)); ylabel('Z (m)'); xlabel('Time (s)'); grid on;

% Velocity & acceleration
figure('Name','Velocity & Acceleration','Color','w');
subplot(2,1,1); plot(t_vec,v_mag,'LineWidth',1);
grid on; ylabel('Velocity (m/s)'); title('Velocity Profile');
subplot(2,1,2); plot(t_vec,a_mag,'LineWidth',1);
grid on; ylabel('Acceleration (m/s^2)'); xlabel('Time (s)');
title('Acceleration Profile');
end
