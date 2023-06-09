import numpy as np 
import seaborn as sn
from matplotlib import pyplot as plt
import tensorflow as tf  
from tensorflow.keras.layers import Input, Dense, Conv2D, Flatten, MaxPooling2D, AveragePooling2D, Concatenate

rng = np.random.default_rng()
mse = tf.keras.losses.MeanSquaredError()

print('tensorflow version:', tf.__version__)
print('keras version:', tf.keras.__version__)


def cal_direction(x0, y0, x1, y1):
    '''
    Calculate the direction vector based on the current position (x0,y0) and the target position (x1,y1).
    The next position is (x0 + dx*speed_max*slot_len, y0 + dy*speed_max*slot_len)
    Parameters: 
        (x0, y0) : the current position 
        (x1, y1) : the target position
    Returns: 
        (dx, dy) : the direction vector, dx and dy are in range (-1,1) 
    ''' 
    delta_x = x1 - x0                               # horizontal distance difference
    delta_y = y1 - y0                               # vertical distance difference
    delta_d = np.sqrt(delta_x**2 + delta_y**2)      # distance difference 
    dx, dy = (delta_x/delta_d, delta_y/delta_d) if delta_d>0 else (0,0) 
    return dx, dy, delta_d


def update_UAV_location(x0,y0, dx, dy, speed=5, slot_len=1): 
    '''Update UAV location in the horizontal plane 
    Parameters:
        - (x0,y0) : current location 
        - (dx,dy) : moving direction, both in range (-1,1) 
        - speed : moving speed, in m/s 
        - slot_len : slot length in second 
    Returns: 
        - (x1,y1) : location in the next time slot  
    '''
    x1 = x0 + dx*speed*slot_len
    y1 = y0 + dy*speed*slot_len
    return (x1,y1)


def update_UAV_altitude(z0, dz, vz, slot_len=1): 
    return z0 + dz*vz*slot_len


class UAV_Movement_Controller():
    def __init__(self, boundary=250, grid_size=20, uav_speedxy_max=10, n_decisions=10, lr=1e-4, n_uavs=4, zmin=50, zmax=150, uav_speedz_max=5):
        self.boundary = boundary        # the boundary of the UAV-covered region, in meters
        self.grid_size = grid_size      # resolution of the heatmap, e.g., grid_size = 10 -> 10x10(m) grids
        self.n_grids = int((2*boundary)/grid_size)      # no. of grids in each vertical and horizontal axis
        self.uav_speedxy_max = uav_speedxy_max          # in m/s
        self.uav_speedz_max = uav_speedz_max            # in m/s 
        self.n_decisions = n_decisions                  # no. of generated decisions output by the actor module
        self.std_var_vxvyvz = 0.5 
        self.n_uavs = n_uavs 
        self.zmin = zmin
        self.zmax= zmax
                             
        ### The DNN model  
        self.n_channels = 3              # no. of user statistics passed to the DNN (queue, traffic, ch_capacity)
        self.n_outputs = 3               # output of the DNN model: dx, dy, dv 
        self.learning_rate = 1e-3                   # learning rate 
        self.model = self.build_dnn()               # initiate the DNN model
        self.batch_size_manual = 1000               # select a batch of samples for training in the case of manual batching 
        self.training_interval = 20 
        self.n_epochs = 1                           # no. of epochs in one training interval 
        self.batchSize_ = 100                       # automatic batching: # of samples for one gradient update, until the whole dataset used 
        self.train_loss_history = []                # monitor the training loss 
        self.val_loss_history = []                  # monitor the validation loss 
        self.test_loss_history = []                 # monitor the test loss 
        
        ### The replay memory: heatmaps (users' statistics) -> uav movement control (dx, dy, dv)
        self.train_test_ratio = 0.8
        self.memory_size = 512         # maximum # of entries in the memory 
        self.val_memory_size = 128     # replay memory for validation 
        self.min_samples_for_training = 256    # no. of samples >= the threshold -> start training 
        self.replay_memory_train = {'uav-location': np.zeros(shape=(self.memory_size, 3)),
                              'user-heatmaps': np.zeros(shape=(self.memory_size, self.n_grids, self.n_grids, self.n_channels)),
                              'best-action': np.zeros(shape=(self.memory_size, self.n_outputs))}
        self.memory_counter_train = 0     # store how many entries have been recorded so far  
        self.replay_memory_val = {'uav-location': np.zeros(shape=(self.val_memory_size, 3)),
                              'user-heatmaps': np.zeros(shape=(self.val_memory_size, self.n_grids, self.n_grids, self.n_channels)),
                              'best-action': np.zeros(shape=(self.val_memory_size, self.n_outputs))}
        self.memory_counter_val = 0 
        
        ### Utility functions
        self.cal_channel_fading_gain = None 
        self.cal_channel_capacity_Mb = None 
        self.gen_heatmap = None 
        # self.uav_altitude = None 
        self.Vlya = None            # Lyapunov parameter (V)


    def preprocess_uav_location(self, uav_location): 
        '''
        Parameters: 
            uav_location: a tupple of (x_uav, y_uav, z_uav). x_uav, y_uav: -boundary to boundary. z_uav: zmin to zmax
        Returns: 
            uav_location_scale: type=np.ndarray, shape=(3,), x_scale, y_scale, z_scale in range (0,1)
        '''
        col_scale = (uav_location[0] + self.boundary)/(2*self.boundary)
        row_scale = (self.boundary - uav_location[1])/(2*self.boundary)
        alt_scale = uav_location[2]/self.zmax
        return np.array([row_scale, col_scale, alt_scale])
    

    def preprocess_user_statistics(self, user_locations, user_statistics, normalization_coeff):
        '''
        Generate heatmaps of all system statistics, including normalization
        Parameters:
            user_locations: current locations of users, a tupple of (x_locations, y_locations), x_locations.shape = (n_users,)
            user_statistics: all statistics of users, a tupple of (queue_Mb, traffic_Mb, ch_capacity_Mb)
            normalization_coeff: normalization coefficients, a tupple of (queue_coeff_Mb, traffic_coeff_Mb, ch_capacity_coeff_Mb), queue_coeff_Mb.shape=(n_users,)
        Returns:
            heatmap_all_statistics: a heatmap of all user statistics, as np.ndarray, shape=(n_grids, n_grids, n_channels)
        '''
        heatmap_all_statistics = np.zeros(shape=(self.n_grids, self.n_grids, len(user_statistics) ))
        for id, statistic in enumerate(user_statistics):
            heatmap_all_statistics[:,:,id] = self.gen_heatmap(x_locations=user_locations[0], 
                                                              y_locations=user_locations[1], 
                                                              val=statistic, 
                                                              norm_val=normalization_coeff[id])
        return heatmap_all_statistics
    
    
    def postprocess_decision(self, decisions):
        ''' 
        Post-process the actor module's outputs 
        Parameters: 
            decisions: potential decisions (v_x, v_y, v_z) 
                        v_x and v_y in range(-1,1), for the decision generated by the DNN: not sure that v_x**2 + v_y**2 <= 1 
                        type=np.ndarray, shape=(n_decisions, 2)      
        Returns: 
            result: ready-to-use movement decisions (dx, dy, vxy, dz, vz), type=np.ndarray, shape=(n_decisions, 5),
                    dx and dy in range (-1,1), dx**2 + dy**2 = 1, vxy in range (0,self.uav_speed_max) 
                    dz = -1 (go down) or 1 (go up), vz in range (0, self.uav_speedz_max)
        '''
        result = []         # a list of available-to-use decisions
        for decision_id in range(self.n_decisions):
            vx, vy, vz = decisions[decision_id]
            v_sqrt = np.sqrt(vx**2 + vy**2)       # in range (0, sqrt(2))
            dx = vx/v_sqrt                        
            dy = vy/v_sqrt                        
            vxy = v_sqrt*self.uav_speedxy_max if v_sqrt<=1 else self.uav_speedxy_max  # vxy: 0 to uav_speedxy_max
            dz = 1 if vz > 0 else -1 
            vz = np.abs(vz)*self.uav_speedz_max     # vz: 0 to uav_speedz_max
            result.append([dx,dy,vxy,dz,vz])
        
        return np.array(result)
    
    
    def build_dnn(self):
        '''Build the DNN model'''
        uav_location = Input(shape=(3,), name='uav_location')      # the current UAV location, x_uav and y_uav 
        
        user_stats_heatmaps = Input(shape=(self.n_grids, self.n_grids, self.n_channels), name='user_stat_heatmaps')  # heatmaps of the users' statistics
        x = tf.reshape(user_stats_heatmaps, [-1, self.n_grids, self.n_grids, self.n_channels])
        x = Conv2D(filters=32, kernel_size=3, strides=1, activation='relu')(x)
        x = AveragePooling2D(pool_size=3, strides=2)(x)
        x = Conv2D(filters=64, kernel_size=3, strides=1, activation='relu')(x)
        x = AveragePooling2D(pool_size=3, strides=2)(x)
        # x = Conv2D(filters=128, kernel_size=3, strides=1, activation='relu')(x)
        # x = AveragePooling2D(pool_size=3, strides=2)(x)        
        x = Flatten()(x)                           
        x = Concatenate()([uav_location, x])        
        # x = Dense(units=512, activation='relu')(x)
        # x = Concatenate()([uav_location, x])
        x = Dense(units=256, activation='relu')(x)
        x = Concatenate()([uav_location, x])
        x = Dense(units=128, activation='relu')(x)
        
        ### the dnn outputs (vx, vy), i.e., the x-axis and y-axis velocity, vx and vy in range (-1,1)
        outputs = Dense(units=3, activation='tanh', name='vx_n_vy')(x)      # in range [-1,1], not sure that vx**2 + v_y**2 <= 1
        
        
        model = tf.keras.Model(inputs   = [uav_location, user_stats_heatmaps], 
                               outputs  = outputs,
                               name     = 'uav_controller')
        
        model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate), 
                      loss      = tf.keras.losses.MeanSquaredError(),   
                      metrics   = ['mae'],           # options: 'acc'='accuracy', 'mse'='MeanSquaredError, 'mae'='MeanAbsoluteError'
                      )
        
        return model 
    
    
    def retrain_dnn(self):
        '''Retrain the dnn'''
        ### prepare data for training: sample a batch of entries from the replay memory
        # if self.memory_counter > self.memory_size:
        #     sample_ids = np.random.choice(self.memory_size, size=self.batch_size)
        # elif self.memory_counter <= self.memory_size and self.memory_counter > self.batch_size:
        #     sample_ids = np.random.choice(self.memory_counter, size=self.batch_size)
        # else: 
        #     sample_ids = np.arange(stop=self.memory_counter)
        
        ### make use of all data observed 
        train_sample_ids = np.arange(stop=np.minimum(self.memory_counter_train, self.memory_size))
        val_sample_ids = np.arange(stop=np.minimum(self.memory_counter_val, self.val_memory_size))
        
        X_train = [self.replay_memory_train['uav-location'][train_sample_ids], self.replay_memory_train['user-heatmaps'][train_sample_ids]]
        y_train = self.replay_memory_train['best-action'][train_sample_ids]
        X_val = [self.replay_memory_val['uav-location'][val_sample_ids], self.replay_memory_val['user-heatmaps'][val_sample_ids]]
        y_val = self.replay_memory_val['best-action'][val_sample_ids]
        
        # train the DNN
        hist = self.model.fit(x = X_train, y=y_train, 
                                 epochs = self.n_epochs,            # An epoch is an iteration over the entire x and y data provided
                                 batch_size = self.batchSize_,      # number of samples per gradient update, default=32
                                 shuffle = True,                    # shuffle the dataset in batch-sized chunks
                                 verbose=2,                         # 0 = silent, 1 = progress bar, 2 = one line per epoch
                                 validation_data = (X_val, y_val)   # Data on which to evaluate the loss and any model metrics at the end of each epoch. The model will not be trained on this data.
                                 )                         
        train_loss = hist.history['loss'][0]
        val_loss = hist.history['val_loss'][0]
        assert train_loss > 0, "Error: Train cost should be bigger than 0"
        assert val_loss > 0, "Error: Validation cost should be bigger than 0"
        self.train_loss_history.append(train_loss)
        self.val_loss_history.append(val_loss)
        
        
    def update_replay_memory(self, uav_location_scale, user_heatmaps, uav_movement_decision):
        '''
        Function: 
            - Replay the old entries with new entries in the replay memory
            - Re-train the DNN periodically 
        Parameters: 
            uav_location: shape=(2,), x_uav and y_uav in range (0,1)
            user_heatmaps: shape=(n_grids, n_grids, n_channels=3) 
            uav_movement: shape=(3,), which are dx, dy, and dv 
        Returns: None 
        '''
        if rng.uniform() <= self.train_test_ratio:
            # Update the replay memory for training 
            sample_id = self.memory_counter_train % self.memory_size 
            replay_memory = self.replay_memory_train
            self.memory_counter_train += 1 
        else:
            # Update the replay memory for validation
            sample_id = self.memory_counter_val % self.val_memory_size 
            replay_memory = self.replay_memory_val
            self.memory_counter_val += 1 
        replay_memory['uav-location'][sample_id] = uav_location_scale
        replay_memory['user-heatmaps'][sample_id] = user_heatmaps
        replay_memory['best-action'][sample_id] = uav_movement_decision
        

    def make_movement_decision(self, uav_location, user_locations, user_statistics, normalization_coeff):
        '''
        Given the UAV's location and heatmaps of all user statistics:  
        - Step 1: preprocess the input data (uav location, user location and user statistics)
        - Step 2: generate a set of k potential decisions based on users' statistics. 
        - Step 3: select the best decision among $k$ ones generated in the first step. 
        - Step 4: update the replay memory
        
        Parameters:
            - uav_location: a tupple of (x_uav, y_uav, z_uav)            
            - user_locations: current location of each users, a tupple of (x_locations, y_locations), x_locations.shape = (n_users,)
            - user_statistics: all statistics of users, a tupple of (queue_Mb, traffic_Mb, ch_capacity_Mb)
            - normalization_coeff: normalization coefficients, a tupple of (queue_coeff_Mb, traffic_coeff_Mb, ch_capacity_coeff_Mb), queue_coeff_Mb.shape=(n_users,)
        
        Returns: 
            - prediction: the DNN output, shape=(2,)
            - best_decision: shape=(2,), a tupple of (velocity_x, velocity_y), 
                where velocity_x**2 + velocity_y**2 <= velocity_max**2
            
        Notes: 
            - uav_location_scale: a tupple of (x_uav, y_uav), where x_uav and y_uav in range (0,1)
            - user_heatmaps: shape=(n_grids,n_grids,n_channels), to be passed to the DNN 
            - n_decisions: # of decisions generated by the actor module 
            - dnn_decisions: a set of $n_decisions$ decisions output by the actor module
            - ready_decisions:  a set of $n_decisions$ potential ready-to-use decisions, 
                as a list of tupples of (dx,dy,velocity),
                dx and dy in range (-1,1), dx**2 + dy**2 = 1
                velocity in range (0,uav_speed_max)
        '''
        ### (1) Preprocessing the input data 
        user_heatmaps = self.preprocess_user_statistics(user_locations=user_locations, user_statistics=user_statistics, normalization_coeff=normalization_coeff)
        uav_location_scale = self.preprocess_uav_location(uav_location)
        assert np.all(user_heatmaps >= 0), f"Error in generating user statistic heatmaps, {user_heatmaps}"
        assert np.all(uav_location_scale<=1) and np.all(uav_location_scale>=0), f"Error in scaling the UAV location, {uav_location_scale}"
        
        
        ### (2) Generate a set of potential decisions, must balance between exploration and exploitation
        # The first decision: purely based on the DNN output 
        # The remaining (k-1) decisions: generated (noise-added) based on the DNN output 
        dnn_prediction = self.model.predict([tf.expand_dims(uav_location_scale, axis=0), 
                                             tf.expand_dims(user_heatmaps, axis=0)])    
        dnn_prediction = np.array(dnn_prediction).flatten()             # shape = (3,)
        assert np.all(dnn_prediction>=-1) and np.all(dnn_prediction<=1), f"DNN output for velocity_x and velocity_y not expected, {dnn_prediction}"
        
        dnn_decisions = []          # potential decisions, a python list of numpy arrays of shape (1,3), 
        dnn_decisions.append(dnn_prediction)        # dnn prediction 
        dnn_decisions.append(np.array([1e-3, 1e-3, 1e-3]))      # no movement 
        for decision_id in range(2,self.n_decisions): 
            noise = np.random.normal(loc=0, scale=self.std_var_vxvyvz, size=(3,))
            new_decision = tf.math.tanh(dnn_prediction + noise).numpy()  # in range [-1,1], aother option: dnn_prediction + noise    
            v_tmp = np.sqrt(np.sum(new_decision[:2]**2))
            if v_tmp > 1 :      # make sure that the UAV can move within a circle with radius = vmax*tslot
                new_decision[:2] = new_decision[:2] * 1/v_tmp       # rescale so that vx**2 + vy**2 = 1 
            assert np.all(new_decision <= 1) and np.all(new_decision>=-1), f"Error in generating new decisions, {new_decision}"
            dnn_decisions.append(new_decision)
               
                    
        ### (3) Select the best decision based on generated ones 
        # Post-process all decions so that they are ready-to-use 
        # Evaluate generated decisions one by one to find the decision with smallest Lyapunov function value 
        ready_decisions = self.postprocess_decision(dnn_decisions)          # type=np.ndarray, shape=(n_decisions,5)
        assert np.all(ready_decisions[:,0]**2 + ready_decisions[:,1]**2 <= 1 + 1e-3), f"Error in post-process the DNN output, dx, dy = {ready_decisions[:,:2]}"
        assert np.all(ready_decisions[:,2] <= self.uav_speedxy_max) and np.all(ready_decisions[:,2] >= 0), f"Error in post-process the DNN output, vxy = {ready_decisions[:,2]}"
        assert np.all(np.mod(ready_decisions[:,3],1)==0) and np.all(ready_decisions[:,4]>=0) and np.all(ready_decisions[:,4]<=self.uav_speedz_max), f"Error in postprocess the DNN output, dz, vz = {ready_decisions[:,3:]}" 
        
        lyapunov_fval_logs = []             # log all lyapunov function values of each decision 
        for decision_id in range(self.n_decisions): 
            fval = self.evaluate_decision(uav_location, user_locations, user_statistics, ready_decisions[decision_id])
            lyapunov_fval_logs.append(fval)
        id_best = np.argmax(lyapunov_fval_logs)     # index of the best decision 
        best_ready_decision = ready_decisions[id_best]      # shape = (5,)
        best_dnn_decision = dnn_decisions[id_best]          # shape = (3,)
        
        # Monitor the test loss 
        test_loss = mse(best_dnn_decision, dnn_prediction)
        self.test_loss_history.append(test_loss)
        
        ### (4) Update the replay memory and conduct validation test 
        self.update_replay_memory(uav_location_scale=uav_location_scale, user_heatmaps=user_heatmaps, uav_movement_decision=best_dnn_decision)
        
        return (dnn_prediction, best_ready_decision)
    
    
    def evaluate_decision(self, uav_location, user_locations, user_statistics, ready_decision): 
        '''
        Parameters: 
            uav_location=(x_uav, y_uav): python tuple, x_uav and y_uav in range (-boundary, boundary)
            user_locations: a tupple of (x_locations, y_locations), x_locations.shape = (n_users,)
            user_statistics: a tupple of (queue_Mb, traffic_Mb, ch_capacity_Mb), queue_Mb.shape=(n_users,)
            ready_decision=[dx, dy, v]: type=np.ndarray, dx**2 + dy**2 = 1, 0 <= uav_speed <=uav_speed_max
        Returns: 
            fval: Lyapunov-drift-plus-penalty function value of the given decision 
        '''
        dx, dy, vxy, dz, vz = ready_decision
        x_uav, y_uav = update_UAV_location(x0=uav_location[0], y0=uav_location[1], dx=dx, dy=dy, speed=vxy)
        z_uav = update_UAV_altitude(z0=uav_location[2], dz=dz, vz=vz)
        
        x_users, y_users = user_locations[0], user_locations[1]
        queue_Mb, traffic_Mb, _ = user_statistics
        n_users = len(x_users)
        ch_capacity_Mb = np.zeros(shape=(n_users,))
        for id in range(n_users):
            ch_gain = self.cal_channel_fading_gain(x_users[id], y_users[id], x_uav, y_uav, z_uav)
            ch_capacity_Mb[id] = self.cal_channel_capacity_Mb(channel_gain=ch_gain, alpha=1)   # equal bandwidth allocation 
        fval = np.sum((queue_Mb+traffic_Mb)*ch_capacity_Mb + self.Vlya * np.minimum(queue_Mb+traffic_Mb, ch_capacity_Mb)) 
        
        # add a penalty if the UAV moving out of the operation zone 
        indicator = 1 if (z_uav < self.zmin) or (z_uav > self.zmax) else 0 
        fval -= indicator*1e3       # NOTE: the optimization is maximization, thus the penalty should be negative 
        
        return fval
    

if __name__ == '__main__':
    boundary = 250
    n_users = 5 

    grid_size = 50
    norm_queue_Mb = 50      # in Mb 
    
    controller = UAV_Movement_Controller(boundary=boundary, grid_size=grid_size)

    t = 750
    x_raw = np.array([-boundary, boundary, boundary, -boundary, 0])
    y_raw = np.array([boundary, boundary, -boundary, -boundary, 0])
    x = x_raw + boundary
    y = y_raw - boundary
    val = np.array([10, 20, 30, 40, 50])
    val_norm = val/norm_queue_Mb

    print('Raw:')
    for i in range(n_users):
        print(f'({x_raw[i]:.0f}, {y_raw[i]:.0f})\t: {val_norm[i]:.1f} Mb')
        
    print('\nAdjusted:')
    for i in range(n_users):
        print(f'({x[i]:.0f}, {y[i]:.0f})\t: {val_norm[i]:.1f} Mb')
    print('\n')

    heatmap = controller.gen_heatmap(x_locations=x, y_locations=y, val=val, norm_val=norm_queue_Mb)
    print(heatmap)

    hm_loc = sn.heatmap(data=heatmap, vmin=0, vmax=1, square=True, annot = True, xticklabels=True)
    plt.show() 
    
    print('finished')


