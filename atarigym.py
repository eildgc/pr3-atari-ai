import gym
import numpy as np
# limitar velocidad 1/2
import time
import keras # used 2.1.16 with tensorflow 1.8.0/



import matplotlib.pyplot as plt
%matplotlib inline

env = gym.make('Pong-v0')
observation = env.reset()
observation.shape

for _ in range(1000):
    # env.render()

    # # Limitar velocidad 2/2
    # time.sleep(0.05)

    # i take some steps to see middle game scene    
    # in pong game,
    # 0 is numeric action to stay at same place 
    # 2 is numeric action to move paddle up in game
    # 3 is numeric action to move paddle down in game
    
    observation, reward, done, info = env.step(env.action_space.sample()) # take a random action
    
    
    new_observation, reward, done, info = env.step(2)
    plt.imshow(observation)

    # 

    # so the definition of system in reinforcement learning method is simple:

    # - state is the screen of game.
    # - action is going up or down

    # we could also define 3 actions (go up, go down, stay still) but here we just use the 2 mentioned above

def preprocess_frames(new_frame,last_frame):
# inputs are 2 numpy 2d arrays
n_frame = new_frame.astype(np.int32)
n_frame[(n_frame==144)|(n_frame==109)]=0 # remove backgound colors
l_frame = last_frame.astype(np.int32)
l_frame[(l_frame==144)|(l_frame==109)]=0 # remove backgound colors
diff = n_frame - l_frame
# crop top and bot 
diff = diff[35:195]
# down sample 
diff=diff[::2,::2]
# convert to grayscale
diff = diff[:,:,0] * 299. / 1000 + diff[:,:,1] * 587. / 1000 + diff[:,:,2] * 114. / 1000
# rescale numbers between 0 and 1
max_val =diff.max() if diff.max()> abs(diff.min()) else abs(diff.min())
if max_val != 0:
    diff=diff/max_val
return diff


plt.imshow(preprocess_frames(new_observation,observation),plt.cm.gray)

preprocess_frames(new_observation,observation)
preprocess_frames(new_observation,observation).shape

# Modeling the Network

# simple 2 layer model 
# with 200 hidden units in first layer
# and 1 sigmoid output
inputs = keras.layers.Input(shape=(80,80))
flattened_layer = keras.layers.Flatten()(inputs)
full_connect_1 = keras.layers.Dense(units=200,activation='relu',use_bias=False,)(flattened_layer)
sigmoid_output = keras.layers.Dense(1,activation='sigmoid',use_bias=False)(full_connect_1)
policy_network_model = keras.models.Model(inputs=inputs,outputs=sigmoid_output)
policy_network_model.summary()

episode_reward = keras.layers.Input(shape=(1,),name='episode_reward')

# Defining loss

def m_loss(episode_reward):
    def loss(y_true,y_pred):
        # feed in y_true as actual action taken 
        # if actual action was up, we feed 1 as y_true and otherwise 0
        # y_pred is the network output(probablity of taking up action)
        # note that we dont feed y_pred to network. keras computes it
        
        # first we clip y_pred between some values because log(0) and log(1) are undefined
        tmp_pred = keras.layers.Lambda(lambda x: keras.backend.clip(x,0.05,0.95))(y_pred)
        # we calculate log of probablity. y_pred is the probablity of taking up action
        # note that y_true is 1 when we actually chose up, and 0 when we chose down
        # this is probably similar to cross enthropy formula in keras, but here we write it manually to multiply it by the reward value
        tmp_loss = keras.layers.Lambda(lambda x:-y_true*keras.backend.log(x)-(1-y_true)*(keras.backend.log(1-x)))(tmp_pred)
        # multiply log of policy by reward
        policy_loss=keras.layers.Multiply()([tmp_loss,episode_reward])
        return policy_loss
    return loss

episode_reward = keras.layers.Input(shape=(1,),name='episode_reward')
policy_network_train = keras.models.Model(inputs=[inputs,episode_reward],outputs=sigmoid_output)

my_optimizer = keras.optimizers.RMSprop(lr=0.0001)
policy_network_train.compile(optimizer=my_optimizer,loss=m_loss(episode_reward),)

# Reward Engineering and why it is important
def generate_episode(policy_network):
    states_list = [] # shape = (x,80,80)
    up_or_down_action_list=[] # 1 if we chose up. 0 if down
    rewards_list=[]
    network_output_list=[]
    env=gym.make("Pong-v0")
    observation = env.reset()
    new_observation = observation
    done = False
    policy_output_list = []
    
    while done == False:
    
        processed_network_input = preprocess_frames(new_frame=new_observation,last_frame=observation)
        states_list.append(processed_network_input)
        reshaped_input = np.expand_dims(processed_network_input,axis=0) # x shape is (80,80) so we need similar reshape(x,(1,80,80))

        up_probability = policy_network.predict(reshaped_input,batch_size=1)[0][0]
        network_output_list.append(up_probability)
        policy_output_list.append(up_probability)
        actual_action = np.random.choice(a=[2,3],size=1,p=[up_probability,1-up_probability]) # 2 is up. 3 is down 
        if actual_action==2:
            up_or_down_action_list.append(1)
        else:
            up_or_down_action_list.append(0)
        
        observation= new_observation
        new_observation, reward, done, info = env.step(actual_action)
        
        rewards_list.append(reward)
        
        if done:
            break
            
    env.close()
    return states_list,up_or_down_action_list,rewards_list,network_output_list

    states_list,up_or_down_action_list,rewards_list,network_output_list = generate_episode(policy_network_model)

    

print("length of states= "+str(len(states_list)))# this is the number of frames
print("shape of each state="+str(states_list[0].shape))
print("length of rewards= "+str(len(rewards_list)))


# lets see sample of policy output
print(network_output_list[30:50]) 



#lets see a sample what we actually did: 1 means we went up, 0 means down
up_or_down_action_list[30:50]

# lets see sample of rewards
print(rewards_list[50:100]) 



# lets see how many times we won through whole game:
print("count win="+str(len(list(filter(lambda r: r>0,rewards_list)))))
print("count lose="+str(len(list(filter(lambda r: r<0,rewards_list)))))
print("count zero rewards="+str(len(list(filter(lambda r: r==0,rewards_list)))))

plt.plot(rewards_list,'.')
ax=plt.gca()
ax.grid(True)

def process_rewards(r_list):
    reward_decay=0.99
    tmp_r=0
    rew=np.zeros_like(r_list,dtype=np.float32)
    for i in range(len(r_list)-1,-1,-1):
        if r_list[i]==0:
            tmp_r=tmp_r*reward_decay
            rew[i]=tmp_r
        else: 
            tmp_r = r_list[i]
            rew[i]=tmp_r
    rew -= np.mean(rew) # subtract by average
    rew /= np.std(rew) # divide by std
    return rew

    

# lets see what this gives us:
plt.plot(process_rewards(rewards_list),'-')
ax=plt.gca()
ax.grid(True)

# Example of simluation and training

# first generate an episode:
states_list,up_or_down_action_list,rewards_list,network_output_list = generate_episode(policy_network_model)

print("length of states= "+str(len(states_list)))# this is the number of frames
print("shape of each state="+str(states_list[0].shape))
print("length of rewards= "+str(len(rewards_list)))

#preprocess inputs for training: 
    
x=np.array(states_list)

episode_reward=np.expand_dims(process_rewards(rewards_list),1)

y_tmp = np.array(up_or_down_action_list) # 1 if we chose up, 0 if down
y_true = np.expand_dims(y_tmp,1) # modify shape. this is neccassary for keras


print("episode_reward.shape =",episode_reward.shape)
print("x.shape =",x.shape)
print("y_true.shape =",y_true.shape)

#  fit the model with inputs and outputs.
policy_network_train.fit(x=[x,episode_reward],y=y_true)


# Training the network

# we define a helper function to create a batch of simulations
# and after the batch simulations, preprocess data and fit the network
def generate_episode_batches_and_train_network(n_batches=10):
    env = gym.make('Pong-v0')
    batch_state_list=[]
    batch_up_or_down_action_list=[]
    batch_rewards_list=[]
    batch_network_output_list=[]
    for i in range(n_batches):
        states_list,up_or_down_action_list,rewards_list,network_output_list = generate_episode(policy_network_model)   
        batch_state_list.extend(states_list[15:])
        batch_network_output_list.extend(network_output_list[15:])
        batch_up_or_down_action_list.extend(up_or_down_action_list[15:])
        batch_rewards_list.extend(rewards_list[15:])
    
    episode_reward=np.expand_dims(process_rewards(batch_rewards_list),1)
    x=np.array(batch_state_list)
    y_tmp = np.array(batch_up_or_down_action_list)
    y_true = np.expand_dims(y_tmp,1)
    policy_network_train.fit(x=[x,episode_reward],y=y_true)

    return batch_state_list,batch_up_or_down_action_list,batch_rewards_list,batch_network_output_list

train_n_times = 21 # for actual training, about 5000 may be a good start. 
for i in range(train_n_times):
    states_list,up_or_down_action_list,rewards_list,network_output_list=generate_episode_batches_and_train_network(10)
    if i%10==0:
        print("i="+str(i))
        rr=np.array(rewards_list)
        # i keep how many times we won in batch. you can use log more details more frequently
        print('count win='+str(len(rr[rr>0]))) 
        policy_network_model.save("policy_network_model_simple.h5")
        policy_network_model.save("policy_network_model_simple"+str(i)+".h5")
        with open('rews_model_simple.txt','a') as f_rew:
            f_rew.write("i="+str(i)+'       reward= '+str(len(rr[rr > 0])))
            f_rew.write("\n")

#  Playing the Trained Network
def play_and_show_episode(policy_network):
    env = gym.make('Pong-v0')
    done=False
    observation = env.reset()
    new_observation = observation
    while done==False:
        time.sleep(1/80)
        
        processed_network_input = preprocess_frames(new_frame=new_observation,last_frame=observation)
        reshaped_input = np.expand_dims(processed_network_input,axis=0) # x shape is (80,80) so we need similar reshape(x,(1,80,80))

        up_probability = policy_network.predict(reshaped_input,batch_size=1)[0][0]
        actual_action = np.random.choice(a=[2,3],size=1,p=[up_probability,1-up_probability])
        
        env.render()
        
        observation= new_observation
        new_observation, reward, done, info = env.step(actual_action)
        if reward!=0:
            print(reward)
        if done:
            break
        
    env.close()

    # Loading model from file

policy_network_model=keras.models.load_model("trained_simple_model_3300.h5")
policy_network_model.summary()

episode_reward = keras.layers.Input(shape=(1,),name='episode_reward')

def m_loss(episode_reward):
    def loss(y_true,y_pred):
        # feed in y_true as actual action taken
        # loss = reward*(-actual*np.log(y_pred)-(1-actual)*np.log(1-y_pred)))
        
        tmp_pred = keras.layers.Lambda(lambda x: keras.backend.clip(x,0.05,0.95))(y_pred) # we could also do gradient clipping
        tmp_loss = keras.layers.Lambda(lambda x:-y_true*keras.backend.log(x)-(1-y_true)*(keras.backend.log(1-x)))(tmp_pred)
        # put reward in effect
        policy_loss=keras.layers.Multiply()([tmp_loss,episode_reward])
        
        return policy_loss
    return loss

policy_network_train = keras.models.Model(inputs=[policy_network_model.input,episode_reward],outputs=policy_network_model.output)
my_optimizer = keras.optimizers.RMSprop(lr=0.0001)
policy_network_train.compile(optimizer=my_optimizer,loss=m_loss(episode_reward),)


play_and_show_episode(policy_network_model)

# Using Convolutional Model

policy_network_model=keras.models.load_model("trained_conv_model.h5")
policy_network_model.summary()

play_and_show_episode(policy_network_model)