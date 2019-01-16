import keras.backend as K
import keras.layers as layers
import keras.models as models
import keras.optimizers as optimizers

class Critic:
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size):
        """Initialize parameters and build model.

        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
        """
        self.state_size = state_size
        self.action_size = action_size

        # Initialize any other variables here

        self.build_model()

    def build_model(self):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        # Define input layers
        states = layers.Input(shape=(self.state_size,), name='states')
        actions = layers.Input(shape=(self.action_size,), name='actions')
        
        # AA: I tried adding batch noramliztion for each layer after the linear combination and before the 
        # non-linear relu activation function, but removed it as it degraded performance. 
        # I also added glorot_normal initialization.
        
        # Add hidden layer(s) for state pathway
        net_states = layers.Dense(units=32, activation='relu', kernel_initializer='glorot_normal')(states)
#         net_states = layers.BatchNormalization()(net_states)
#         net_states = layers.Activation('relu')(net_states)
        net_states = layers.Dense(units=64, activation='relu', kernel_initializer='glorot_normal')(net_states)
#         net_states = layers.BatchNormalization()(net_states)
#         net_states = layers.Activation('relu')(net_states)
        # AA: added layer
        net_states = layers.Dense(units=64, activation='relu', kernel_initializer='glorot_normal')(net_states)
#         net_states = layers.BatchNormalization()(net_states)
#         net_states = layers.Activation('relu')(net_states)
        
        # Add hidden layer(s) for action pathway
        net_actions = layers.Dense(units=32, activation='relu', kernel_initializer='glorot_normal')(actions)
#         net_actions = layers.BatchNormalization()(net_actions)
#         net_actions = layers.Activation('relu')(net_actions)
        net_actions = layers.Dense(units=64, activation='relu', kernel_initializer='glorot_normal')(net_actions)
#         net_actions = layers.BatchNormalization()(net_actions)
#         net_actions = layers.Activation('relu')(net_actions)
        # AA: added layer
        net_actions = layers.Dense(units=64, activation=None, kernel_initializer='glorot_normal')(net_actions)
#         net_actions = layers.BatchNormalization()(net_actions)
#         net_actions = layers.Activation('relu')(net_actions)
        
        # Combine state and action pathways
        net = layers.Add()([net_states, net_actions])
        net = layers.Activation('relu')(net)

        # Add final output layer to prduce action values (Q values)
        Q_values = layers.Dense(units=1, name='q_values')(net)

        # Create Keras model
        self.model = models.Model(inputs=[states, actions], outputs=Q_values)

        # Define optimizer and compile model for training with built-in loss function
        optimizer = optimizers.Adam(lr=0.001)
        self.model.compile(optimizer=optimizer, loss='mse')

        # Compute action gradients (derivative of Q values w.r.t. to actions)
        action_gradients = K.gradients(Q_values, actions)

        # Define an additional function to fetch action gradients (to be used by actor model)
        self.get_action_gradients = K.function(
            inputs=[*self.model.input, K.learning_phase()],
            outputs=action_gradients)
