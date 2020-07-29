from keras.models import Model
from keras.layers import Input, Dense, Concatenate, Reshape, ZeroPadding1D, LSTM, Lambda
from keras import regularizers
from ipdb import set_trace as db

def get_model_fcnn_scalar(input_profile_names, target_profile_names, scalar_input_names, scalar_target_names,
                          actuator_names, lookbacks, lookahead, profile_length, std_activation, **kwargs):
    l2 = kwargs.get('l2',0)
    kernel_init = kwargs.get('kernel_initializer','glorot_uniform')
    bias_init = kwargs.get('bias_initializer','zeros')
    max_actuator_lookback = 0
    for sig in actuator_names:
        if lookbacks[sig] > max_actuator_lookback:
            max_actuator_lookback = lookbacks[sig]
    max_scalar_lookback = 0
    for sig in scalar_input_names:
        if lookbacks[sig] > max_scalar_lookback:
            max_scalar_lookback = lookbacks[sig]
    num_scalars = len(scalar_input_names)
    num_targets_scalar = len(scalar_target_names)
    num_actuators = len(actuator_names)

    assert num_scalars == num_targets_scalar
    assert num_scalars > 0

    scalar_inputs = []
    scalars = []
    for i in range(num_scalars):
        scalar_inputs.append(
            Input((lookbacks[scalar_input_names[i]]+1,), name='input_' + scalar_input_names[i]))
        scalars.append(Reshape((lookbacks[scalar_input_names[i]]+1,1))(scalar_inputs[i]))
        scalars[i] = ZeroPadding1D(padding=(max_scalar_lookback - lookbacks[scalar_input_names[i]], 0))(scalars[i])
    if num_scalars>1:
        scalars = Concatenate(axis=-1)(scalars)
    else:
        scalars = scalars[0]
    scalars = Dense(units=256,activation=std_activation,
                   kernel_regularizer=regularizers.l2(l2),bias_regularizer=regularizers.l2(l2),
                   kernel_initializer=kernel_init, bias_initializer=bias_init)(scalars)
    scalars = LSTM(units=256, activation=std_activation,
                   recurrent_activation='hard_sigmoid',recurrent_regularizer=regularizers.l2(l2),
                  kernel_regularizer=regularizers.l2(l2),bias_regularizer=regularizers.l2(l2),
                  kernel_initializer=kernel_init, bias_initializer=bias_init)(scalars)
    actuator_future_inputs = []
    actuator_past_inputs = []
    actuators = []
    for i in range(num_actuators):
        actuator_future_inputs.append(
            Input((lookahead, ), name='input_future_' + actuator_names[i]))
        actuator_past_inputs.append(
            Input((lookbacks[actuator_names[i]]+1, ), name='input_past_' + actuator_names[i]))
        actuators.append(Concatenate(
            axis=1)([actuator_past_inputs[i], actuator_future_inputs[i]]))
        actuators[i] = Reshape(
            (lookbacks[actuator_names[i]]+lookahead+1, 1))(actuators[i])
        actuators[i] = ZeroPadding1D(padding=(max_actuator_lookback - lookbacks[actuator_names[i]], 0))(actuators[i])
    if num_actuators>1:
        actuators = Concatenate(axis=-1)(actuators)
    else:
        actuators = actuators[0]
    # shaoe = (time, num_actuators)
    actuators = Dense(units=256,activation=std_activation,
                     kernel_regularizer=regularizers.l2(l2),bias_regularizer=regularizers.l2(l2),
                     kernel_initializer=kernel_init, bias_initializer=bias_init)(actuators)
    actuators = LSTM(units=256, activation=std_activation,
                     recurrent_activation='hard_sigmoid',recurrent_regularizer=regularizers.l2(l2),
                    kernel_regularizer=regularizers.l2(l2),bias_regularizer=regularizers.l2(l2),
                    kernel_initializer=kernel_init, bias_initializer=bias_init)(actuators)

    merged = Concatenate(axis=-1)([actuators, scalars])
    merged = Dense(units = 256,activation=std_activation,
                   kernel_regularizer=regularizers.l2(l2),bias_regularizer=regularizers.l2(l2),
                   kernel_initializer=kernel_init, bias_initializer=bias_init)(merged)
    merged = Dense(units = 256,activation=std_activation,
                   kernel_regularizer=regularizers.l2(l2),bias_regularizer=regularizers.l2(l2),
                   kernel_initializer=kernel_init, bias_initializer=bias_init)(merged)
    new_scalars = Dense(units = num_scalars, activation=None,
                   kernel_regularizer=regularizers.l2(l2),bias_regularizer=regularizers.l2(l2),
                   kernel_initializer=kernel_init, bias_initializer=bias_init)(merged)
    scalar_outputs = []
    for i in range(num_targets_scalar):
        scalar_outputs.append(Lambda(lambda x: x[:, i:i+1], name='target_' + scalar_target_names[i])(new_scalars))
    model_inputs = scalar_inputs + actuator_past_inputs + actuator_future_inputs
    model = Model(inputs=model_inputs, output=scalar_outputs)
    return model
