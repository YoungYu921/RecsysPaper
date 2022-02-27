def FM(features, labels, mode, params):   # 网络结构
    feature_columns = build_features()
    #input = tf.feature_column.input_layer(features, feature_columns)
    inputs = []
    fea_embed = []
    for column in feature_columns:
        input_tmp = tf.feature_column.input_layer(features, [column])
        inputs.append(input_tmp)
        fea_name = '_'.join(column.name.split('_')[:-1])
        tf.logging.info("--------Feature_name: %s, Embed_tensor: %s, Input_feature_tensor: %s", fea_name, input_tmp, features[fea_name])
        if params['fea']==fea_name: fea_embed = input_tmp
    input = tf.concat(inputs, axis=-1, name='concat_all_features')
    # input = tf.layers.batch_normalization(input, momentum=0.95, center=True, scale=True, trainable=True, training=(mode == tf.estimator.ModeKeys.TRAIN))
    input_dim = input.get_shape().as_list()[-1]
    with tf.variable_scope('linear'):
        init = tf.random_normal(shape=(input_dim, 1))
        w = tf.get_variable('w', dtype=tf.float32, initializer=init, validate_shape=False)
        b = tf.get_variable('b', shape=[1], dtype=tf.float32)
        linear_term = tf.add(tf.matmul(input, w), b)
        add_layer_summary(linear_term.name, linear_term)
    with tf.variable_scope('fm_interaction'):
        init = tf.truncated_normal(shape=(input_dim, params['factor_dim']))
        v = tf.get_variable('v', dtype=tf.float32, initializer=init, validate_shape=False)
        sum_square = tf.pow(tf.matmul(input, v), 2)
        square_sum = tf.matmul(tf.pow(input, 2), tf.pow(v, 2))
        interaction_term = 0.5 * tf.reduce_sum(sum_square - square_sum, axis=1, keep_dims=True)
        add_layer_summary(interaction_term.name, interaction_term)
    with tf.variable_scope('output'):
        y = tf.math.add(interaction_term, linear_term)
        add_layer_summary(y.name, y)
    return y, fea_embed
