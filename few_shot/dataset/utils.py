import tensorflow as tf


def pad_validation_inputs(n_shot=1,
                          n_queries_train=5,
                          n_queries_test=1,
                          k_way_train=15,
                          k_way_test=5):
    """Pads a train or validation tensor in order to have the same shape as the training data."""

    def pad_input(x_tensors, query_y):
        (support_x, support_y, query_x) = x_tensors
        support_x = tf.pad(support_x,
                           [

                               [0, n_shot * k_way_train - n_shot * k_way_test],
                               [0, 0],
                               [0, 0],
                               [0, 0]
                           ])
        support_y = tf.pad(support_y,
                           [

                               [0, n_shot * k_way_train - n_shot * k_way_test],
                               [0, k_way_train - k_way_test]
                           ])
        query_x = tf.pad(query_x,
                         [

                             [0, n_queries_train * k_way_train - n_queries_test * k_way_test],
                             [0, 0],
                             [0, 0],
                             [0, 0]
                         ])
        query_y = tf.pad(query_y,
                         [

                             [0, n_queries_train * k_way_train - n_queries_test * k_way_test],
                             [0, k_way_train - k_way_test]
                         ])
        return (support_x, support_y, query_x), query_y

    return pad_input
