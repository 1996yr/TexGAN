import tensorflow as tf

def tileImage(input_batch, nCol = 10, pad_value = 1.0):
    assert nCol >= 1
    input_shape = input_batch.get_shape().as_list()
    nImg = input_shape[0]
    assert nImg >= 1
    nRow = (nImg - 1) // nCol + 1
    output_rows = []
    for r in range(nRow):
        output_row = []
        for c in range(nCol):
            if r * nCol + c < nImg:
                output_row.append(input_batch[r * nCol + c,:,:,:])
            else:
                output_row.append(pad_value * tf.ones_like(input_batch[0]))
        output_rows.append(tf.concat(output_row, axis=1))
    output = tf.concat(output_rows, axis=0)
    return output[tf.newaxis,...]