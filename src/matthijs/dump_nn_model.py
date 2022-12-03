import torch
def dump_nn_model_steps(model, example_input):
    """
    List the steps of each convolution in a model and show the result for a given input. If possible, also show what happens in a dense layer
    """
    current = example_input
    print("===============")
    print(f'Starting model with shape {current.shape}')
    print("===============")
    if(model.convolutions):
        for conv in model.convolutions:
            print("-----------------")
            print("Convolution:")
            print(conv)
            current = conv(current)
            print(current.shape)
    if(model.dense):
        print("Contents of dense:")
        for dense in model.dense:
            print("-----------------")
            print("Dense:")
            print(dense)
            current = dense(current)
            print(current.shape)