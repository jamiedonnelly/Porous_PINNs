


def inference(model, data):
    # To speed up inference, since the flow moves left -> right:
    # rather than perform inference for every (x,y,t+1) tuple, only perform 
    # in (y, x[:, x_max], t+1), where x_max is the right-most co-ordinate 
    # for which gamma_{t} > 0.
    pass

