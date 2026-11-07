import math

def lr_at(epoch, epochs):
    if epoch < 5:
        return 1e-6 + epoch * (.001 - 1e-6) / 5
    return 1e-5 + .5 * (.001 - 1e-5) * (1 + math.cos(math.pi * epoch / epochs))
