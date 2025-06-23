import math
import numpy as np
from tensorboardX import SummaryWriter

def main():
    degrees = np.linspace(0, 3600 * math.pi / 180.0, 3600)
    degrees = degrees.reshape(3600, 1)
    labels = ["%d" % (i) for i in range(0, 3600)]
    with SummaryWriter() as writer:
        for epoch in range(0, 16):
            shift = epoch * 2 * math.pi / 16.0
            mat = np.concatenate([
                np.sin(shift + degrees * 2 * math.pi / 180.0),
                np.sin(shift + degrees * 3 * math.pi / 180.0),
                np.sin(shift + degrees * 5 * math.pi / 180.0),
                np.sin(shift + degrees * 7 * math.pi / 180.0),
                np.sin(shift + degrees * 11 * math.pi / 180.0)
            ], axis=1)
            writer.add_embedding(
                mat=mat,
                metadata=labels,
                tag="sin",
                global_step=epoch)
            mat = np.concatenate([
                np.cos(shift + degrees * 2 * math.pi / 180.0),
                np.cos(shift + degrees * 3 * math.pi / 180.0),
                np.cos(shift + degrees * 5 * math.pi / 180.0),
                np.cos(shift + degrees * 7 * math.pi / 180.0),
                np.cos(shift + degrees * 11 * math.pi / 180.0)
            ], axis=1)
            writer.add_embedding(
                mat=mat,
                metadata=labels,
                tag="cos",
                global_step=epoch)
            mat = np.concatenate([
                np.tan(shift + degrees * 2 * math.pi / 180.0),
                np.tan(shift + degrees * 3 * math.pi / 180.0),
                np.tan(shift + degrees * 5 * math.pi / 180.0),
                np.tan(shift + degrees * 7 * math.pi / 180.0),
                np.tan(shift + degrees * 11 * math.pi / 180.0)
            ], axis=1)
            writer.add_embedding(
                mat=mat,
                metadata=labels,
                tag="tan",
                global_step=epoch)

if __name__ == "__main__":
    main()