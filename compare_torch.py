import argparse
import pathlib

import numpy as np
import torch

def assert_allclose(t_1, t_2, atol=1e-6):
    a_1, a_2 = t_1.detach().numpy(), t_2.detach().numpy()
    np.testing.assert_allclose(a_1, a_2, atol=atol)

def assert_array_equal(t_1, t_2):
    a_1, a_2 = t_1.detach().numpy(), t_2.detach().numpy()
    np.testing.assert_array_equal(a_1, a_2)



def main(argv=None):
    parser = argparse.ArgumentParser()

    parser.add_argument("path1", type=str)
    parser.add_argument("path2", type=str)

    args = parser.parse_args()

    path_1 = pathlib.Path(args.path1)
    path_2 = pathlib.Path(args.path2)

    # Load via torch.load
    raw_model_1 = torch.load(path_1)
    raw_model_2 = torch.load(path_2)

    # Make sure keys are the same
    assert list(raw_model_1.keys()) == list(raw_model_2.keys())

    # Make sure values are the same
    for (k_1, v_1), (k_2, v_2) in zip(raw_model_1.items(), raw_model_2.items()):
        assert k_1 == k_2

        assert isinstance(v_1, torch.Tensor)
        assert isinstance(v_2, torch.Tensor)

        assert_array_equal(v_1, v_2)

        if v_1.dtype == torch.float32:
            print(v_1.mean(), v_2.mean())

    print("Done")


if __name__ == "__main__":
    main()
