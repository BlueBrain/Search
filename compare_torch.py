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

def compare_models(model_1, model_2):
    # https://discuss.pytorch.org/t/check-if-models-have-same-weights/4351/6
    models_differ = 0
    for key_item_1, key_item_2 in zip(model_1.items(), model_2.items()):
        if torch.equal(key_item_1[1].cpu(), key_item_2[1].cpu()):
            pass
        else:
            models_differ += 1
            if (key_item_1[0] == key_item_2[0]):
                rdiff = torch.norm(key_item_1[1].cpu()-key_item_2[1].cpu()) / torch.norm(key_item_1[1].cpu())
                print(f"Weights mismtach with relative difference {rdiff:.2e} found at {key_item_1[0]}")
            else:
                raise Exception
    if models_differ == 0:
        print('Models weights match perfectly! :)')


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

    compare_models(raw_model_1, raw_model_2)


if __name__ == "__main__":
    main()
