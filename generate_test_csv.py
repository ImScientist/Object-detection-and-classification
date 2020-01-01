import os
import itertools
import argparse
import pandas as pd
import numpy as np


def generate_test_csv(data_dir):
    files = os.listdir(os.path.join(data_dir, 'test_images'))
    files = filter(lambda x: '.jpg' in x, files)
    files = map(lambda x: [f"{x}_Fish", f"{x}_Flower", f"{x}_Gravel", f"{x}_Sugar"], files)
    files = itertools.chain(*files)
    files = list(files)

    tmp = '1 1'
    df = pd.DataFrame(data=np.array(files).reshape(-1, 1), columns=['Image_Label'])
    df = df.sort_values(by=['Image_Label'])
    df['EncodedPixels'] = df['Image_Label'].apply(lambda x: tmp if 'Fish' in x else None)

    df.to_csv(os.path.join(data_dir, 'test.csv'), index=False)


if __name__ == "__main__":
    """ Generate e dummy train.csv file that is required by the dataloader
    
    python genera_test_csv.py \
        --data_dir /content/drive/My Drive/data/source/clouds
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir',
                        type=str,
                        dest='data_dir',
                        default=None,
                        help='Directory with training data.')

    args = parser.parse_args()

    generate_test_csv(data_dir=args.data_dir)
