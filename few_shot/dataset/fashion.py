import os
from typing import Tuple

import numpy as np
import pandas as pd


def build_fashion_df(basedir, min_rows: int = 10):
    img_dir = os.path.join(basedir, 'images')

    df = pd.read_csv(os.path.join(basedir, 'styles.csv'), error_bad_lines=False)

    valid_image_ids = set(
        [int(os.path.splitext(i)[0]) for i in os.listdir(img_dir) if os.path.splitext(i)[0].isnumeric()])

    valid_image_ids.intersection_update(df.id.unique())

    df['filepath'] = df.id.apply(
        lambda x: os.path.abspath(os.path.join(img_dir, f'{x}.jpg')))
    df['class_name'] = df.articleType

    rows_per_class = df[df.id.isin(valid_image_ids)].groupby('articleType').id.nunique()
    valid_classes = set(rows_per_class[rows_per_class > min_rows].index)

    valid_df = df[(df.articleType.isin(valid_classes)) & (df.id.isin(valid_image_ids))]

    return valid_df


TRAINING_CLASSES = {
    'Cufflinks',
    'Rompers',
    'Laptop Bag',
    'Sports Sandals',
    'Hair Colour',
    'Suspenders',
    'Trousers',
    'Kajal and Eyeliner',
    'Compact',
    'Concealer',
    'Jackets',
    'Mufflers',
    'Backpacks',
    'Sandals',
    'Shorts',
    'Waistcoat',
    'Watches',
    'Pendant',
    'Basketballs',
    'Bath Robe',
    'Boxers',
    'Deodorant',
    'Rain Jacket',
    'Necklace and Chains',
    'Ring',
    'Formal Shoes',
    'Nail Polish',
    'Baby Dolls',
    'Lip Liner',
    'Bangle',
    'Tshirts',
    'Flats',
    'Stockings',
    'Skirts',
    'Mobile Pouch',
    'Capris',
    'Dupatta',
    'Lip Gloss',
    'Patiala',
    'Handbags',
    'Leggings',
    'Ties',
    'Flip Flops',
    'Rucksacks',
    'Jeggings',
    'Nightdress',
    'Waist Pouch',
    'Tops',
    'Dresses',
    'Water Bottle',
    'Camisoles',
    'Heels',
    'Gloves',
    'Duffel Bag',
    'Swimwear',
    'Booties',
    'Kurtis',
    'Belts',
    'Accessory Gift Set',
    'Bra'
    }


TEST_CLASSES = {
    'Jeans',
    'Bracelet',
    'Eyeshadow',
    'Sweaters',
    'Sarees',
    'Earrings',
    'Casual Shoes',
    'Tracksuits',
    'Clutches',
    'Socks',
    'Innerwear Vests',
    'Night suits',
    'Salwar',
    'Stoles',
    'Face Moisturisers',
    'Perfume and Body Mist',
    'Lounge Shorts',
    'Scarves',
    'Briefs',
    'Jumpsuit',
    'Wallets',
    'Foundation and Primer',
    'Sports Shoes',
    'Highlighter and Blush',
    'Sunscreen',
    'Shoe Accessories',
    'Track Pants',
    'Fragrance Gift Set',
    'Shirts',
    'Sweatshirts',
    'Mask and Peel',
    'Jewellery Set',
    'Face Wash and Cleanser',
    'Messenger Bag',
    'Free Gifts',
    'Kurtas',
    'Mascara',
    'Lounge Pants',
    'Caps',
    'Lip Care',
    'Trunk',
    'Tunics',
    'Kurta Sets',
    'Sunglasses',
    'Lipstick',
    'Churidar',
    'Travel Accessory'
    }


def fashion_dfs(dataset_path: str,
                min_rows: int = 10,
                n_val_classes: int = 5) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Builds train, validation and test DataFrames from the kaggle fashion dataset.

    :param dataset_path: path to the dataset
    :param min_rows: required number of images for the class to be added to the DataFrames
    :param n_val_classes: how many classes to use in the validation set
    :returns: a tuple of train, validation and test DataFrames
    """
    # TODO: use new validation format
    df = build_fashion_df(dataset_path, min_rows)
    print(df.class_name.nunique())

    valid_train_classes = TRAINING_CLASSES.intersection(df.class_name.unique())

    val_classes = set(np.random.choice(list(valid_train_classes), n_val_classes, replace=False))
    train_df = df[df.class_name.isin(TRAINING_CLASSES - val_classes)]
    val_df = df[df.class_name.isin(val_classes)]

    test_df = df[df.class_name.isin(TEST_CLASSES)]

    return train_df, val_df, test_df