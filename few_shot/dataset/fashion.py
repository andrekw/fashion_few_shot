import os

import pandas as pd

basedir = '../datasets/fashion-dataset'


def build_fashion_df(basedir, min_rows: int = 10):
    img_dir = os.path.join(basedir, 'images')

    df = pd.read_csv(os.path.join(basedir, 'styles.csv'), error_bad_lines=False)

    valid_image_ids = set(
        [int(os.path.splitext(i)[0]) for i in os.listdir(img_dir) if os.path.splitext(i)[0].isnumeric()])

    valid_image_ids.intersection_update(df.id.unique())

    df['filepath'] = df.id.apply(
        lambda x: os.path.abspath(os.path.join(img_dir, f'{x}.jpg')))
    df['class_name'] = df.articleType

    rows_per_class = df[df.id.isin(valid_image_ids)].groupby('articleType').id.nunique().reset_index()
    valid_classes = rows_per_class[rows_per_class > min_rows]

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
