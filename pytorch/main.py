import os
from sklearn.model_selection import train_test_split

def setup(path='./data', test_size=0.2, random_state=42):

    classes = sorted(os.listdir(path))

    print("Generating images and labels ...")
    images, encoded = [], []
    for ix, label in enumerate(classes):
        _images = os.listdir(f'{path}/{label}')
        images += [f'{path}/{label}/{img}' for img in _images]
        encoded += [ix]*len(_images)
    print(f'Number of images: {len(images)}')

     # train / val split
    print("Generating train / val splits ...")
    train_images, val_images, train_labels, val_labels = train_test_split(
        images,
        encoded,
        stratify=encoded,
        test_size=test_size,
        random_state=random_state
    )

    print("Training samples: ", len(train_labels))
    print("Validation samples: ", len(val_labels))

    return classes, train_images, train_labels, val_images, val_labels

classes, train_images, train_labels, val_images, val_labels = setup('./data')