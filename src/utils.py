import matplotlib.pyplot as plt
from PIL import Image, ImageDraw


def cloud_colors():
    """ Colors assigned to each cloud type """

    colors = {
        'Sugar': (128, 0, 0, 255),
        'Gravel': (0, 128, 0, 255),
        'Flower': (0, 0, 128, 255),
        'Fish': (128, 0, 128, 255)}

    return colors


def display_colors_map():
    colors = cloud_colors()

    img = Image.new(mode='RGBA', size=(240, 30), color=(0, 0, 0, 255))
    draw = ImageDraw.Draw(img)

    for idx, (k, v) in enumerate(colors.items()):
        draw.text(xy=(10 + 60 * idx, 10), text=k, fill=v)

    plt.figure(figsize=(8, 3))
    plt.imshow(img)
    plt.pause(0.001)
    plt.show()
