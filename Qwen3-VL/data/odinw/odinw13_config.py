datasets = [
{
    'metainfo': {'classes': ('movable-objects', 'boat', 'car', 'dock', 'jetski', 'lift')},
    'data_root': '/home/zirui/Qwen3-VL/data/odinw/AerialMaritimeDrone',
    'ann_file': '/large/test/_annotations.coco.json',
    'data_prefix': {'img': '/large/test/'}
},
{
    'metainfo': {'classes': ('creatures', 'fish', 'jellyfish', 'penguin', 'puffin', 'shark', 'starfish', 'stingray')},
    'data_root': '/home/zirui/Qwen3-VL/data/odinw/Aquarium',
    'ann_file': '/Aquarium Combined.v2-raw-1024.coco/test/_annotations.coco.json',
    'data_prefix': {'img': '/Aquarium Combined.v2-raw-1024.coco/test/'}
},
{
    'metainfo': {'classes': ('Cottontail-Rabbit', 'Cottontail-Rabbit')},
    'data_root': '/home/zirui/Qwen3-VL/data/odinw/Cottontail Rabbits',
    'ann_file': '/test/_annotations.coco.json',
    'data_prefix': {'img': '/test/'}
},
{
    'metainfo': {'classes': ('hands', 'myleft', 'myright', 'yourleft', 'yourright')},
    'data_root': '/home/zirui/Qwen3-VL/data/odinw/EgoHands',
    'ann_file': '/specific/test/_annotations.coco.json',
    'data_prefix': {'img': '/specific/test/'}
},
{
    'metainfo': {'classes': ('mushroom', 'CoW', 'chanterelle')},
    'data_root': '/home/zirui/Qwen3-VL/data/odinw/NorthAmerica Mushrooms',
    'ann_file': '/North American Mushrooms.v2-416x416augmented.coco/test/_annotations.coco.json',
    'data_prefix': {'img': '/North American Mushrooms.v2-416x416augmented.coco/test/'}
},
{
    'metainfo': {'classes': ('packages', 'package')},
    'data_root': '/home/zirui/Qwen3-VL/data/odinw/Packages',
    'ann_file': '/augmented-v1/test/_annotations.coco.json',
    'data_prefix': {'img': '/augmented-v1/test/'}
},
{
    'metainfo': {'classes': ('VOC', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')},
    'data_root': '/home/zirui/Qwen3-VL/data/odinw/Pascal VOC',
    'ann_file': '/valid/_annotations.coco.json',
    'data_prefix': {'img': '/valid/'}
},
{
    'metainfo': {'classes': ('potholes', 'pothole')},
    'data_root': '/home/zirui/Qwen3-VL/data/odinw/Pothole',
    'ann_file': '/test/_annotations.coco.json',
    'data_prefix': {'img': '/test/'}
},
{
    'metainfo': {'classes': ('raccoons', 'raccoon')},
    'data_root': '/home/zirui/Qwen3-VL/data/odinw/Raccoon',
    'ann_file': '/Raccoon.v38-416x416-resize.coco/test/_annotations.coco.json',
    'data_prefix': {'img': '/Raccoon.v38-416x416-resize.coco/test/'}
},
{
    'metainfo': {'classes': ('shellfish', 'Crab', 'Lobster', 'Shrimp')},
    'data_root': '/home/zirui/Qwen3-VL/data/odinw/ShellfishOpenImages',
    'ann_file': '/416x416/test/_annotations.coco.json',
    'data_prefix': {'img': '/416x416/test/'}
},
{
    'metainfo': {'classes': ('dogs-person', 'dog', 'person')},
    'data_root': '/home/zirui/Qwen3-VL/data/odinw/Thermal Dogs and People',
    'ann_file': '/test/_annotations.coco.json',
    'data_prefix': {'img': '/test/'}
},
{
    'metainfo': {'classes': ('vehicles', 'Ambulance', 'Bus', 'Car', 'Motorcycle', 'Truck')},
    'data_root': '/home/zirui/Qwen3-VL/data/odinw/Vehicles OpenImages',
    'ann_file': '/416x416/test/_annotations.coco.json',
    'data_prefix': {'img': '/416x416/test/'}
}
]

dataset_prefixes = ['AerialMaritimeDrone', 'Aquarium', 'Cottontail Rabbits', 'EgoHands', 'NorthAmerica Mushrooms', 'Packages', 'Pascal VOC', 'Pothole', 'Raccoon', 'ShellfishOpenImages', 'Thermal Dogs and People', 'Vehicles OpenImages']