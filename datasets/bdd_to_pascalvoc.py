import os
import json
import xml.etree.ElementTree as ET
from PIL import Image

# Paths to BDD100K dataset and output Pascal VOC format directory
bdd_images_train_dir = 'bdd100k_images/train'
bdd_images_test_dir = 'bdd100k_images/val'
bdd_labels_dir = 'bdd100k_det_20_labels_trainval/labels'
output_dir = 'VOC2007'

# Ensure the Pascal VOC directories exist
os.makedirs(os.path.join(output_dir, 'Annotations'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'JPEGImages'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'ImageSets', 'Main'), exist_ok=True)

# Create val.txt file for ImageSets/Main (only for daytime images)
imageset_path = os.path.join(output_dir, 'ImageSets', 'Main')
train_file_rainy = open(os.path.join(imageset_path, 'train_t_bddrain.txt'), 'w')
train_file_snowy = open(os.path.join(imageset_path, 'train_t_bddsnow.txt'), 'w')
train_file_foggy = open(os.path.join(imageset_path, 'train_t_bddfog.txt'), 'w')

test_file_rainy = open(os.path.join(imageset_path, 'test_bddrain.txt'), 'w')
test_file_snowy = open(os.path.join(imageset_path, 'test_bddsnow.txt'), 'w')
test_file_foggy = open(os.path.join(imageset_path, 'test_bddfog.txt'), 'w')


cityscapes_classes = ('person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle')
bdd100k_classes = ('pedestrian', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle', 'traffic light', 'traffic sign', 'other vehicle')

def create_voc_xml_annotation(image_info, objects, output_xml_file):
    annotation = ET.Element('annotation')
    ET.SubElement(annotation, 'folder').text = 'VOC2012'
    ET.SubElement(annotation, 'filename').text = image_info['filename']
    ET.SubElement(annotation, 'path').text = os.path.join(output_dir, 'JPEGImages', image_info['filename'])

    source = ET.SubElement(annotation, 'source')
    ET.SubElement(source, 'database').text = 'The BDD100K Database'

    size = ET.SubElement(annotation, 'size')
    ET.SubElement(size, 'width').text = str(image_info['width'])
    ET.SubElement(size, 'height').text = str(image_info['height'])
    ET.SubElement(size, 'depth').text = str(image_info['depth'])

    for obj in objects:
        try:
            idx = bdd100k_classes.index(obj['category'])
        except:
            continue
        if idx > 7 : continue
        new_category = cityscapes_classes[idx]
        object_elem = ET.SubElement(annotation, 'object')
        ET.SubElement(object_elem, 'name').text = new_category
        ET.SubElement(object_elem, 'pose').text = 'Unspecified'
        ET.SubElement(object_elem, 'truncated').text = '0'
        ET.SubElement(object_elem, 'difficult').text = '0'
        
        bndbox = ET.SubElement(object_elem, 'bndbox')
        ET.SubElement(bndbox, 'xmin').text = str(obj['bbox'][0])
        ET.SubElement(bndbox, 'ymin').text = str(obj['bbox'][1])
        ET.SubElement(bndbox, 'xmax').text = str(obj['bbox'][2])
        ET.SubElement(bndbox, 'ymax').text = str(obj['bbox'][3])

    tree = ET.ElementTree(annotation)
    tree.write(output_xml_file)

global_index = 0
# train -> train_t: Process each JSON file in BDD100K labels
for json_file in os.listdir(bdd_labels_dir):
    # Only use det_train.json
    if 'train' not in json_file: continue
    
    with open(os.path.join(bdd_labels_dir, json_file), 'r') as f:
        json_data = json.load(f)
    
    for idx, label_data in enumerate(json_data):
        # Extract time and weather attributes
        time_of_day = label_data['attributes']['timeofday']
        weather = label_data['attributes']['weather'].replace(' ', '_')
        image_filename = label_data['name']
        image_index = os.path.splitext(image_filename)[0].zfill(6)

        if time_of_day not in ['daytime', 'night']: continue
        if weather not in ['rainy', 'snowy','foggy']: continue
        if 'labels' not in label_data.keys(): continue
        
        original_image_path = os.path.join(bdd_images_train_dir, image_filename)
        if not os.path.isfile(original_image_path): continue
        print("valid!")
        global_index_str = str(global_index).zfill(6)
        
        # Create new filename format
        new_filename = f"train_{time_of_day}_{weather}_{global_index_str}"
        global_index += 1

        # Copy and rename the image file
        new_image_path = os.path.join(output_dir, 'JPEGImages', f"{new_filename}.jpg")
        os.system(f"cp {original_image_path} {new_image_path}")

        # Create Pascal VOC XML annotation file
        image = Image.open(new_image_path)
        image_info = {
            'filename': f"{new_filename}.jpg",
            'width': image.width,
            'height': image.height,
            'depth': 3
        }

        objects = []
        try:
            for obj in label_data['labels']:
                if 'box2d' in obj:
                    objects.append({
                        'category': obj['category'],
                        'bbox': [
                            int(obj['box2d']['x1']),
                            int(obj['box2d']['y1']),
                            int(obj['box2d']['x2']),
                            int(obj['box2d']['y2'])
                        ]
                    })
        except:
            
        output_xml_file = os.path.join(output_dir, 'Annotations', f"{new_filename}.xml")
        create_voc_xml_annotation(image_info, objects, output_xml_file)

        # Add to ImageSets/Main/val.txt only if the image was taken during daytime
        if time_of_day == 'daytime' and weather =='rainy':
            train_file_rainy.write(f"{new_filename}\n")
        elif time_of_day == 'daytime' and weather =='snowy':
            train_file_snowy.write(f"{new_filename}\n")
        elif time_of_day == 'daytime' and weather =='foggy':
            train_file_foggy.write(f"{new_filename}\n")
            
train_file_rainy.close()
train_file_snowy.close()
train_file_foggy.close()

      
            
            
            
# val -> test: Process each JSON file in BDD100K labels
for json_file in os.listdir(bdd_labels_dir):
    # Only use det_val.json
    if 'val' not in json_file: continue
    
    with open(os.path.join(bdd_labels_dir, json_file), 'r') as f:
        json_data = json.load(f)
        
    for idx, label_data in enumerate(json_data):
        # Extract time and weather attributes
        time_of_day = label_data['attributes']['timeofday']
        weather = label_data['attributes']['weather'].replace(' ', '_')
        image_filename = label_data['name']
        image_index = os.path.splitext(image_filename)[0].zfill(6)

        if time_of_day not in ['daytime', 'night']: continue
        if weather not in ['rainy', 'snowy','foggy']: continue
        if 'labels' not in label_data.keys(): continue
        
        original_image_path = os.path.join(bdd_images_test_dir, image_filename)
        if not os.path.isfile(original_image_path): continue
        
        global_index_str = str(global_index).zfill(6)
        
        # Create new filename format
        new_filename = f"val_{time_of_day}_{weather}_{global_index_str}"
        global_index += 1

        # Copy and rename the image file
        new_image_path = os.path.join(output_dir, 'JPEGImages', f"{new_filename}.jpg")
        os.system(f"cp {original_image_path} {new_image_path}")

        # Create Pascal VOC XML annotation file
        image = Image.open(new_image_path)
        image_info = {
            'filename': f"{new_filename}.jpg",
            'width': image.width,
            'height': image.height,
            'depth': 3
        }

        objects = []
        for obj in label_data['labels']:
            if 'box2d' in obj:
                objects.append({
                    'category': obj['category'],
                    'bbox': [
                        int(obj['box2d']['x1']),
                        int(obj['box2d']['y1']),
                        int(obj['box2d']['x2']),
                        int(obj['box2d']['y2'])
                    ]
                })

        output_xml_file = os.path.join(output_dir, 'Annotations', f"{new_filename}.xml")
        create_voc_xml_annotation(image_info, objects, output_xml_file)

        # Add to ImageSets/Main/val.txt only if the image was taken during daytime
        if time_of_day == 'daytime' and weather =='rainy':
            test_file_rainy.write(f"{new_filename}\n")
        elif time_of_day == 'daytime' and weather =='snowy':
            test_file_snowy.write(f"{new_filename}\n")
        elif time_of_day == 'daytime' and weather =='foggy':
            test_file_foggy.write(f"{new_filename}\n")
            
test_file_rainy.close()
test_file_snowy.close()
test_file_foggy.close()