from PIL import Image
import os


def check_integrity(image_path):
    try:
        img = Image.open(image_path).convert('RGB')
        img.verify()
        return True
    except Exception as e:
        print(f"Corrupted image: {image_path}, {str(e)}")
        return False

if __name__ == '__main__':
    for root, dirs, files in os.walk('../data/images/images'):
        for file in files:
            image_path = os.path.join(root,file)
            print(image_path)
            if check_integrity(image_path=image_path) is False:
                print("Removing image")
                os.remove(image_path)