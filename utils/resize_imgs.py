import sys, os, shutil
from PIL import Image

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Error: %s <input_dir> <output_dir> <scale>" % sys.argv[0])
        exit()
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    scale = float(sys.argv[3])

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    image_suffix = ['.jpg', '.png']
    for name in os.listdir(input_dir):
        img_path = os.path.join(input_dir, name)
        out_path = os.path.join(output_dir, name)
        
        if not os.path.isfile(img_path):
            print(img_path)
            continue
        if name.lower()[-4:] not in image_suffix:
            continue

        im = Image.open(img_path).convert('RGB')
        width, height = im.size
        im = im.resize((int(width/scale), int(height/scale)), Image.ANTIALIAS)
        im.save(out_path)
        
        print("Resize %s (%d x %d) => %s (%d x %d)" % (img_path, width, height, out_path, int(width/scale), int(height/scale)))
