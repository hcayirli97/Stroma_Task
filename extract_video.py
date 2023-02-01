# In this script, video images(.mp4) and annotation files(.json) are extracted in yolo format.
import cv2, os, json

if not os.path.exists("dataset"): # The file to extract the image and annotation files has been created.
    os.mkdir("dataset") 

def convert_dataset(type, save_dir): # Function that saves the data in the annotation file together with the images in the video in yolo format.
    if not os.path.exists(save_dir+"/"+type):
        os.mkdir(save_dir+"/"+type)
    with open("challenge/annotations/instances_" + type + ".json", "r") as f:
        coco_anns = json.load(f)

    cap = cv2.VideoCapture("challenge/images/" + type + "/" + type + ".mp4")

    for img in coco_anns["images"]:
        img_id = img['id']
        file_name = img['file_name']
        ret, frame = cap.read()
        image_h, image_w, _ = frame.shape
        flag = False
        f = open(save_dir + "/" + type + "/" + file_name.replace("jpg","txt"), "w")
        for anno in coco_anns['annotations']:
            if anno["image_id"] == img_id:
                flag = True
                x,y,w,h = anno['bbox']
                c = anno['category_id']-1
                x,y,w,h = int(x), int(y), int(w), int(h)
                xc, yc = x + w/2 , y + h/2
                f.write("{} {:.6f} {:.6f} {:.6f} {:.6f}\n".format(int(c), xc/image_w, yc/image_h, w/image_w, h/image_h))
        if not flag:
            f.write("")
        f.close()
        cv2.imwrite(save_dir+"/"+type+"/"+ file_name, frame)
    cap.release()


            
convert_dataset("train","dataset")
convert_dataset("val","dataset")
convert_dataset("test","dataset")

