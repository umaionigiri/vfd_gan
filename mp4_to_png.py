
import cv2
import glob
import os

def main():

    root_list = glob.glob("../data/fullcolor_mod_dataset/*")
    save_root = "/media/ohshiro/HD-PNFU3/fullcolor_mod_dataset_png/"
    mask = False

    for root in root_list:
        data_list = glob.glob(os.path.join(root+"/*"))
        name1 = root.rsplit("/", 1)[-1]

        for data in data_list:
            video_list = glob.glob(os.path.join(data+"/*"))

            for video in video_list:
                count = 0
                video_name = video.rsplit("/", 1)[-1]
                video_name_ = video_name.rsplit(".", 1)[0]
                name2 = video_name_.rsplit("]", 1)[-1]
                if 'Fake' in video_name_:
                    mask = False
                    file_name = 'inpaint'
                elif 'Original' in video_name_:
                    mask = False
                    file_name = 'original'
                elif 'Mask' in video_name_:
                    mask = True
                    file_name = 'mask'
                save_path = os.path.join(save_root, name1, name2, file_name)
                print("\n read_path == {}".format(video))
                print("save_path == {} ".format(save_path))
                if not os.path.exists(save_path): os.makedirs(save_path)
                cap = cv2.VideoCapture(video)

                while True:
                    ret, frame = cap.read()
                    if ret == True:
                        count += 1
                        if mask:
                            frame = cv2.bitwise_not(frame)
                        cv2.imwrite(os.path.join(save_path, str("{0:06d}".format(count)+".png")), frame)
                    else: break
                
if __name__ == "__main__":
    main()
