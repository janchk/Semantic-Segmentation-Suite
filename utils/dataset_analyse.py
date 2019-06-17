import cv2
import os
import numpy as np
import helpers
import csv


if __name__ == "__main__":
    dataset_path = "../autovision_dataset/"
    dirs = ["train_labels"]
    # dirs = ["test_labels", "train_labels", "val_labels"]
    class_names_list, label_values = helpers.get_label_info(os.path.join(dataset_path, "class_dict.csv"))
    # ds_classes = dict(zip(label_values, class_names_list))
    per_class_coverage = [0]*class_names_list.__len__()
    os.listdir(dataset_path)
    for dir in dirs:
        labels_path = os.path.join(dataset_path, dir)
        imgs_list = os.listdir(labels_path)
        for num_imgs, img_path in enumerate(imgs_list):
            label_img = cv2.imread(os.path.join(dataset_path, dir, img_path))
            all_px = label_img.size / 3
#            label_img = cv2.cvtColor(label_img, cv2.COLOR_BGR2RGB)
            for num, value in enumerate(label_values):
                pixel_counter = np.count_nonzero(np.all(label_img == value, axis=2))
                percentage_of_color = pixel_counter / all_px * 100
                per_class_coverage[num] += percentage_of_color/imgs_list.__len__()
                # print("percentage of class %s is %f" % (class_names_list[num], percentage_of_color))

        split_dict = dict(zip(class_names_list,per_class_coverage))
        with open(dataset_path + "%s.csv" %dir, "w") as csf:
            for key in split_dict.keys():
                csf.write("%s,%s\n" % (key, split_dict[key]))



