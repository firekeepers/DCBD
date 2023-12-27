import os


# file = os.listdir(r'C:\Users\China\Desktop\make-your-yolov5_dataset-main\dataset_source\voc\ImageSets\Main\rename\1')
root = r'C:\Users\China\Desktop\make-your-yolov5_dataset-main\dataset_source\voc\ImageSets\Main\rename\1\trainval.txt'
dis = r'C:\Users\China\Desktop\make-your-yolov5_dataset-main\dataset_source\voc\ImageSets\Main\rename\1\123.txt'

file = open(root, 'r')
for line in file:
    print(line)
    line_new = line.split('.')
    print(line_new[0])
    f2 = open(dis,'a')
    f2.write(str(line_new[0])+'\n')
    #
    # with open(dis, 'w') as f2:
    #     f2.write(line[0])