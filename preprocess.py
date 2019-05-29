import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET
import argparse

def xml_to_csv(path):
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            value = (root.find('filename').text,
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     member[0].text,
                     int(member[4][0].text),
                     int(member[4][1].text),
                     int(member[4][2].text),
                     int(member[4][3].text)
                     )
            xml_list.append(value)
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df


def main():
    # Initiate argument parser
    parser = argparse.ArgumentParser(
        description="Annonation to-CSV converter")
    parser.add_argument("-i",
                        "--inputPath",
                        help="Path to image dir",
                        type=str)
    parser.add_argument("-o",
                        "--outputPath",
                        help="Name of csv file output dir",
                        type=str)

    args = parser.parse_args()

    assert(os.path.isdir(args.inputPath))
    assert(os.path.isdir(args.outputPath))

    xml_df = xml_to_csv(args.inputPath)
    train_labels = os.path.join(args.outputPath, 'train_labels.csv')
    xml_df.to_csv(train_labels, index=None)
    print('Train labels saved at: {}'.format(train_labels))


if __name__ == '__main__':
    main()