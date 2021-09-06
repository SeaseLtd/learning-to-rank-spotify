import getopt
import sys
from part1.dataPreprocessing import preprocessing

def main(argv):
    unix_options = "ho:d:e:"
    gnu_options = ["help", "output_dir=", "dataset_path=", "encoding="]
    args1 = []

    try:
        arguments, values = getopt.getopt(argv, unix_options, gnu_options)
    except getopt.error as err:
        # output error, and return with an error code
        print(str(err))
        sys.exit(2)
    for currentArgument, currentValue in arguments:
        if currentArgument in ("-h", "--help"):
            print("'main_preprocessing.py -o <output_dir> -d <dataset_path> -e <encoding>'")
            sys.exit(1)
        elif currentArgument in ("-o", "--output_dir"):
            args1.append(currentValue)
        elif currentArgument in ("-d", "--dataset_path"):
            args1.append(currentValue)
        elif currentArgument in ("-e", "--encoding"):
            args1.append(currentValue)

    preprocessing.preprocessing(*args1)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main(sys.argv[1:])