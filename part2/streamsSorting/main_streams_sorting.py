import getopt
import sys
from part2.streamsSorting import streams_sorting

def main(argv):
    unix_options = "hd:n:s:"
    gnu_options = ["help", "dataset_path=", "dataset_name=", "highest_streams_dataset_path="]
    args1 = []

    try:
        arguments, values = getopt.getopt(argv, unix_options, gnu_options)
    except getopt.error as err:
        # output error, and return with an error code
        print(str(err))
        sys.exit(2)
    for currentArgument, currentValue in arguments:
        if currentArgument in ("-h", "--help"):
            print("'main_streams_sorting.py -d <dataset_path> -n <dataset_name> -s <highest_streams_dataset_path>'")
            sys.exit(1)
        elif currentArgument in ("-d", "--dataset_path"):
            args1.append(currentValue)
        elif currentArgument in ("-n", "--dataset_name"):
            args1.append(currentValue)
        elif currentArgument in ("-s", "--highest_streams_dataset_path"):
            args1.append(currentValue)

    streams_sorting.streams_sorting(*args1)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main(sys.argv[1:])