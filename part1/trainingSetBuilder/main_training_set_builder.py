import getopt
import sys
from part1.trainingSetBuilder import training_set_builder

def main(argv):
    unix_options = "ho:d:r:s:t:"
    gnu_options = ["help", "output_dir=", "dataset_name=", "relevance_label_number=", "test_set_size=", "query_id_sample_threshold="]
    args1 = []

    try:
        arguments, values = getopt.getopt(argv, unix_options, gnu_options)
    except getopt.error as err:
        # output error, and return with an error code
        print(str(err))
        sys.exit(2)
    for currentArgument, currentValue in arguments:
        if currentArgument in ("-h", "--help"):
            print("'main_training_set_builder.py -o <output_dir> -d <dataset_name> -r <relevance_label_number> -s <test_set_size> -t <query_id_sample_threshold>'")
            sys.exit(1)
        elif currentArgument in ("-o", "--output_dir"):
            args1.append(currentValue)
        elif currentArgument in ("-d", "--dataset_name"):
            args1.append(currentValue)
        elif currentArgument in ("-r", "--relevance_label_number"):
            args1.append(currentValue)
        elif currentArgument in ("-s", "--test_set_size"):
            args1.append(int(currentValue))
        elif currentArgument in ("-t", "--query_id_sample_threshold"):
            args1.append(int(currentValue))

    training_set_builder.training_set_builder(*args1)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main(sys.argv[1:])