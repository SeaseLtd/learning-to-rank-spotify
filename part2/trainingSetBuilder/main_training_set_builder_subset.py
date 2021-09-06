import getopt
import sys
from part2.trainingSetBuilder import training_set_builder_subset

def main(argv):
    unix_options = "ho:"
    gnu_options = ["help", "output_dir="]
    args1 = []

    try:
        arguments, values = getopt.getopt(argv, unix_options, gnu_options)
    except getopt.error as err:
        # output error, and return with an error code
        print(str(err))
        sys.exit(2)
    for currentArgument, currentValue in arguments:
        if currentArgument in ("-h", "--help"):
            print("'main_training_set_builder_subset.py -o <output_dir>'")
            sys.exit(1)
        elif currentArgument in ("-o", "--output_dir"):
            args1.append(currentValue)

    training_set_builder_subset.training_set_builder_subset(*args1)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main(sys.argv[1:])