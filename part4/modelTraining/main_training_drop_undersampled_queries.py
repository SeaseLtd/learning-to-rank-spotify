import getopt
import sys
from part4.modelTraining import training_drop_undersampled_queries


def main(argv):
    unix_options = "ho:t:s:u:n:e:"
    gnu_options = ["help", "output_dir=","training_set=", "test_set=", "under_sampled_only_train=", "model_name=", "eval_metric="]
    args1 = []

    try:
        arguments, values = getopt.getopt(argv, unix_options, gnu_options)
    except getopt.error as err:
        # output error, and return with an error code
        print(str(err))
        sys.exit(2)
    for currentArgument, currentValue in arguments:
        if currentArgument in ("-h", "--help"):
            print("'main_training_drop_undersampled_queries.py -o <output_dir> -t <training_set> -s <test_set> -u <under_sampled_only_train> -n <model_name> -e <eval_metric>'")
            sys.exit(1)
        elif currentArgument in ("-o", "--output_dir"):
            args1.append(currentValue)
        elif currentArgument in ("-t", "--training_set"):
            args1.append(currentValue)
        elif currentArgument in ("-s", "--test_set"):
            args1.append(currentValue)
        elif currentArgument in ("-u", "--under_sampled_only_train"):
            args1.append(currentValue)
        elif currentArgument in ("-n", "--model_name"):
            args1.append(currentValue)
        elif currentArgument in ("-e", "--eval_metric"):
            args1.append(currentValue)

    training_drop_undersampled_queries.train_model(*args1)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main(sys.argv[1:])