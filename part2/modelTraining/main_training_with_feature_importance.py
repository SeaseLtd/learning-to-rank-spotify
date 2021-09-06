import getopt
import sys
from part2.modelTraining import training_with_feature_importance

def main(argv):
    unix_options = "ho:t:s:n:e:i:f:"
    gnu_options = ["help", "output_dir=","training_set=", "test_set=", "model_name=", "eval_metric=", "images_path=",
                   "feature_to_analyze="]
    args1 = []

    try:
        arguments, values = getopt.getopt(argv, unix_options, gnu_options)
    except getopt.error as err:
        # output error, and return with an error code
        print(str(err))
        sys.exit(2)
    for currentArgument, currentValue in arguments:
        if currentArgument in ("-h", "--help"):
            print("'main_training_with_feature_importance.py -o <output_dir> -t <training_set> -s <test_set> -n <model_name> -e <eval_metric> -i <images_path> -f <feature_to_analyze>'")
            sys.exit(1)
        elif currentArgument in ("-o", "--output_dir"):
            args1.append(currentValue)
        elif currentArgument in ("-t", "--training_set"):
            args1.append(currentValue)
        elif currentArgument in ("-s", "--test_set"):
            args1.append(currentValue)
        elif currentArgument in ("-n", "--model_name"):
            args1.append(currentValue)
        elif currentArgument in ("-e", "--eval_metric"):
            args1.append(currentValue)
        elif currentArgument in ("-i", "--images_path"):
            args1.append(currentValue)
        elif currentArgument in ("-f", "--feature_to_analyze"):
            args1.append(currentValue)

    training_with_feature_importance.train_model(*args1)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main(sys.argv[1:])