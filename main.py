import configurations
from handwriting import train
from handwriting import validate


if __name__ == "__main__":

    conf = configurations.get_args()

    # TRAINING
    if conf.train_unconditional:
        train.train_unconditional(conf)

    if conf.train_conditional:
        train.train_conditional(conf)

    # VALIDATION/PREDICTION/GENERATION
    if conf.validate_unconditional:
        validate.validate_unconditional(conf)

    if conf.validate_conditional:
        validate.validate_conditional(conf)
