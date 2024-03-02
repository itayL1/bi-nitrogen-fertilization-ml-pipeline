from keras.optimizers import Adam

from bi_nitrogen_fertilization_ml_pipeline.core.data_classes.evaluation_functions import EvaluationFunctions

# categorical features settings defaults
MIN_SIGNIFICANT_CATEGORY_PERCENTAGE_THRESHOLD = 3.0
MAX_ALLOWED_CATEGORIES_COUNT_PER_FEATURE = 15
ALLOW_UNKNOWN_CATEGORIES_DURING_INFERENCE = False

# train params defaults
LOSS_FUNCTION = EvaluationFunctions.mse
EVALUATION_METRIC = EvaluationFunctions.rmse
ADAM_OPTIMIZER = Adam
