from tests.utils.test_datasets import load_Nitrogen_with_Era5_and_NDVI_dataset


def test_feature_extraction_e2e():
    raw_train_dataset_df = load_Nitrogen_with_Era5_and_NDVI_dataset()
    print(raw_train_dataset_df.shape)
