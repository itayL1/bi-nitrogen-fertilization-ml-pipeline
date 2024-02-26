from bi_nitrogen_fertilization_ml_pipeline.core.data_classes.base_model import BaseModel


class OneHotEncodingMetadata(BaseModel):
    OTHER_CATEGORY = 'other'
    categories_ordered_by_relative_offset = list[str]

    # def get_category_offset(self, category: str) -> int:
    #     if category == self.OTHER_CATEGORY:
    #         return len(self.categories_ordered_by_relative_offset)
    #     else:
    #         try:
    #             return self.categories_ordered_by_relative_offset.index(category)
    #         except ValueError as ex:
    #             raise Exception(f"the category '{category}' is unknown") from ex


class CategoricalFeatureEncoding(BaseModel):
    one_hot_encoded_features: dict[str, OneHotEncodingMetadata]


class TrainingArtifacts(BaseModel):
    pass
